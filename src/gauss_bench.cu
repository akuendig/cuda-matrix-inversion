#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include <cblas.h>
#include <lapacke.h>

#include "../include/types.h"
#include "../include/helper_cpu.h"
#include "../include/helper_gpu.h"
#include "../include/timer.h"
#include "../include/inverse_cpu.h"
#include "../include/inverse_gpu.h"
#include "../include/gauss_cpu.h"

static const DataType ELEMENT_ZERO = DataType(0);
static const DataType ELEMENT_ONE = DataType(1);

// Memory model
// ************
//
// All matrices are allocated on a continouse portion of memory.
//
// PRO
// + Easy to transfer to and from the GPU as it is one memcpy
// + Allocation goes a lot quicker (needs profiling) since memory can be allocated in one portion
//   and we do not need to loop to allocate memory on the GPU
//
// CON
// - We need to generate the array of pointers to use the xxxBatched APIs
// - Some memory access in the kernels will be missaligned which can hamper performance (needs profiling)
//      > Seems like this can be avoided with cudaMallocPitch and cudaMemcpy2D

__global__ void addDiagonal(Array devLeft[], Array devRight[], int batchSize, int n)
{
    devLeft[blockIdx.x][threadIdx.x*n + threadIdx.x] =
        devLeft[blockIdx.x][threadIdx.x*n + threadIdx.x] +
        devRight[blockIdx.x][threadIdx.x];
}  /* add */

// Adds all matrices in devLeft to their corresponding matrix in devRight.
// The data inside devRight is modified, devLeft is left untouched.
// Both devLeft and devRight are expected to be already allocated on the GPU.
// devLeft += devRight
static void batchedAddDiagonal(
    cublasHandle_t handle,
    int n,
    Array devLeft[],
    Array devRight[],
    int batchSize) {
    // TODO: implement addition. Can also be done on the CPU but then we
    // need to do it before transferring the data to the GPU.
    // SEE: http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-axpy
    // SEE: http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-geam

    addDiagonal<<<batchSize, n>>> (devLeft, devRight, batchSize, n);
}

// Inverts all matrices in devMatrices and stores the result in devInvMatrices.
// devMatrices and devInvMatrices are already allocated on the GPU. However,
// maybe one of the two methods for inversion does not need a workspace the size
// of the input. In that case this function signature has to change!
// defInvMatrices = devMatrices^{-1}
static void batchedInverse(
    cublasHandle_t handle,
    int n,
    Array devMatrices[],
    Array devInvMatrices[],
    int batchSize) {
    // TODO: implement matrix inversion. Please see how you want to dispatch to the corresponding inversion algorithm.
    // SEE: http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-getrfbatched
    // SEE: http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-getribatched
    inverse_lu_cuda_batched_device(handle, n, devMatrices, devInvMatrices, batchSize);
}

// Multiplies all matrices in devLeft to their corresponding matrix in devRight.
// The data inside devLeft and devRight is untouched, devResult is modified.
// devLeft, devRight and devResult are expected to be already allocated on the GPU.
// m = number of rows of transa(devLeft) and devResult
// n = number of columns of transb(devRight) and devResult
// k = number of columns of transa(devLeft) and rows of transb(devRight)
// devResult = alpha*transa(devLeft)*transb(devRight) + beta*devResult
static void batchedMul(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const DataType *alpha,
    Array devLeft[],
    Array devRight[],
    const DataType *beta,
    Array devResult[],
    int batchSize) {
    // TODO: implement matrix multiplication.
    // SEE: http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemmbatched
    cublasErrchk( cublasSgemmBatched(handle,
        transa, transb,
        m, n, k,
        alpha, const_cast<const float**>(devLeft), m,
        const_cast<const float**>(devRight), k,
        beta, devResult, m,
        batchSize)
    );
}

// Calculates the mean of the matrix set {A, B, C, D}.
// Mean = A*(B+C)^{-1}*D
// As       batchSize x n x 1
// Bs       batchSize x n x n
// Cs       batchSize x n x 1
// Ds       batchSize x n x 1
// Means    batchSize x n x 1
// Means is assumed to be already allocated.
// Total memory requirement without optimization
// batchSize x n x (2n+3)
// --> 8x8:     608 bytes/calculation       => 5298068 calculations can live on the GPU
// --> 16x16:   2'240 bytes/calculation     => 1438047 calculations can live on the GPU
// --> 32x32:   8'576 bytes/calculation     => 375609 calculations can live on the GPU
// --> 64x64:   33'536 bytes/calculation    => 96052 calculations can live on the GPU
// --> 128x128: 132'608 bytes/calculation   => 24291 calculations can live on the GPU
// --> 256x256: 527'360 bytes/calculation   => 6108 calculations can live on the GPU
// --> 512x512: 2'103'296 bytes/calculation => 1531 calculations can live on the GPU
// 2n because of workspace required
static void calcluateMean(
    cublasHandle_t handle,
    int n,
    Array As,
    Array Bs,
    Array Cs,
    Array Ds,
    Array Means,
    int batchSize) {

    Array *devBs;
    size_t pitchBs;
    Array *devBInvs;
    size_t pitchBInvs;
    Array *devDs;
    size_t pitchDs;

    const size_t sizeOfMatrixA = sizeof(DataType)*n;
    const size_t sizeOfMatrixB = sizeof(DataType)*n*n;
    const size_t sizeOfMatrixC = sizeof(DataType)*n;
    const size_t sizeOfMatrixD = sizeof(DataType)*n;
    const size_t sizeOfResult = sizeof(DataType);

    // Allocate pointer list
    gpuErrchk( cudaHostAlloc(&devBs, sizeof(Array)*batchSize, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc(&devBInvs, sizeof(Array)*batchSize, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc(&devDs, sizeof(Array)*batchSize, cudaHostAllocDefault) );

    // Allocate and copy Bs, Cs and Ds to the GPU
    gpuErrchk( batchedCudaMalloc(devBs, &pitchBs, sizeOfMatrixB, batchSize) );
    gpuErrchk( batchedCudaMalloc(devBInvs, &pitchBInvs, sizeOfMatrixB, batchSize) );
    gpuErrchk( batchedCudaMalloc(devDs, &pitchDs, sizeOfMatrixD, batchSize) );

    gpuErrchk( cudaMemcpy2D(devBs[0], pitchBs, Bs, sizeOfMatrixB, sizeOfMatrixB, batchSize,
               cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy2D(devBInvs[0], pitchBInvs, Cs, sizeOfMatrixC, sizeOfMatrixC, batchSize,
               cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy2D(devDs[0], pitchDs, Ds, sizeOfMatrixD, sizeOfMatrixD, batchSize,
               cudaMemcpyHostToDevice) );

    // Calculate Madd = B + C for every matrix, store result in Bs
    batchedAddDiagonal(handle, n, devBs, devBInvs, batchSize);
    // devBs: Bs
    // devBInvs: Madd
    // devDs: Ds

    // Calculate Minv = Madd^-1, store result in devBInvs
    batchedInverse(handle, n, devBs, devBInvs, batchSize);
    // devBs: Madd
    // devBInvs: Minv
    // devDs: Ds

    // Calculate Mmul = Minv * Ds, store result in Cs
    batchedMul(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, 1, n, &ELEMENT_ONE, devBInvs, devDs, &ELEMENT_ZERO, devBs, batchSize);
    // devBs: Mmul
    // devBInvs: Minv
    // devDs: Ds

    // Load As into GPU memory overwriting devDs.
    gpuErrchk( cudaMemcpy2D(devDs[0], pitchDs, As, sizeOfMatrixA, sizeOfMatrixA, batchSize,
               cudaMemcpyHostToDevice) );
    // devBs: Mmul
    // devBInvs: Minv
    // devDs: As

    // Calculate Mmean = AT * Mmul, store result in Bs
    batchedMul(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, n, &ELEMENT_ONE, devBs, devDs, &ELEMENT_ZERO, devBInvs, batchSize);
    // devBs: Mmul
    // devBInvs: Mmean
    // devDs: As

    // Fetch result from GPU and free used memory.
    gpuErrchk( cudaMemcpy2D(Means, sizeOfResult, devBInvs[0], pitchBInvs, sizeOfResult, batchSize,
               cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaFree(devBs[0]) );
    gpuErrchk( cudaFree(devBInvs[0]) );
    gpuErrchk( cudaFree(devDs[0]) );

    gpuErrchk( cudaFreeHost((void*)devBs) );
    gpuErrchk( cudaFreeHost((void*)devBInvs) );
    gpuErrchk( cudaFreeHost((void*)devDs) );
}

// Calculates the variance of the matrix set {A, B, C, E}.
// Var = E-AT*(B+C)^{-1}*A
// As       batchSize x n x 1
// Bs       batchSize x n x n
// Cs       batchSize x n x n
// Es       batchSize x 1 x 1
// Variances    batchSize x 1 x 1
// Variances is assumed to be already allocated.
static void calcluateVariance(
    cublasHandle_t handle,
    int n,
    Array As,
    Array Bs,
    Array Cs,
    Array Es,
    Array Variances,
    int batchSize) {

    Array *devAs;
    size_t pitchAs;
    Array *devBs;
    size_t pitchBs;
    Array *devBInvs;
    size_t pitchBInvs;

    const size_t sizeOfMatrixA = sizeof(DataType)*n;
    const size_t sizeOfMatrixB = sizeof(DataType)*n*n;
    const size_t sizeOfMatrixC = sizeof(DataType)*n;
    const size_t sizeOfMatrixE = sizeof(DataType);

    gpuErrchk( cudaHostAlloc((void**)&devAs, sizeof(Array)*batchSize, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc((void**)&devBs, sizeof(Array)*batchSize, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc((void**)&devBInvs, sizeof(Array)*batchSize, cudaHostAllocDefault) );

    // Allocate and copy Bs, Cs and As to the GPU
    gpuErrchk( batchedCudaMalloc(devAs, &pitchAs, sizeOfMatrixA, batchSize) );
    gpuErrchk( batchedCudaMalloc(devBs, &pitchBs, sizeOfMatrixB, batchSize) );
    gpuErrchk( batchedCudaMalloc(devBInvs, &pitchBInvs, sizeOfMatrixB, batchSize) );

    gpuErrchk( cudaMemcpy2D(devAs[0], pitchAs, As, sizeOfMatrixA, sizeOfMatrixA, batchSize,
               cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy2D(devBs[0], pitchBs, Bs, sizeOfMatrixB, sizeOfMatrixB, batchSize,
               cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy2D(devBInvs[0], pitchBInvs, Cs, sizeOfMatrixC, sizeOfMatrixC, batchSize,
               cudaMemcpyHostToDevice) );

    // Calculate Madd = B + C for every matrix, store result in devBs
    batchedAddDiagonal(handle, n, devBs, devBInvs, batchSize);
    // devAs: As
    // devBs: Bs
    // devBInvs: Madd

    // Calculate Minv = Madd^-1, store result in devBInvs
    batchedInverse(handle, n, devBs, devBInvs, batchSize);
    // devAs: As
    // devBs: Madd
    // devBInvs: Minv

    // Calculate Mmul = Minv * A, store result in devBs
    batchedMul(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, 1, n, &ELEMENT_ONE, devBInvs, devAs, &ELEMENT_ZERO, devBs, batchSize);
    // devAs: As
    // devBs: Mmul
    // devBInvs: Minv

    // Load Es to the GPU overwriting devBInvs
    gpuErrchk( cudaMemcpy2D(devBInvs[0], pitchBInvs, Es, sizeOfMatrixE, sizeOfMatrixE, batchSize,
               cudaMemcpyHostToDevice) );

    const DataType ELEMENT_MINUS_ONE = DataType(-1);
    // Calculate Mmul2 = AT * Mmul, store result in Bs
    batchedMul(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 1, n, &ELEMENT_MINUS_ONE, devAs, devBs, &ELEMENT_ONE, devBInvs, batchSize);
    // devAs: As
    // devBs: Mmul2
    // devBInvs: Mmul

    // Fetch result from GPU and free used memory.
    gpuErrchk( cudaMemcpy2D(Variances, sizeOfMatrixE, devBInvs[0], pitchBInvs, sizeOfMatrixE, batchSize,
               cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaFree((void*)devAs[0]) );
    gpuErrchk( cudaFree((void*)devBs[0]) );
    gpuErrchk( cudaFree((void*)devBInvs[0]) );

    gpuErrchk( cudaFreeHost((void*)devAs) );
    gpuErrchk( cudaFreeHost((void*)devBs) );
    gpuErrchk( cudaFreeHost((void*)devBInvs) );
}

static void readTest(const char *directory, int *numMatrices, int *n,
        Array *a, Array *b, Array *c, Array *d, Array *e, Array *means, Array *variances) {
    char filePath[1024];

    int numMatricesA, numMatricesB, numMatricesC, numMatricesD,
        numMatricesE, numMeans, numVariances;
    int mA, mB, mC, mD, mE, mMeans, mVariances;
    int nA, nB, nC, nD, nE, nMeans, nVariances;

    snprintf(filePath, 1024, "%s/a.mats", directory);
    readMatricesFile(filePath, &numMatricesA, &mA, &nA, a);

    snprintf(filePath, 1024, "%s/b.mats", directory);
    readMatricesFile(filePath, &numMatricesB, &mB, &nB, b);

    snprintf(filePath, 1024, "%s/c.mats", directory);
    readMatricesFile(filePath, &numMatricesC, &mC, &nC, c);

    snprintf(filePath, 1024, "%s/d.mats", directory);
    readMatricesFile(filePath, &numMatricesD, &mD, &nD, d);

    snprintf(filePath, 1024, "%s/e.mats", directory);
    readMatricesFile(filePath, &numMatricesE, &mE, &nE, e);

    snprintf(filePath, 1024, "%s/means.mats", directory);
    readMatricesFile(filePath, &numMeans, &mMeans, &nMeans, means);

    snprintf(filePath, 1024, "%s/variances.mats", directory);
    readMatricesFile(filePath, &numVariances, &mVariances, &nVariances, variances);

    ensure(
        numMatricesA == numMatricesB && numMatricesB == numMatricesC && numMatricesC == numMatricesD &&
        numMatricesD == numMatricesE && numMatricesE == numMeans && numMeans == numVariances,
        "test in directory %s invalid, number of matrices in files not matching\r\n"
        "numMatricesA(%d) numMatricesB(%d) numMatricesC(%d) numMatricesD(%d)\r\n"
        "numMatricesE(%d) numMeans(%d) numVariances(%d)\r\n",
        directory,
        numMatricesA, numMatricesB, numMatricesC, numMatricesD,
        numMatricesE, numMeans, numVariances
    );

    ensure(
        mA == mB && mB == mC && mC == mD && 1 == mE && mMeans == 1 && mVariances == 1 &&
        nA == 1 && nB == mB && nC == 1 && nD == 1 && nE == 1 && nMeans == 1 && nVariances == 1,
        "test in directory %s invalid, dimensions not matching\r\n"
        "mA(%d) mB(%d) mC(%d) mD(%d)\r\n"
        "mE(%d) mMeans(%d) mVariances(%d)\r\n"
        "nA(%d) nB(%d) nC(%d) nD(%d)\r\n"
        "nE(%d) nMeans(%d) nVariances(%d)\r\n",
        directory,
        mA, mB, mC, mD, mE, mMeans, mVariances,
        nA, nB, nC, nD, nE, nMeans, nVariances
    );

    *numMatrices = numMatricesA;
    *n = mA;
}

// b -= a
static void vec_diff(const Array a, Array b, const int N) {
    cblas_saxpy(N, -1.f, a, 1, b, 1);
}

static DataType vec_sum(Array a, const int N) {
    return cblas_sasum(N, a, 1);
}

#define BENCH_VAR(name) \
    double total_error_means_##name = 0; \
    double total_error_variances_##name = 0; \
    TIMER_INIT(means_##name) \
    TIMER_ACC_INIT(means_##name) \
    TIMER_INIT(variances_##name) \
    TIMER_ACC_INIT(variances_##name)

#define BENCH_SETUP() \
    cblas_scopy(numMatrices*n, _a, 1, a, 1); \
    cblas_scopy(numMatrices*n*n, _b, 1, b, 1); \
    cblas_scopy(numMatrices*n, _c, 1, c, 1); \
    cblas_scopy(numMatrices*n, _d, 1, d, 1); \
    cblas_scopy(numMatrices, _e, 1, e, 1);

#define BENCH_ERROR_MEAN(name) \
    vec_diff(means_out, _means, numMatrices); \
    total_error_means_##name += vec_sum(means_out, numMatrices);

#define BENCH_ERROR_VARIANCE(name) \
    vec_diff(variances_out, _variances, numMatrices); \
    total_error_variances_##name += vec_sum(variances_out, numMatrices);

#define BENCH_CLEANUP(name)

#ifndef DETAILED_LOGGING
#define BENCH_REPORT_TIME(name) \
    if (csv) { \
        if (numReps > 1) { \
            printf("%d %d %d means_" #name " %e %e %e %e\n", \
                numMatrices, n, numReps, TIMER_TOTAL(means_##name), TIMER_MEAN(means_##name), TIMER_VARIANCE(means_##name), total_error_means_##name/numMatrices/numReps); \
            printf("%d %d %d variances_" #name " %e %e %e %e\n", \
                numMatrices, n, numReps, TIMER_TOTAL(variances_##name), TIMER_MEAN(variances_##name), TIMER_VARIANCE(means_##name), total_error_variances_##name/numMatrices/numReps); \
        } else { \
            printf("%d %d %d means_" #name " %e %e\n", \
                numMatrices, n, numReps, TIMER_TOTAL(means_##name), total_error_means_##name/numMatrices/numReps); \
            printf("%d %d %d variances_" #name " %e %e\n", \
                numMatrices, n, numReps, TIMER_TOTAL(variances_##name), total_error_variances_##name/numMatrices/numReps); \
        } \
    } else { \
        if (numReps > 1) { \
            printf("means_"#name " - %d %dx%d matrices, replicated %d times, runtime %.4f ms (%.4f ms average, %.4f ms variance), average error %.4e\n", \
                numMatrices, n, n, numReps, TIMER_TOTAL(means_##name), TIMER_MEAN(means_##name), TIMER_VARIANCE(means_##name), total_error_means_##name/numMatrices/numReps); \
            printf("variances_"#name " - %d %dx%d matrices, replicated %d times, runtime %.4f ms (%.4f ms average, %.4f ms variance), average error %.4e\n", \
                numMatrices, n, n, numReps, TIMER_TOTAL(variances_##name), TIMER_MEAN(variances_##name), TIMER_VARIANCE(means_##name), total_error_variances_##name/numMatrices/numReps); \
        } else { \
            printf("means_"#name " - %d %dx%d matrices, replicated %d times, runtime %.4f ms, average error %.4e\n", \
                numMatrices, n, n, numReps, TIMER_TOTAL(means_##name), total_error_means_##name/numMatrices/numReps); \
            printf("variances_"#name " - %d %dx%d matrices, replicated %d times, runtime %.4f ms, average error %.4e\n", \
                numMatrices, n, n, numReps, TIMER_TOTAL(variances_##name), total_error_variances_##name/numMatrices/numReps); \
        } \
    }
#else
#define BENCH_REPORT_TIME(name)
#endif

// Print device properties
void printDevProp(cudaDeviceProp devProp) {
    // Source: https://www.cac.cornell.edu/vw/gpu/example_submit.aspx
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %lu\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %lu\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %lu\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %lu\n",  devProp.totalConstMem);
    printf("Texture alignment:             %lu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ?"Yes" : "No"));
}

void printDeviceInfo() {
    // Source: https://www.cac.cornell.edu/vw/gpu/example_submit.aspx
    int devCount;

    cudaGetDeviceCount(&devCount);

    printf("CUDA Device Query...\n");
    printf("There are %d CUDA devices.\n", devCount);

    for (int i = 0; i < devCount; ++i)
    {
        // Get device properties
        printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        printDevProp(devProp);
    }
}

int main(int argc, char const *argv[]) {
    int numMatrices, n, rep, numReps, numDuplicates;
    Array a, b, c, d, e,
        _a, _b, _c, _d, _e, _means, _variances;
    Array means_out, variances_out;

    ensure(argc >= 4, "Usage: gauss_bench TEST_FOLDER TEST_REPLICATIONS MATRIX_DUPLICATES [-csv]");

    bool csv = (argc >= 5) && !strncmp("-csv", argv[4], 4);

    numReps = atoi(argv[2]);
    numDuplicates = atoi(argv[3]);

    if (!csv) {
        // printDeviceInfo();
    }

    readTest(argv[1], &numMatrices, &n, &_a, &_b, &_c, &_d, &_e, &_means, &_variances);
    replicateMatrices(&_a, n, 1, numMatrices, numDuplicates);
    replicateMatrices(&_b, n, n, numMatrices, numDuplicates);
    replicateMatrices(&_c, n, 1, numMatrices, numDuplicates);
    replicateMatrices(&_d, n, 1, numMatrices, numDuplicates);
    replicateMatrices(&_e, 1, 1, numMatrices, numDuplicates);
    replicateMatrices(&_means, 1, 1, numMatrices, numDuplicates);
    replicateMatrices(&_variances, 1, 1, numMatrices, numDuplicates);

    numMatrices *= numDuplicates;

    a = (Array)malloc(numMatrices*n*sizeof(DataType));
    ensure(a, "Could not allocate memory for A");
    b = (Array)malloc(numMatrices*n*n*sizeof(DataType));
    ensure(b, "Could not allocate memory for B");
    c = (Array)malloc(numMatrices*n*sizeof(DataType));
    ensure(c, "Could not allocate memory for C");
    d = (Array)malloc(numMatrices*n*sizeof(DataType));
    ensure(d, "Could not allocate memory for D");
    e = (Array)malloc(numMatrices*sizeof(DataType));
    ensure(e, "Could not allocate memory for E");
    means_out = (Array)malloc(numMatrices*sizeof(DataType));
    ensure(means_out, "Could not allocate memory for calculated means");
    variances_out = (Array)malloc(numMatrices*sizeof(DataType));
    ensure(variances_out, "Could not allocate memory for calculated variances");

    BENCH_VAR(cpu)
    BENCH_VAR(gpu)

    // CPU Benchmark //
    ///////////////////
    for (rep = 0; rep < numReps; ++rep) {
        BENCH_SETUP()
        TIMER_START(means_cpu)
#ifdef GAUSS_SOLVE
        calcluateMeanSolveCPU(n, a, b, c, d, means_out, numMatrices);
#else
        calcluateMeanCPU(n, a, b, c, d, means_out, numMatrices);
#endif // GAUSS_SOLVE
        TIMER_STOP(means_cpu)
        TIMER_ACC(means_cpu)
#ifdef DETAILED_LOGGING
        TIMER_LOG(means_cpu, numMatrices, n)
#endif // DETAILED_LOGGING
        BENCH_ERROR_MEAN(cpu)
        BENCH_CLEANUP()


        BENCH_SETUP()
        TIMER_START(variances_cpu)
#ifdef GAUSS_SOLVE
        calcluateVarianceSolveCPU(n, a, b, c, e, variances_out, numMatrices);
#else
        calcluateVarianceCPU(n, a, b, c, e, variances_out, numMatrices);
#endif // GAUSS_SOLVE
        TIMER_STOP(variances_cpu)
        TIMER_ACC(variances_cpu)
#ifdef DETAILED_LOGGING
        TIMER_LOG(variances_cpu, numMatrices, n)
#endif // DETAILED_LOGGING
        BENCH_ERROR_VARIANCE(cpu)
        BENCH_CLEANUP()
    }

    // GPU Benchmark //
    ///////////////////
    cublasHandle_t handle;
    cublasErrchk( cublasCreate(&handle) );

    for (rep = 0; rep < numReps; ++rep) {
        BENCH_SETUP()
        TIMER_START(means_gpu)
        calcluateMean(handle, n, a, b, c, d, means_out, numMatrices);
        TIMER_STOP(means_gpu)
        TIMER_ACC(means_gpu)
#ifdef DETAILED_LOGGING
        TIMER_LOG(means_gpu, numMatrices, n)
#endif // DETAILED_LOGGING
        BENCH_ERROR_MEAN(gpu)
        BENCH_CLEANUP()

        BENCH_SETUP()
        TIMER_START(variances_gpu)
        calcluateVariance(handle, n, a, b, c, e, variances_out, numMatrices);
        TIMER_STOP(variances_gpu)
        TIMER_ACC(variances_gpu)
#ifdef DETAILED_LOGGING
        TIMER_LOG(variances_gpu, numMatrices, n)
#endif // DETAILED_LOGGING
        BENCH_ERROR_VARIANCE(gpu)
        BENCH_CLEANUP()
    }

    BENCH_REPORT_TIME(cpu)
    BENCH_REPORT_TIME(gpu)

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    free(a); free(b); free(c); free(d); free(e); free(means_out); free(variances_out);
    free(_a); free(_b); free(_c); free(_d); free(_e); free(_means); free(_variances);

    // gpuErrchk( cudaFreeHost(means) );
    // cublasErrchk( cublasDestroy(handle) );

    // cudaDeviceReset();

    return 0;
}
