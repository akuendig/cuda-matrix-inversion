#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include <cblas.h>
#ifdef __APPLE__
    #include <lapacke.h>
#else
    #include <clapack.h>
#endif // __APPLE__

#include "../include/types.h"
#include "../include/timer.h"
#include "../include/helper_cpu.h"
#include "../include/helper_gpu.h"
#include "../include/inverse_cpu.h"
#include "../include/inverse_gpu.h"

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

__global__ void add(const DataType alpha, const Array devLeft[], Array devRight[], int batchSize, int n)
{
    for(int i = 0; i < n; i++)
    {
        devRight[blockIdx.x][threadIdx.x*n+i] =
            alpha*devRight[blockIdx.x][threadIdx.x*n+i] +
            devLeft[blockIdx.x][threadIdx.x*n+i];
    }
}  /* add */

// Adds all matrices in devLeft to their corresponding matrix in devRight.
// The data inside devRight is modified, devLeft is left untouched.
// Both devLeft and devRight are expected to be already allocated on the GPU.
// defRight += devLeft
static void batchedAdd(
    cublasHandle_t handle,
    int n,
    const DataType *alpha,
    const Array devLeft[],
    Array devRight[],
    int batchSize) {
    // TODO: implement addition. Can also be done on the CPU but then we
    // need to do it before transferring the data to the GPU.
    // SEE: http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-axpy
    // SEE: http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-geam

    add<<<batchSize, n>>> (*alpha, devLeft, devRight, batchSize, n);
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
        const_cast<const float**>(devRight), n,
        beta, devResult, k,
        batchSize)
    );
}

// Calculates the mean of the matrix set {A, B, C, D}.
// Mean = A^{T} * (B+C)^{-1} * D
// As       batchSize x n x 1
// Bs       batchSize x n x n
// Cs       batchSize x n x 1 diagonal
// Ds       batchSize x n x 1
// Means    batchSize x 1 x 1
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
static void calcluateMeanDev(
    cublasHandle_t handle,
    int n,
    Array devAs[],
    Array devBs[],
    Array devCs[],
    Array devDs[],
    Array devMeans[],
    int batchSize) {


}

// Calculates the mean of the matrix set {A, B, C, D}.
// Mean = A*(B+C)^{-1}*D
// As       batchSize x n x 1
// Bs       batchSize x n x n
// Cs       batchSize x n x 1
// Ds       batchSize x n x 1
// Means    batchSize x n x 1
// Means is assumed to be already allocated.
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
    Array *devCs;
    size_t pitchCs;
    Array *devDs;
    size_t pitchDs;

    const size_t sizeOfMatrixA = sizeof(DataType)*n;
    const size_t sizeOfMatrixB = sizeof(DataType)*n*n;
    const size_t sizeOfMatrixC = sizeof(DataType)*n*n;
    const size_t sizeOfMatrixD = sizeof(DataType)*n;
    const size_t sizeOfResult = sizeof(DataType);

    gpuErrchk( cudaHostAlloc(&devBs, sizeof(Array)*batchSize, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc(&devCs, sizeof(Array)*batchSize, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc(&devDs, sizeof(Array)*batchSize, cudaHostAllocDefault) );

    // Allocate and copy Bs, Cs and Ds to the GPU
    gpuErrchk( batchedCudaMalloc(devBs, &pitchBs, sizeOfMatrixB, batchSize) );
    gpuErrchk( batchedCudaMalloc(devCs, &pitchCs, sizeOfMatrixC, batchSize) );
    gpuErrchk( batchedCudaMalloc(devDs, &pitchDs, sizeOfMatrixD, batchSize) );

    gpuErrchk( cudaMemcpy2D(devBs[0], pitchBs, Bs, sizeOfMatrixB, sizeOfMatrixB, batchSize,
               cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy2D(devCs[0], pitchCs, Cs, sizeOfMatrixC, sizeOfMatrixC, batchSize,
               cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy2D(devDs[0], pitchDs, Ds, sizeOfMatrixD, sizeOfMatrixD, batchSize,
               cudaMemcpyHostToDevice) );

    // Calculate Madd = B + C for every matrix, store result in Cs
    batchedAdd(handle, n, &ELEMENT_ONE, devBs, devCs, batchSize);
    // devBs: Bs
    // devCs: Madd
    // devDs: Ds

    // Calculate Minv = Madd^-1, store result in Bs
    batchedInverse(handle, n, devCs, devBs, batchSize);
    // devBs: Minv
    // devCs: Madd
    // devDs: Ds

    // Calculate Mmul = Minv * Ds, store result in Cs
    batchedMul(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, 1, &ELEMENT_ONE, devBs, devDs, &ELEMENT_ZERO, devCs, batchSize);
    // devBs: Minv
    // devCs: Mmul
    // devDs: Ds

    // Load As into GPU memory overwriting devDs.
    gpuErrchk( cudaMemcpy2D(devDs[0], pitchDs, As, sizeOfMatrixA, sizeOfMatrixA, batchSize,
               cudaMemcpyHostToDevice) );
    // devBs: Minv
    // devCs: Mmul
    // devDs: As

    // Calculate Mmean = AT * Mmul + (whatever is in Bs), store result in Bs
    batchedMul(handle, CUBLAS_OP_T, CUBLAS_OP_N, 1, n, n, &ELEMENT_ONE, devCs, devDs, &ELEMENT_ZERO, devBs, batchSize);
    // devBs: Mmean
    // devCs: Mmul
    // devDs: As

    // Fetch result from GPU and free used memory.
    gpuErrchk( cudaMemcpy2D(Means, sizeOfResult, devBs, pitchBs, sizeOfResult, batchSize,
               cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaFree(devBs[0]) );
    gpuErrchk( cudaFree(devCs[0]) );
    gpuErrchk( cudaFree(devDs[0]) );

    gpuErrchk( cudaFreeHost((void*)devBs) );
    gpuErrchk( cudaFreeHost((void*)devCs) );
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
    Array *devCs;
    size_t pitchCs;

    const size_t sizeOfMatrixA = sizeof(DataType)*n;
    const size_t sizeOfMatrixB = sizeof(DataType)*n*n;
    const size_t sizeOfMatrixC = sizeof(DataType)*n*n;
    const size_t sizeOfMatrixE = sizeof(DataType);

    gpuErrchk( cudaHostAlloc((void**)&devAs, sizeof(Array)*batchSize, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc((void**)&devBs, sizeof(Array)*batchSize, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc((void**)&devCs, sizeof(Array)*batchSize, cudaHostAllocDefault) );

    // Allocate and copy Bs, Cs and As to the GPU
    gpuErrchk( batchedCudaMalloc(devAs, &pitchAs, sizeOfMatrixA, batchSize) );
    gpuErrchk( batchedCudaMalloc(devBs, &pitchBs, sizeOfMatrixB, batchSize) );
    gpuErrchk( batchedCudaMalloc(devCs, &pitchCs, sizeOfMatrixC, batchSize) );

    gpuErrchk( cudaMemcpy2D(devAs[0], pitchAs, As, sizeOfMatrixA, sizeOfMatrixA, batchSize,
               cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy2D(devBs[0], pitchBs, Bs, sizeOfMatrixB, sizeOfMatrixB, batchSize,
               cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy2D(devCs[0], pitchCs, Cs, sizeOfMatrixC, sizeOfMatrixC, batchSize,
               cudaMemcpyHostToDevice) );

    // Calculate Madd = B + C for every matrix, store result in Cs
    batchedAdd(handle, n, &ELEMENT_ONE, devBs, devCs, batchSize);
    // devAs: As
    // devBs: Bs
    // devCs: Madd

    // Calculate Minv = Madd^-1, store result in Bs
    batchedInverse(handle, n, devCs, devBs, batchSize);
    // devAs: As
    // devBs: Minv
    // devCs: Madd

    // Calculate Mmul = Minv * A + (whatever is in Cs), store result in Cs
    batchedMul(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, 1, &ELEMENT_ONE, devBs, devAs, &ELEMENT_ZERO, devCs, batchSize);
    // devAs: As
    // devBs: Minv
    // devCs: Mmul

    // Calculate Mmul2 = AT * Mmul + (whatever is in Bs), store result in Bs
    batchedMul(handle, CUBLAS_OP_T, CUBLAS_OP_N, 1, n, n, &ELEMENT_ONE, devCs, devAs, &ELEMENT_ZERO, devBs, batchSize);
    // devAs: As
    // devBs: Mmul2
    // devCs: Mmul

    // Load Es to the GPU overwriting As
    gpuErrchk( cudaMemcpy2D(devAs[0], pitchAs, Es, sizeOfMatrixE, sizeOfMatrixE, batchSize,
               cudaMemcpyHostToDevice) );

    const DataType minusOne = DataType(-1);
    batchedAdd(handle, n, &minusOne, devBs, devAs, batchSize);

    // Fetch result from GPU and free used memory.
    gpuErrchk( cudaMemcpy2D(Variances, sizeOfMatrixE, devAs[0], pitchAs, sizeOfMatrixE, batchSize,
               cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaFree((void*)devAs[0]) );
    gpuErrchk( cudaFree((void*)devBs[0]) );
    gpuErrchk( cudaFree((void*)devCs[0]) );

    gpuErrchk( cudaFreeHost((void*)devAs) );
    gpuErrchk( cudaFreeHost((void*)devBs) );
    gpuErrchk( cudaFreeHost((void*)devCs) );
}

// Calculates the mean of the matrix set {A, B, C, D}.
// Mean = A*(B+C)^{-1}*D
// As       batchSize x n x 1
// Bs       batchSize x n x n
// Cs       batchSize x n x 1
// Ds       batchSize x n x 1
// Means    batchSize x n x 1
// Means is assumed to be already allocated.
static void calcluateMeanCPU(
    const int n,
    Array As,
    Array Bs,
    Array Cs,
    Array Ds,
    Array Means,
    const int batchSize) {

    int i, j;

    Array workspace = (Array)malloc(sizeof(DataType)*n*n);
    ensure(workspace, "Could not allocate workspace for matrix inversion");

    for (i = 0; i < batchSize; ++i) {
        Array currentA = As+(i*n);
        Array currentB = Bs+(i*n*n);
        Array currentC = Cs+(i*n);
        Array currentD = Ds+(i*n);

        // Update diagonal
        for (j = 0; j < n; ++j) {
            currentB[j + j*n] += currentC[j];
        }

        // inverse_lu_blas(currentB, workspace, n);
        inverse_chol_blas(currentB, n);

        cblas_ssymv (CblasColMajor, CblasUpper,
            n, // rows in A
            1, // alpha
            currentB, // A
            n, // LDA
            currentD, // x
            1, // inc x
            0, // beta
            currentC, // y
            1 // inc y
        );

        Means[i] = cblas_sdot (
            n, // rows in x
            currentA, // x
            1, // inc x
            currentC, // y
            1 // inc y
        );
    }

    free(workspace);
}

// Calculates the variance of the matrix set {A, B, C, E}.
// Var = E-AT*(B+C)^{-1}*A
// As       batchSize x n x 1
// Bs       batchSize x n x n
// Cs       batchSize x n x 1
// Es       batchSize x 1 x 1
// Variances    batchSize x 1 x 1
// Variances is assumed to be already allocated.
//
// Bs and Cs are destroyed
static void calcluateVarianceCPU(
    int n,
    Array As,
    Array Bs,
    Array Cs,
    Array Es,
    Array Variances,
    int batchSize) {

    int i, j;

    Array workspace = (Array)malloc(sizeof(DataType)*n*n);
    ensure(workspace, "Could not allocate workspace for matrix inversion");

    for (i = 0; i < batchSize; ++i) {
        Array currentA = As+(i*n);
        Array currentB = Bs+(i*n*n);
        Array currentC = Cs+(i*n);

        // Update diagonal
        for (j = 0; j < n; ++j) {
            currentB[j + j*n] += currentC[j];
        }

        // inverse_lu_blas(currentB, workspace, n);
        inverse_chol_blas(currentB, n);

        cblas_ssymv (CblasColMajor, CblasUpper,
            n, // rows in A
            1, // alpha
            currentB, // A
            n, // LDA
            currentA, // x
            1, // inc x
            0, // beta
            currentC, // y
            1 // inc y
        );

        Variances[i] = Es[i] + cblas_sdot (
            n, // rows in x
            currentA, // x
            1, // inc x
            currentC, // y
            1 // inc y
        );
    }

    free(workspace);
}


void readTest(const char *directory, int *numMatrices, int *n,
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
    cblas_saxpy(N, -1.f, a, 1, b, N);
}

static DataType vec_sum(Array a, const int N) {
    return cblas_sasum(N, a, 1);
}

#define BENCH_VAR(name) \
    double total_error_means_##name = 0; \
    double total_error_variances_##name = 0; \
    TIMER_INIT(means_##name) \
    TIMER_INIT(variances_##name)

#define BENCH_SETUP() \
    cblas_scopy(numMatrices*n, _a, 1, a, 1); \
    cblas_scopy(numMatrices*n*n, _b, 1, b, 1); \
    cblas_scopy(numMatrices*n, _c, 1, c, 1); \
    cblas_scopy(numMatrices*n, _d, 1, d, 1); \
    cblas_scopy(numMatrices, _e, 1, e, 1);

#define BENCH_ERROR_MEAN(name) \
    vec_diff(means_out, _means, n); \
    total_error_means_##name += vec_sum(means_out, n);

#define BENCH_ERROR_VARIANCE(name) \
    vec_diff(variances_out, _variances, n); \
    total_error_variances_##name += vec_sum(variances_out, n);

#define BENCH_CLEANUP(name)

#define BENCH_REPORT_ERROR(name) \
    printf("Total error in means calculation for %d matrices of " #name ": %.2e (%.2e average)\n", \
        numMatrices, total_error_means_##name, total_error_means_##name/numMatrices/numReps); \
    printf("Total error in variances calculation for %d matrices of " #name ": %.2e (%.2e average)\n", \
        numMatrices, total_error_variances_##name, total_error_variances_##name/numMatrices/numReps);

#ifdef __APPLE__
#define BENCH_REPORT_TIME(name) \
    printf("Total execution time in means for %d matrices and %d replications of " #name ": %lu cycles (%lu cycles average)\n", \
        numMatrices, numReps, timer_total_means_##name, timer_total_means_##name/numMatrices/numReps); \
    printf("Total execution time in variances for %d matrices and %d replications of " #name ": %lu cycles (%lu cycles average)\n", \
        numMatrices, numReps, timer_total_variances_##name, timer_total_variances_##name/numMatrices/numReps);
#else
#define BENCH_REPORT_TIME(name) \
    printf("Total execution time in means for %d matrices and %d replications of " #name ": %.4f ms (%.4f ms average)\n", \
        numMatrices, numReps, time_to_ms(&timer_total_means_##name), time_to_ms(&timer_total_means_##name)/numMatrices/numReps); \
    printf("Total execution time in variances for %d matrices and %d replications of " #name ": %.4f ms (%.4f ms average)\n", \
        numMatrices, numReps, time_to_ms(&timer_total_variances_##name), time_to_ms(&timer_total_variances_##name)/numMatrices/numReps);
#endif // __APPLE__

int main(int argc, char const *argv[]) {
    int numMatrices, n, rep, numReps;
    Array a, b, c, d, e,
        _a, _b, _c, _d, _e, _means, _variances;
    Array means_out, variances_out;

    ensure(argc >= 3, "Usage: gauss_bench TEST_FOLDER NUM_REPLICATIONS [-d]");

    bool detailed = (argc >= 4) && !strncmp("-d", argv[3], 2);

    numReps = atoi(argv[2]);

    // cublasHandle_t handle;

    // cublasErrchk( cublasCreate(&handle) );
    // gpuErrchk( cudaHostAlloc(&means, sizeof(DataType)*numMatrices, cudaHostAllocDefault) ); \

    readTest(argv[1], &numMatrices, &n, &_a, &_b, &_c, &_d, &_e, &_means, &_variances);

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

    for (rep = 0; rep < numReps; ++rep) {
        BENCH_SETUP()
        TIMER_START(means_cpu)
        calcluateMeanCPU(n, a, b, c, d, means_out, numMatrices);
        TIMER_STOP(means_cpu)
        TIMER_ACC(means_cpu)
        TIMER_LOG(means_cpu)
        BENCH_ERROR_MEAN(cpu)
        BENCH_CLEANUP()


        BENCH_SETUP()
        TIMER_START(variances_cpu)
        calcluateVarianceCPU(n, a, b, c, e, variances_out, numMatrices);
        TIMER_STOP(variances_cpu)
        TIMER_ACC(variances_cpu)
        TIMER_LOG(variances_cpu)
        BENCH_ERROR_VARIANCE(cpu)
        BENCH_CLEANUP()
    }

    BENCH_REPORT_ERROR(cpu)
    BENCH_REPORT_TIME(cpu)

    // gpuErrchk( cudaPeekAtLastError() );
    // gpuErrchk( cudaDeviceSynchronize() );

    free(a); free(b); free(c); free(d); free(e);
    free(_a); free(_b); free(_c); free(_d); free(_e); free(_means); free(_variances);
    free(means_out); free(variances_out);

    // gpuErrchk( cudaFreeHost(means) );
    // cublasErrchk( cublasDestroy(handle) );

    // cudaDeviceReset();

    return 0;
}
