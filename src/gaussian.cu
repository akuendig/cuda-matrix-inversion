#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define MAX_MATRIX_BYTE_READ 67108864

#define ELEMENT_TYPE float

static const ELEMENT_TYPE ELEMENT_ZERO = ELEMENT_TYPE(0);
static const ELEMENT_TYPE ELEMENT_ONE = ELEMENT_TYPE(1);

typedef ELEMENT_TYPE *Array;

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

#define fail(...) \
  fprintf(stderr, "%s:%d\t", __FILE__, __LINE__); \
  fprintf(stderr, __VA_ARGS__); \
  fprintf(stderr, "\r\n"); \
  exit(EXIT_FAILURE);

#define ensure(condition, ...) \
  do { \
    if (! (condition)) { \
      fprintf(stderr, "ENSURE FAILED %s:%d\r\n", __FILE__, __LINE__); \
      fprintf(stderr, __VA_ARGS__); \
      fprintf(stderr, "\r\n"); \
      if (errno) { perror("possible reason for failure from ERRNO"); } \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s:%d\n", cudaGetErrorString(code), file, line);
        cudaDeviceReset();
        if (abort) { exit(code); }
    }
}

// Simple 8-bit bit reversal Compute test

#define N 256

__global__ void bitreverse(unsigned int *data)
{
    unsigned int *idata = data;

    unsigned int x = idata[threadIdx.x];

    x = ((0xf0f0f0f0 & x) >> 4) | ((0x0f0f0f0f & x) << 4);
    x = ((0xcccccccc & x) >> 2) | ((0x33333333 & x) << 2);
    x = ((0xaaaaaaaa & x) >> 1) | ((0x55555555 & x) << 1);

    idata[threadIdx.x] = x;
}

// Allocates one continous array of memory of size arraySize*batchSize and writes the
// pointers of all subarrays into the array of pointers located at devArrayPtr.
static cudaError_t batchedCudaMalloc(Array* devArrayPtr, size_t *pitch, size_t arraySize, int batchSize) {
    char *devPtr;

    cudaError_t result = cudaMallocPitch((void**)&devPtr, pitch, arraySize, batchSize);

    if (cudaSuccess != result) {
        return result;
    }

    for (int i = 0; i < batchSize; ++i) {
        devArrayPtr[i] = (Array)devPtr;
        devPtr += *pitch;
    }

    return cudaSuccess;
}

// Adds all matrices in devLeft to their corresponding matrix in devRight.
// The data inside devRight is modified, devLeft is left untouched.
// Both devLeft and devRight are expected to be already allocated on the GPU.
// defRight += devLeft
static void batchedAdd(
    cublasHandle_t handle,
    int n,
    const ELEMENT_TYPE *alpha,
    const Array devLeft[],
    Array devRight[],
    int batchSize) {
    // TODO: implement addition. Can also be done on the CPU but then we
    // need to do it before transferring the data to the GPU.
    // SEE: http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-axpy
    // SEE: http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-geam
}

// Inverts all matrices in devMatrices and stores the result in devInvMatrices.
// devMatrices and devInvMatrices are already allocated on the GPU. However,
// maybe one of the two methods for inversion does not need a workspace the size
// of the input. In that case this function signature has to change!
// defInvMatrices = devMatrices^{-1}
static void batchedInverse(
    cublasHandle_t handle,
    int n,
    const Array devMatrices[],
    Array devInvMatrices[],
    int batchSize) {
    // TODO: implement matrix inversion. Please see how you want to dispatch to the corresponding inversion algorithm.
    // SEE: http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-getrfbatched
    // SEE: http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-getribatched
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
    const ELEMENT_TYPE *alpha,
    const Array devLeft[],
    const Array devReight[],
    const ELEMENT_TYPE *beta,
    Array devResult[],
    int batchSize) {
    // TODO: implement matrix multiplication.
    // SEE: http://docs.nvidia.com/cuda/cublas/index.html#cublas-lt-t-gt-gemmbatched
}

// Calculates the mean of the matrix set {A, B, C, D}.
// Mean = A*(B+C)^{-1}*D
// As       batchSize x n x 1
// Bs       batchSize x n x n
// Cs       batchSize x n x n
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

    const size_t sizeOfMatrixA = sizeof(ELEMENT_TYPE)*n;
    const size_t sizeOfMatrixB = sizeof(ELEMENT_TYPE)*n*n;
    const size_t sizeOfMatrixC = sizeof(ELEMENT_TYPE)*n*n;
    const size_t sizeOfMatrixD = sizeof(ELEMENT_TYPE)*n;
    const size_t sizeOfResult = sizeof(ELEMENT_TYPE);

    gpuErrchk( cudaHostAlloc((void**)&devBs, sizeof(Array)*batchSize, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc((void**)&devCs, sizeof(Array)*batchSize, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc((void**)&devDs, sizeof(Array)*batchSize, cudaHostAllocDefault) );

    // Allocate and copy Bs, Cs and Ds to the GPU
    gpuErrchk( batchedCudaMalloc(devBs, &pitchBs, sizeOfMatrixB, batchSize) );
    gpuErrchk( batchedCudaMalloc(devCs, &pitchCs, sizeOfMatrixC, batchSize) );
    gpuErrchk( batchedCudaMalloc(devDs, &pitchDs, sizeOfMatrixD, batchSize) );

    gpuErrchk( cudaMemcpy2D(devBs, pitchBs, Bs, sizeOfMatrixB, sizeOfMatrixB, batchSize,
               cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy2D(devCs, pitchCs, Cs, sizeOfMatrixC, sizeOfMatrixC, batchSize,
               cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy2D(devDs, pitchDs, Ds, sizeOfMatrixD, sizeOfMatrixD, batchSize,
               cudaMemcpyHostToDevice) );

    // Calculate Madd = B + C for every matrix, store result in Cs
    batchedAdd(handle, n, &ELEMENT_ONE, devBs, devCs, batchSize);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // devBs: Bs
    // devCs: Madd
    // devDs: Ds

    // Calculate Minv = Madd^-1, store result in Bs
    batchedInverse(handle, n, devCs, devBs, batchSize);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // devBs: Minv
    // devCs: Madd
    // devDs: Ds

    // Calculate Mmul = Minv * Ds, store result in Cs
    batchedMul(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, 1, &ELEMENT_ONE, devBs, devDs, &ELEMENT_ZERO, devCs, batchSize);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // devBs: Minv
    // devCs: Mmul
    // devDs: Ds

    // Load As into GPU memory overwriting devDs.
    gpuErrchk( cudaMemcpy2D(devDs, pitchDs, As, sizeOfMatrixA, sizeOfMatrixA, batchSize,
               cudaMemcpyHostToDevice) );
    // devBs: Minv
    // devCs: Mmul
    // devDs: As

    // Calculate Mmean = AT * Mmul + (whatever is in Bs), store result in Bs
    batchedMul(handle, CUBLAS_OP_T, CUBLAS_OP_N, 1, n, n, &ELEMENT_ONE, devCs, devDs, &ELEMENT_ZERO, devBs, batchSize);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // devBs: Mmean
    // devCs: Mmul
    // devDs: As

    // Fetch result from GPU and free used memory.
    gpuErrchk( cudaMemcpy2D(devBs, pitchBs, Means, sizeOfResult, sizeOfResult, batchSize,
               cudaMemcpyHostToDevice) );

    gpuErrchk( cudaFree((void*)devBs[0]) );
    gpuErrchk( cudaFree((void*)devCs[0]) );
    gpuErrchk( cudaFree((void*)devDs[0]) );

    gpuErrchk( cudaFreeHost((void*)devBs) );
    gpuErrchk( cudaFreeHost((void*)devCs) );
    gpuErrchk( cudaFreeHost((void*)devDs) );
}

// Calculates the variance of the matrix set {A, B, C, D, E}.
// Var = E-AT*(B+C)^{-1}*A
// As       batchSize x n x 1
// Bs       batchSize x n x n
// Cs       batchSize x n x n
// Ds       batchSize x n x 1
// Means    batchSize x n x 1
// Means is assumed to be already allocated.
static void calcluateVariance(
    cublasHandle_t handle,
    int n,
    Array As,
    Array Bs,
    Array Cs,
    Array Ds,
    Array Es,
    Array Variances,
    int batchSize) {

    Array *devAs;
    size_t pitchAs;
    Array *devBs;
    size_t pitchBs;
    Array *devCs;
    size_t pitchCs;

    const size_t sizeOfMatrixA = sizeof(ELEMENT_TYPE)*n;
    const size_t sizeOfMatrixB = sizeof(ELEMENT_TYPE)*n*n;
    const size_t sizeOfMatrixC = sizeof(ELEMENT_TYPE)*n*n;
    const size_t sizeOfMatrixE = sizeof(ELEMENT_TYPE);

    gpuErrchk( cudaHostAlloc((void**)&devAs, sizeof(Array)*batchSize, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc((void**)&devBs, sizeof(Array)*batchSize, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc((void**)&devCs, sizeof(Array)*batchSize, cudaHostAllocDefault) );

    // Allocate and copy Bs, Cs and As to the GPU
    gpuErrchk( batchedCudaMalloc(devAs, &pitchAs, sizeOfMatrixA, batchSize) );
    gpuErrchk( batchedCudaMalloc(devBs, &pitchBs, sizeOfMatrixB, batchSize) );
    gpuErrchk( batchedCudaMalloc(devCs, &pitchCs, sizeOfMatrixC, batchSize) );

    gpuErrchk( cudaMemcpy2D(devAs, pitchAs, As, sizeOfMatrixA, sizeOfMatrixA, batchSize,
               cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy2D(devBs, pitchBs, Bs, sizeOfMatrixB, sizeOfMatrixB, batchSize,
               cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy2D(devCs, pitchCs, Cs, sizeOfMatrixC, sizeOfMatrixC, batchSize,
               cudaMemcpyHostToDevice) );

    // Calculate Madd = B + C for every matrix, store result in Cs
    batchedAdd(handle, n, &ELEMENT_ONE, devBs, devCs, batchSize);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // devAs: As
    // devBs: Bs
    // devCs: Madd

    // Calculate Minv = Madd^-1, store result in Bs
    batchedInverse(handle, n, devCs, devBs, batchSize);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // devAs: As
    // devBs: Minv
    // devCs: Madd

    // Calculate Mmul = Minv * A + (whatever is in Cs), store result in Cs
    batchedMul(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, 1, &ELEMENT_ONE, devBs, devAs, &ELEMENT_ZERO, devCs, batchSize);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // devAs: As
    // devBs: Minv
    // devCs: Mmul

    // Calculate Mmul2 = AT * Mmul + (whatever is in Bs), store result in Bs
    batchedMul(handle, CUBLAS_OP_T, CUBLAS_OP_N, 1, n, n, &ELEMENT_ONE, devCs, devAs, &ELEMENT_ZERO, devBs, batchSize);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // devAs: As
    // devBs: Mmul2
    // devCs: Mmul

    // Load Es to the GPU overwriting As
    gpuErrchk( cudaMemcpy2D(devAs, pitchAs, Es, sizeOfMatrixE, sizeOfMatrixE, batchSize,
               cudaMemcpyHostToDevice) );

    const ELEMENT_TYPE minusOne = ELEMENT_TYPE(-1);
    batchedAdd(handle, n, &minusOne, devBs, devAs, batchSize);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Fetch result from GPU and free used memory.
    gpuErrchk( cudaMemcpy2D(devAs, pitchAs, Variances, sizeOfMatrixE, sizeOfMatrixE, batchSize,
               cudaMemcpyHostToDevice) );

    gpuErrchk( cudaFree((void*)devAs[0]) );
    gpuErrchk( cudaFree((void*)devBs[0]) );
    gpuErrchk( cudaFree((void*)devCs[0]) );

    gpuErrchk( cudaFreeHost((void*)devAs) );
    gpuErrchk( cudaFreeHost((void*)devBs) );
    gpuErrchk( cudaFreeHost((void*)devCs) );
}

void readMatricesFile(const char *path, int *numMatrices, int *m, int *n, Array *matrices) {
    int ret;
    int _numMatrics, _m, _n;

    FILE* fp = fopen(path, "r");
    ensure(fp, "could not open matrix file %s", path);

    ret = fscanf(fp, "%d %d %d", &_numMatrices, &_m, &_n);
    ensure(3 == ret, "could not read number of matrices from file %s", path);

    *numMatrices = _numMatrices;
    *m = _m;
    *n = _n;

    size_t arraySize = sizeof(ELEMENT_TYPE) * (_numMatrices) * (_m) * (_n);
    ensure(arraySize <= MAX_MATRIX_BYTE_READ, "cannot read file %s because "
        "the allocated array would be bigger than 0x%X bytes", path, arraySize);

    *matrices = malloc(arraySize);
    ensure(*matrices, "could not allocate 0x%X bytes of memory for file %s", arraySize, path);

    Array currentElement = *matrices;

    for (int k = 0; k < _numMatrices; ++k) {
        for (int i = 0; i < _m; ++i) {
            for (int j = 0; j < _n; ++j, ++currentElement) {
                ret = fscanf(fp, "%d", currentElement);
                ensure(ret, "could not read matrix from file %s, stuck at matrix %d element %d, %d", path, k, i, j);
            }
        }
    }

    fclose(fp);
}

void readMeanTest(const char *directory, int *numMatrices, int *n,
        Array *a, Array *b, Array *c, Array *d) {
    char filePath[1024];

    int numMatricesA, numMatricesB, numMatricesC, numMatricesD;
    int mA, mB, mC, mD;
    int nA, nB, nC, nD;
    Array a, b, c, d;

    snprintf(filePath, 1024, "%s/a.mats", directory);
    readMatricesFile(filePath, &numMatricesA, &mA, &nA, &a);

    snprintf(filePath, 1024, "%s/b.mats", directory);
    readMatricesFile(filePath, &numMatricesB, &mB, &nB, &b);

    snprintf(filePath, 1024, "%s/c.mats", directory);
    readMatricesFile(filePath, &numMatricesC, &mC, &nC, &c);

    snprintf(filePath, 1024, "%s/d.mats", directory);
    readMatricesFile(filePath, &numMatricesD, &mD, &nD, &d);

    ensure(
        mA == mB && mB == mC && mC == mD &&
        nA == 1 && nB == mB && nC == mC && nD == 1,
        "test in directory %s invalid, dimensions not matching\r\n"
        "mA(%d) mB(%d) mC(%d) mD(%d)\r\n",
        "nA(%d) nB(%d) nC(%d) nD(%d)\r\n",
        mA, mB, mC, mD, nA, nB, nC, nD
    );
}

int main(void)
{
    cublasHandle_t handle;

    int numMatrices, n;
    Array a, b, c, d;
    Array means;

    readMeanTest("tests/simpleMean", &numMatrices, &n, &a, &b, &c, *d);

    gpuErrchk( cublasCreate(&handle) );
    gpuErrchk( cudaMalloc((void**)&means, sizeof(ELEMENT_TYPE)*numMatrices)) ;


    calcluateMean(handle, n, a, b, c, d, means, numMatrices);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    for (int i = 0; i < numMatrices; i++) {
        printf("%f\r\n", means[i]);
    }

    gpuErrchk( cudaFree((void*)means) );
    gpuErrchk( cublasDestroy(&handle) );

    return 0;
}
