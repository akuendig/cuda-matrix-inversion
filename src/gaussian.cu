#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <iostream>

#define ELEMENT_TYPE float

static const ELEMENT_ZERO ELEMENT_TYPE(0);
static const ELEMENT_ONE ELEMENT_TYPE(1);

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

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s:%d\n", cudaGetErrorString(code), file, line);
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
    void *devPtr;

    cudaError_t result = cudaMallocPitch(&devPtr, pitch, arraySize, batchSize);

    if (cudaSuccess != result) {
        return result;
    }

    for (void *devPtrEnd = devPtr + (*pitch)*batchSize; devPtr < devPtrEnd; devPtr += *pitch) {
        devArrayPtr[i] = devPtr;
    }

    return cudaSuccess;
}

// Adds all matrices in devLeft to their corresponding matrix in devRight.
// The data inside devRight is modified, devLeft is left untouched.
// Both devLeft and devRight are expected to be already allocated on the GPU.
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
static void batchedMul(
    cublasHandle_t handle,
    cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const ELEMENT_TYPE *alpha,
    const Array devLeft[],
    const Array devReight[],
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
static void calcluate_mean(
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

    // Calculate Madd = B + C for every matrix, the result is stored in Cs
    batchedAdd(handle, n, &ELEMENT_ONE, devBs, devCs, batchSize);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // devBs: Bs
    // devCs: Madd
    // devDs: Ds

    // Calculate Minv = Madd^-1, reuse Bs as workspace
    batchedInverse(handle, n, devCs, devBs, batchSize);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // devBs: Minv
    // devCs: Madd
    // devDs: Ds

    // Set all first elements of devCs to zero because usually the
    // array of values provided as third arcument to the BLAS
    // multiplication function is added to the result of the computation.
    // However, the result of the computation are vectors and not matrices
    // anymore.
    gpuErrchk( cudaMemset2D(devCs, pitchCs, 0, sizeOfMatrixD, batchSize) );
    // Calculate Mmul = Minv * Ds, reuse Cs as workspace
    batchedMul(handle, n, devBs, devDs, devCs, batchSize);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // devBs: Minv
    // devCs: Mmul
    // devDs: Ds

    // Load As into GPU memory overwriting devDs.
    gpuErrchk( cudaMemcpy2D(devDs, pitchDs, As, sizeOfMatrixA, batchSize,
               cudaMemcpyHostToDevice) );
    // devBs: Minv
    // devCs: Mmul
    // devDs: As

    // Again, set all first elements of devBs to zero because usually the
    // array of values provided as third arcument to the BLAS
    // multiplication function is added to the result of the computation.
    // However, the result of the computation are scalars and not matrices
    // anymore.
    gpuErrchk( cudaMemset2D(devBs, pitchBs, 0, sizeOfResult, batchSize) );
    // Calculate Mmean = Mmul * A
    batchedMul(handle, n, devCs, devDs, devBs, batchSize);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // devBs: Mmean
    // devCs: Mmul
    // devDs: As

    // Fetch result from GPU and free used memory.
    gpuErrchk( cudaMemcpy2D(devBs, pitchBs, Means, sizeOfResult, batchSize,
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
static void calcluate_variance(
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

    // Set all first elements of devCs to zero because usually the
    // array of values provided as third arcument to the BLAS
    // multiplication function is added to the result of the computation.
    // However, the result of the computation are vectors and not matrices
    // anymore.
    gpuErrchk( cudaMemset2D(devCs, pitchCs, 0, sizeOfMatrixA, batchSize) );
    // Calculate Mmul = Minv * A + (whatever is in Cs), store result in Cs
    batchedMul(handle, n, devBs, devDs, devCs, batchSize);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // devAs: As
    // devBs: Minv
    // devCs: Mmul

    // Again, set all first elements of devBs to zero because usually the
    // array of values provided as third arcument to the BLAS
    // multiplication function is added to the result of the computation.
    // However, the result of the computation are scalars and not matrices
    // anymore.
    gpuErrchk( cudaMemset2D(devBs, pitchBs, 0, sizeOfMatrixE, batchSize) );
    // Calculate Mmul2 = AT * Mmul + (whatever is in Bs), store result in Bs
    batchedMul(handle, n, devCs, devDs, devBs, batchSize);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    // devAs: As
    // devBs: Mmul2
    // devCs: Mmul

    // Load Es to the GPU overwriting As
    gpuErrchk( cudaMemcpy2D(devAs, pitchAs, Es, sizeOfMatrixE, sizeOfMatrixE, batchSize,
               cudaMemcpyHostToDevice) );

    const ELEMENT_TYPE minusOne = ELEMENT_TYPE(-1);
    batchedAdd(handle, n, &minusOne, Mmul2, devAs, batchSize)
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Fetch result from GPU and free used memory.
    gpuErrchk( cudaMemcpy2D(devAs, pitchAs, Variances, sizeOfMatrixE, batchSize,
               cudaMemcpyHostToDevice) );

    gpuErrchk( cudaFree((void*)devAs[0]) );
    gpuErrchk( cudaFree((void*)devBs[0]) );
    gpuErrchk( cudaFree((void*)devCs[0]) );

    gpuErrchk( cudaFreeHost((void*)devAs) );
    gpuErrchk( cudaFreeHost((void*)devBs) );
    gpuErrchk( cudaFreeHost((void*)devCs) );
}

int main(void)
{
    unsigned int *d = NULL; int i;
    unsigned int idata[N], odata[N];

    for (i = 0; i < N; i++)
         idata[i] = (unsigned int)i;

    gpuErrchk( cudaMalloc((void**)&d, sizeof(int)*N)) ;
    gpuErrchk( cudaMemcpy(d, idata, sizeof(int)*N,
               cudaMemcpyHostToDevice) );

    bitreverse<<<1, N>>>(d);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    gpuErrchk( cudaMemcpy(odata, d, sizeof(int)*N,
               cudaMemcpyDeviceToHost) );

    for (i = 0; i < N; i++)
        printf("%u -> %u\n", idata[i], odata[i]);

    gpuErrchk( cudaFree((void*)d) );
    return 0;
}