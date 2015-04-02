#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <cuda.h>
#include "cublas_v2.h"

#include "../../include/types.h"
#include "../../include/helper.h"
#include "../../include/inverse.h"

#define SWAP(x, y, z)   ((z) = (x),(x) = (y),(y) = (z))

__global__
void transform_matrix(Array a, Array a_inv, int row, int N) {
    __shared__ DataType scalars[64];
    __shared__ DataType currRowA[64], currRowI[64];

    // store the scalars corresponding to the column 'row'
    scalars[threadIdx.x] = a[row * N + threadIdx.x];
    currRowA[threadIdx.x] = a[threadIdx.x * N + row];
    currRowI[threadIdx.x] = a_inv[threadIdx.x * N + row];
    __syncthreads();

    // No need to transform 'row'th row
    if(threadIdx.x == row)
        return;

    // Each thread transforms row
    for(int i = 0; i < N; i++) {
        a[i * N + threadIdx.x] -= (scalars[threadIdx.x] * currRowA[i]);
        a_inv[i * N + threadIdx.x] -= (scalars[threadIdx.x] * currRowI[i]);
    }
}

__global__
void inverse_gauss_kernel(Array *a, Array *aInv, int N) {
    int row, pivot;
    cublasHandle_t handle;

    cublasCreate(&handle);

    for (row = 0; row < N; ++row) {
        /*cublasErrchk*/( cublasIsamax(handle,
            N - row,            // Number of elements to be searched
            &a[blockIdx.x][(row * N) + row],        // Starting position
            1,              // Increment in words (NOT BYTES)
            &pivot) );            // Maximum element in the row
        int pivotRow = pivot - 1 + row;          // Row number with maximum element (starts with 1)

        // printf("Pivot: %d\nRow: %d\n", pivot, pivotRow);
        if(pivotRow != row) {
            /*cublasErrchk*/( cublasSswap(handle,
                N,              // Nuber of elements to be swapped
                &a[blockIdx.x][row],            // Current pivotRow
                N,              // Increment (becuase of column major)
                &a[blockIdx.x][pivotRow],            // Row with max pivot
                N) );
            /*cublasErrchk*/( cublasSswap(handle,
                N,
                &aInv[blockIdx.x][row],
                N,
                &aInv[blockIdx.x][pivotRow],
                N) );
        }

        DataType scalar = 1/a[blockIdx.x][row * N + row];

        /*cublasErrchk*/( cublasSscal(handle,
            N,
            &scalar,
            &a[blockIdx.x][row],
            N) );
        /*cublasErrchk*/( cublasSscal(handle,
            N,
            &scalar,
            &aInv[blockIdx.x][row],
            N) );

        transform_matrix<<<1, N>>>(a[blockIdx.x], aInv[blockIdx.x], row, N);
    }

    cublasDestroy(handle);
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

extern "C" void inverse_gauss_kernel_gpu(
        cublasHandle_t handle,
        int n,
        Array As,
        Array aInvs,
        int batchSize) {

    int k, i;
    Array *devAs;
    size_t pitchAs;
    Array *devAInvs;
    size_t pitchAInvs;

    const size_t ArraySize = sizeof(DataType) * n * n;

    gpuErrchk( cudaHostAlloc((void**)&devAs, sizeof(Array)*batchSize, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc((void**)&devAInvs, sizeof(Array)*batchSize, cudaHostAllocDefault) );

    gpuErrchk( batchedCudaMalloc(devAs, &pitchAs, ArraySize, batchSize) );
    gpuErrchk( batchedCudaMalloc(devAInvs, &pitchAInvs, ArraySize, batchSize) );

    memset(aInvs, 0, batchSize*ArraySize);

    for (k = 0; k < batchSize; ++k) {
        for (i = 0; i < n; ++i) {
            aInvs[k*n*n + i*n + i] = 1.f;
        }
    }

    gpuErrchk( cudaMemcpy2D(devAs[0], pitchAs, As, ArraySize, ArraySize, batchSize,
                cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy2D(devAInvs[0], pitchAInvs, aInvs, ArraySize, ArraySize, batchSize,
                cudaMemcpyHostToDevice) );

    inverse_gauss_kernel<<<batchSize, 1>>>(devAs, devAInvs, n);

    gpuErrchk( cudaMemcpy2D(aInvs, ArraySize, devAInvs[0], pitchAInvs, ArraySize, batchSize,
                cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree((void*)devAs[0]) );
    gpuErrchk( cudaFree((void*)devAInvs[0]) );
    gpuErrchk( cudaFreeHost((void*)devAs) );
    gpuErrchk( cudaFreeHost((void*)devAInvs) );
}
