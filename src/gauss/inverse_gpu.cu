#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda.h>
#include "cublas_v2.h"

#include "../../include/types.h"
#include "../../include/timer.h"
#include "../../include/helper_cpu.h"
#include "../../include/helper_gpu.h"
#include "../../include/inverse_cpu.h"
#include "../../include/inverse_gpu.h"

__global__
void identity(Array a, int M, int N) {
    int i, j;

    i = blockIdx.x * blockDim.x + threadIdx.x;
    j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < N) {
        a[i + j*M] = DataType(i==j);
    }
}

__global__
void transform_matrix(Array a, Array a_inv, int row, int N) {
    extern __shared__ DataType shared[];

    DataType *scalars = &shared[0];
    DataType *currRowA = &shared[N];
    DataType *currRowI = &shared[2 * N];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx >= N)
        return;

    // store the scalars corresponding to the column 'row'
    scalars[idx] = a[row * N + idx];
    currRowA[idx] = a[idx * N + row];
    currRowI[idx] = a_inv[idx * N + row];
    __syncthreads();

    // No need to transform 'row'th row
    if(idx == row)
        return;

    // Each thread transforms row
    for(int i = 0; i < N; i++) {
        a[i * N + idx] -= (scalars[idx] * currRowA[i]);
        a_inv[i * N + idx] -= (scalars[idx] * currRowI[i]);
    }
}

__global__
void inverse_gauss_kernel(Array *a, Array *aInv, int N, int batchSize) {
    int row, pivot;
    cublasHandle_t handle;

    cublasCreate(&handle);

    dim3 threadsPerBlock(min(16, N), min(16, N), 1);
    dim3 numBlocks(div_ceil(N, threadsPerBlock.x), div_ceil(N, threadsPerBlock.y));
    identity<<<numBlocks, threadsPerBlock>>>(aInv[blockIdx.x], N, N);

    for (row = 0; row < N; ++row) {
	if (a[blockIdx.x][(row * N) + row] == 0) {
            /*cublasErrchk*/( cublasIsamax(handle,
                N - row,            // Number of elements to be searched
                &a[blockIdx.x][(row * N) + row],        // Starting position
                1,              // Increment in words (NOT BYTES)
                &pivot) );            // Maximum element in the row
            int pivotRow = pivot - 1 + row;          // Row number with maximum element (starts with 1)

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

        threadsPerBlock = dim3(min(N, 16*16), 1, 1);
        numBlocks = dim3(div_ceil(N, threadsPerBlock.x));
        transform_matrix<<<numBlocks, threadsPerBlock, 3*N*sizeof(DataType)>>>(a[blockIdx.x], aInv[blockIdx.x], row, N);
    }

    cublasDestroy(handle);
}

extern "C" void inverse_gauss_kernel_device(cublasHandle_t handle, int N, Array *devAs, Array *devAInvs, int batchSize) {
    inverse_gauss_kernel<<<batchSize, 1>>>(devAs, devAInvs, N, batchSize);
}

// Inverts `a` by inplace factorizing `a` and then inverting it into `aInv`.
extern "C" void inverse_lu_cuda_batched_device(cublasHandle_t handle, int N, Array *devAs, Array *devAInvs, int batchSize) {
    int i;
    int *PivotArray;
    int *infoArray;

    gpuErrchk( cudaHostAlloc((void**)&PivotArray, sizeof(int)*batchSize*N, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc((void**)&infoArray, sizeof(int)*batchSize, cudaHostAllocDefault) );

    cublasErrchk( cublasSgetrfBatched(handle,
        N, // number of rows and columns of Aarray[i].
        devAs, // array of pointers to <type> array, with each array of dimension n*n with lda>=max(1,n).
        N, // leading dimension of two-dimensional array used to store each matrix Aarray[i].
        PivotArray, // array of size n*batchSize that contains the pivoting sequence of each factorization of Aarray[i] stored in a linear fashion. If PivotArray is nil, pivoting is disabled.
        infoArray, // array of size batchSize that info(=infoArray[i]) contains the information of inversion of A[i].
                   // If info=0, the execution is successful.
                   // If info = k, U(k,k) is 0. The U is exactly singular and the inversion failed.
        batchSize) // number of pointers contained in A
    );

    for (i = 0; i < batchSize; ++i) {
        ensure(!infoArray[i], "Error during lu decomposition of batched matrix %d", i);
    }

    cublasErrchk( cublasSgetriBatched(handle,
        N, // number of rows and columns of Aarray[i].
        const_cast<const float**>(devAs),  // array of pointers to <type> array, with each array of dimension n*n with lda>=max(1,n).
        N, // leading dimension of two-dimensional array used to store each matrix Aarray[i].
        PivotArray,  // array of size n*batchSize that contains the pivoting sequence of each factorization of Aarray[i] stored in a linear fashion. If PivotArray is nil, pivoting is disabled.
        devAInvs, // array of pointers to <type> array, with each array of dimension n*n with ldc>=max(1,n).
        N, // leading dimension of two-dimensional array used to store each matrix Carray[i].
        infoArray, // array of size batchSize that info(=infoArray[i]) contains the information of inversion of A[i].
                   // If info=0, the execution is successful.
                   // If info = k, U(k,k) is 0. The U is exactly singular and the inversion failed.
        batchSize)  // number of pointers contained in A
    );

    for (i = 0; i < batchSize; ++i) {
        ensure(!infoArray[i], "Error during inversion of batched matrix %d", i);
    }

    gpuErrchk( cudaFreeHost((void*)infoArray) );
    gpuErrchk( cudaFreeHost((void*)PivotArray) );
}

extern "C" void inverse_gauss_kernel_gpu(
        cublasHandle_t handle,
        int n,
        Array As,
        Array aInvs,
        int batchSize) {

    Array *devAs;
    size_t pitchAs;
    Array *devAInvs;
    size_t pitchAInvs;

#ifdef DETAILED_LOGGING
    TIMER_INIT(inverse_gauss_kernel_gpu_mem_htod)
    TIMER_INIT(inverse_gauss_kernel_gpu_ker)
    TIMER_INIT(inverse_gauss_kernel_gpu_mem_dtoh)
#endif // DETAILED_LOGGING

    const size_t ArraySize = sizeof(DataType) * n * n;

    gpuErrchk( cudaHostAlloc((void**)&devAs, sizeof(Array)*batchSize, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc((void**)&devAInvs, sizeof(Array)*batchSize, cudaHostAllocDefault) );

    gpuErrchk( batchedCudaMalloc(devAs, &pitchAs, ArraySize, batchSize) );
    gpuErrchk( batchedCudaMalloc(devAInvs, &pitchAInvs, ArraySize, batchSize) );

#ifdef DETAILED_LOGGING
    TIMER_START(inverse_gauss_kernel_gpu_mem_htod)
#endif // DETAILED_LOGGING

    gpuErrchk( cudaMemcpy2D(devAs[0], pitchAs, As, ArraySize, ArraySize, batchSize,
                cudaMemcpyHostToDevice) );

#ifdef DETAILED_LOGGING
    TIMER_STOP(inverse_gauss_kernel_gpu_mem_htod)
    TIMER_START(inverse_gauss_kernel_gpu_ker)
#endif // DETAILED_LOGGING

    inverse_gauss_kernel_device(handle, n, devAs, devAInvs, batchSize);

#ifdef DETAILED_LOGGING
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    TIMER_STOP(inverse_gauss_kernel_gpu_ker)
    TIMER_START(inverse_gauss_kernel_gpu_mem_dtoh)
#endif // DETAILED_LOGGING

    gpuErrchk( cudaMemcpy2D(aInvs, ArraySize, devAInvs[0], pitchAInvs, ArraySize, batchSize,
                cudaMemcpyDeviceToHost) );

#ifdef DETAILED_LOGGING
    TIMER_STOP(inverse_gauss_kernel_gpu_mem_dtoh)

    TIMER_LOG(inverse_gauss_kernel_gpu_mem_htod)
    TIMER_LOG(inverse_gauss_kernel_gpu_ker)
    TIMER_LOG(inverse_gauss_kernel_gpu_mem_dtoh)
#endif // DETAILED_LOGGING

    gpuErrchk( cudaFree((void*)devAs[0]) );
    gpuErrchk( cudaFree((void*)devAInvs[0]) );
    gpuErrchk( cudaFreeHost((void*)devAs) );
    gpuErrchk( cudaFreeHost((void*)devAInvs) );
}

extern "C" void inverse_lu_cuda_batched_gpu(
        cublasHandle_t handle,
        int n,
        Array As,
        Array aInvs,
        int batchSize) {

    Array *devAs;
    size_t pitchAs;
    Array *devAInvs;
    size_t pitchAInvs;

#ifdef DETAILED_LOGGING
    TIMER_INIT(inverse_lu_cuda_batched_gpu_mem_htod)
    TIMER_INIT(inverse_lu_cuda_batched_gpu_ker)
    TIMER_INIT(inverse_lu_cuda_batched_gpu_mem_dtoh)
#endif // DETAILED_LOGGING

    const size_t ArraySize = sizeof(DataType) * n * n;

    gpuErrchk( cudaHostAlloc((void**)&devAs, sizeof(Array)*batchSize, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc((void**)&devAInvs, sizeof(Array)*batchSize, cudaHostAllocDefault) );

    gpuErrchk( batchedCudaMalloc(devAs, &pitchAs, ArraySize, batchSize) );
    gpuErrchk( batchedCudaMalloc(devAInvs, &pitchAInvs, ArraySize, batchSize) );

#ifdef DETAILED_LOGGING
    TIMER_START(inverse_lu_cuda_batched_gpu_mem_htod)
#endif // DETAILED_LOGGING

    gpuErrchk( cudaMemcpy2D(devAs[0], pitchAs, As, ArraySize, ArraySize, batchSize,
                cudaMemcpyHostToDevice) );

#ifdef DETAILED_LOGGING
    TIMER_STOP(inverse_lu_cuda_batched_gpu_mem_htod)
    TIMER_START(inverse_lu_cuda_batched_gpu_ker)
#endif // DETAILED_LOGGING

    inverse_lu_cuda_batched_device(handle, n, devAs, devAInvs, batchSize);

#ifdef DETAILED_LOGGING
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    TIMER_STOP(inverse_lu_cuda_batched_gpu_ker)
    TIMER_START(inverse_lu_cuda_batched_gpu_mem_dtoh)
#endif // DETAILED_LOGGING

    gpuErrchk( cudaMemcpy2D(aInvs, ArraySize, devAInvs[0], pitchAInvs, ArraySize, batchSize,
                cudaMemcpyDeviceToHost) );

#ifdef DETAILED_LOGGING
    TIMER_STOP(inverse_lu_cuda_batched_gpu_mem_dtoh)

    TIMER_LOG(inverse_lu_cuda_batched_gpu_mem_htod)
    TIMER_LOG(inverse_lu_cuda_batched_gpu_ker)
    TIMER_LOG(inverse_lu_cuda_batched_gpu_mem_dtoh)
#endif // DETAILED_LOGGING

    gpuErrchk( cudaFree((void*)devAs[0]) );
    gpuErrchk( cudaFree((void*)devAInvs[0]) );
    gpuErrchk( cudaFreeHost((void*)devAs) );
    gpuErrchk( cudaFreeHost((void*)devAInvs) );
}
