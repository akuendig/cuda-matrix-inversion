#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <cuda.h>
#include "cublas_v2.h"

#include "../../include/types.h"
#include "../../include/helper.h"
#include "../../include/inverse.h"

__global__
void transform_matrix(Array a, Array a_inv, int row, int N) {
    __shared__ DataType scalars[64];
    __shared__ DataType currRowA[64], currRowI[64];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // store the scalars corresponding to the column 'row'
    scalars[idx] = a[row * N + idx];
    currRowA[idx] = a[idx * N + row];
    currRowI[idx] = a_inv[idx * N + row];
    __syncthreads();

    // No need to transform 'row'th row
    if(idx == row || idx > N)
        return;

    // Each thread transforms row
    for(int i = 0; i < N; i++) {
        a[i * N + idx] -= (scalars[idx] * currRowA[i]);
        a_inv[i * N + idx] -= (scalars[idx] * currRowI[i]);
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

        dim3 threadsPerBlock(16*16, 1, 1);
        dim3 numBlocks(div_ceil(N, threadsPerBlock.x));
        transform_matrix<<<numBlocks, threadsPerBlock>>>(a[blockIdx.x], aInv[blockIdx.x], row, N);
    }

    cublasDestroy(handle);
}

// Inverts `a` by inplace factorizing `a` and then inverting it into `aInv`.
static void inverse_cublas(cublasHandle_t handle, Array *a, Array *aInv, int N, int batchSize) {
    int i;
    int *PivotArray;
    int *infoArray;

    gpuErrchk( cudaHostAlloc((void**)&PivotArray, sizeof(int)*batchSize*N, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc((void**)&infoArray, sizeof(int)*batchSize, cudaHostAllocDefault) );

    cublasErrchk( cublasSgetrfBatched(handle,
        N, // number of rows and columns of Aarray[i].
        a, // array of pointers to <type> array, with each array of dimension n*n with lda>=max(1,n).
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
        const_cast<const float**>(a),  // array of pointers to <type> array, with each array of dimension n*n with lda>=max(1,n).
        N, // leading dimension of two-dimensional array used to store each matrix Aarray[i].
        PivotArray,  // array of size n*batchSize that contains the pivoting sequence of each factorization of Aarray[i] stored in a linear fashion. If PivotArray is nil, pivoting is disabled.
        aInv, // array of pointers to <type> array, with each array of dimension n*n with ldc>=max(1,n).
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

    const size_t ArraySize = sizeof(DataType) * n * n;

    gpuErrchk( cudaHostAlloc((void**)&devAs, sizeof(Array)*batchSize, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc((void**)&devAInvs, sizeof(Array)*batchSize, cudaHostAllocDefault) );

    gpuErrchk( batchedCudaMalloc(devAs, &pitchAs, ArraySize, batchSize) );
    gpuErrchk( batchedCudaMalloc(devAInvs, &pitchAInvs, ArraySize, batchSize) );

    gpuErrchk( cudaMemcpy2D(devAs[0], pitchAs, As, ArraySize, ArraySize, batchSize,
                cudaMemcpyHostToDevice) );

    inverse_cublas(handle, devAs, devAInvs, n, batchSize);

    gpuErrchk( cudaMemcpy2D(aInvs, ArraySize, devAInvs[0], pitchAInvs, ArraySize, batchSize,
                cudaMemcpyDeviceToHost) );

    gpuErrchk( cudaFree((void*)devAs[0]) );
    gpuErrchk( cudaFree((void*)devAInvs[0]) );
    gpuErrchk( cudaFreeHost((void*)devAs) );
    gpuErrchk( cudaFreeHost((void*)devAInvs) );
}
