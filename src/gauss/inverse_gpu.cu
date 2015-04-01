#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <cuda.h>
#include "cublas_v2.h"

#include "../../include/types.h"
#include "../../include/helper.h"
#include "../../include/inverse.h"

#define SWAP(x, y, z)   ((z) = (x),(x) = (y),(y) = (z))


void pivotRow(cublasHandle_t &handle, Array a, Array a_inv, int col, int N) {
    int pivot = -1;

    cublasErrchk( cublasIsamax(handle,
        N - col,            // Number of elements to be searched
        a + (col * N) + col,        // Starting position
        1,              // Increment in words (NOT BYTES)
        &pivot) );            // Maximum element in the col
    int row = pivot - 1 + col;          // Row number with maximum element (starts with 1)

    // printf("Pivot: %d\nRow: %d\n", pivot, row);
    if(row == col)
        return;

    cublasErrchk( cublasSswap(handle,
        N,              // Nuber of elements to be swapped
        a + col,            // Current row
        N,              // Increment (becuase of column major)
        a + row,            // Row with max pivot
        N) );
    cublasErrchk( cublasSswap(handle, N, a_inv + col, N, a_inv + row, N) );
}

void normalizeRow(cublasHandle_t handle, Array a, Array a_inv, int row, int N) {
    DataType scalar;

    gpuErrchk( cudaMemcpy(&scalar, &a[row * N + row], sizeof(DataType), cudaMemcpyDeviceToHost) );
    scalar = 1 / scalar;
    cublasErrchk( cublasSscal(handle, N, &scalar, a + row, N) );
    cublasErrchk( cublasSscal(handle, N, &scalar, a_inv + row, N) );
}

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

void invert(cublasHandle_t &handle, Array devA, Array devAInv, int N) {
    for(int i = 0; i < N; i++) {
        // Pivot the matrix
        pivotRow(handle, devA, devAInv, i, N);

        // Make column entry to be one
        normalizeRow(handle, devA, devAInv, i, N);

        // Number of threads equals number of rows
        transform_matrix<<<1, N>>>(devA, devAInv, i, N);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
}

__global__
void inverse_gauss_kernel(Array a, Array aInv, int N) {
    int row, pivot;
    cublasHandle_t handle;

    cublasCreate(&handle);

    for (row = 0; row < N; ++row) {
        /*cublasErrchk*/( cublasIsamax(handle,
            N - row,            // Number of elements to be searched
            a + (row * N) + row,        // Starting position
            1,              // Increment in words (NOT BYTES)
            &pivot) );            // Maximum element in the row
        int pivotRow = pivot - 1 + row;          // Row number with maximum element (starts with 1)

        // printf("Pivot: %d\nRow: %d\n", pivot, pivotRow);
        if(pivotRow != row) {
            /*cublasErrchk*/( cublasSswap(handle,
                N,              // Nuber of elements to be swapped
                a + row,            // Current pivotRow
                N,              // Increment (becuase of column major)
                a + pivotRow,            // Row with max pivot
                N) );
            /*cublasErrchk*/( cublasSswap(handle,
                N,
                aInv + row,
                N,
                aInv + pivotRow,
                N) );
        }

        DataType scalar = 1/a[row * N + row];

        /*cublasErrchk*/( cublasSscal(handle,
            N,
            &scalar,
            a + row,
            N) );
        /*cublasErrchk*/( cublasSscal(handle,
            N,
            &scalar,
            aInv + row,
            N) );

        transform_matrix<<<1, N>>>(a, aInv, row, N);
    }

    cublasDestroy(handle);
}

extern "C" void inverse_gauss_gpu(Array a, int n) {
    int i;
    Array aInv, devA, devAInv;

    const size_t ArraySize = n*n * sizeof(DataType);
    aInv = (Array)malloc(ArraySize);
    ensure(aInv, "could not allocate 0x%lX bytes of memory for matrix inverse", ArraySize);

    memset(aInv, 0, ArraySize);
    for (i = 0; i < n; ++i) { aInv[i*n + i] = 1.f; }

    gpuErrchk( cudaMalloc(&devA, ArraySize) );
    gpuErrchk( cudaMalloc(&devAInv, ArraySize) );

    gpuErrchk( cudaMemcpy(devA, a, ArraySize, cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(devAInv, aInv, ArraySize, cudaMemcpyHostToDevice) );

    /* Invert the matrix */
    inverse_gauss_kernel<<<1, 1>>>(devA, devAInv, n);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    /* Display the result */
    gpuErrchk( cudaMemcpy(a, devAInv, ArraySize, cudaMemcpyDeviceToHost) );

    /* Cleanup the mess */
    gpuErrchk( cudaFree(devAInv) );
    gpuErrchk( cudaFree(devA) );

    free(aInv);
}

// int main(int argc, char *argv[]) {
//  Array a, a_inv;
//  Array dev_a, dev_a_inv;
//  cublasHandle_t handle;

//  /* Pre-processing steps */
//  if(!(a = (Array)malloc(ArraySize))) {
//      perror("");
//      return errno;
//  }
//  if(!(a_inv = (Array)malloc(ArraySize))) {
//      perror("");
//      return errno;
//  }
//  cudaMalloc(&dev_a, ArraySize);
//  cudaMalloc(&dev_a_inv, ArraySize);
//  cublasCreate(&handle);

//  /* Input column major matrix */
//  for(int i = 0; i < N; i++)
//      for(int j = 0; j < N; j++) {
//          scanf("%f", &a[i * N + j]);
//          if(i == j)
//              a_inv[i * N + j] = 1;
//          else
//              a_inv[i * N + j] = 0;
//      }
//  cudaMemcpy(dev_a, a, ArraySize, cudaMemcpyHostToDevice);
//  cudaMemcpy(dev_a_inv, a_inv, ArraySize, cudaMemcpyHostToDevice);

//  /* Invert the matrix */
//  invert(handle, dev_a, dev_a_inv);

//  /* Display the result */
//  cudaMemcpy(a, dev_a, ArraySize, cudaMemcpyDeviceToHost);
//  cudaMemcpy(a_inv, dev_a_inv, ArraySize, cudaMemcpyDeviceToHost);
//  printf("Inverse is:\n");
//  //printMatrix(a);
//  printMatrix(a_inv);

//  /* Cleanup the mess */
//  free(a);
//  free(a_inv);
//  cudaFree(dev_a);
//  cudaFree(dev_a_inv);
//  cublasDestroy(handle);

//  return 0;
// }
