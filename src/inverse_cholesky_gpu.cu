#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "../include/types.h"
#include "../include/helper.h"

#define OPS_PER_THREAD 62


__global__
void decomposeCholeskyKernel(Array a, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Get thread identifier

    int i, j, k;

    for (k = 0; k < N; k++) {


        if (tid == 0) { // computing diagonal elements in the first thread

            a[k * N + k] = sqrt(a[k * N + k]);
            //printf("[%d][%d] = sqrt([%d][%d])\n", k, k, k, k);

            for (j = (k + 1); j < N; j++) {
                a[k * N + j] /= a[k * N + k]; // divide by diagonal elemnents
                //printf("[%d][%d] /= [%d][%d]\n", j, k, k, k);
            }
        }

        __syncthreads(); // all diagonal elemnents need to be computed

        int lower = tid * OPS_PER_THREAD + (k + 1);
        int upper = lower + OPS_PER_THREAD - 1;
        upper = upper > N - 1 ? N - 1 : upper;

        for (i = lower; i <= upper; i++) {
            for (j = i; j < N; j++) {
                a[i * N + j] -= a[k * N + i] * a[k * N + j];
                //printf("[%d][%d] -= [%d][%d] * [%d][%d]\n", j, i, i, k, j, k);
            }
        }

        __syncthreads(); // compute row by row
    }

    __syncthreads();

    // set zeroes
    int lower = tid * OPS_PER_THREAD;
    int upper = lower + OPS_PER_THREAD - 1;
    upper = upper > N - 1 ? N - 1 : upper;
    for (i = lower; i <= upper; i++) {
        for (j = 0; j < i; j++) {
            a[i * N + j] = 0;
        }
    }


}

void decomposeCholeskyGPU(Array a, int N) {
    int threads = N * N / OPS_PER_THREAD;
    decomposeCholeskyKernel<<< 1, threads >>>(a, N);
}

__global__
void inverseLowerKernel(Array a, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Get thread identifier

    int i, j, k;

    for (k = 0; k < N; k++) {

        if (tid == 0) { // re-compute diagonal elements in the first thread
            a[k * N + k] = 1.0 / a[k * N + k];

            for (j = (k + 1); j < N; j++) {
                a[k * N + j] = 0 - a[k * N + j] * a[k * N + k] / a[j * N + j];
                //printf("[%d][%d] = -[%d][%d] * [%d][%d] / [%d][%d]\n", j, k, j, k, k, k, j, j);
            }
        }

        __syncthreads(); // all diagonal elemnents need to be re-computed

        int lower = tid * OPS_PER_THREAD + (k + 2);
        int upper = lower + OPS_PER_THREAD - 1;

        upper = upper > N - 1 ? N - 1 : upper;

        for (i = lower; i <= upper; i++) {

            for (j = k + 1; j < i; j++) {
                a[k * N + i] -= a[j * N + i] * a[k * N + j] / a[i * N + i];
                //printf("[%d][%d] -= [%d][%d] * [%d][%d] / [%d][%d]\n", i, k, i, j, j, k, i, i);
            }

            __syncthreads();
        }

        __syncthreads(); // compute row by row
    }
}

void inverseLowerGPU(Array a, int N) {
    int threads = N * N / OPS_PER_THREAD;
    inverseLowerKernel<<< 1, threads >>>(a, N);
}


__global__
void multiplyLowerKernel(Array a, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // Get thread identifier

    int i, j, k;

    for (k = 0; k < N; k++) {

        int lower = tid * OPS_PER_THREAD + k;
        int upper = lower + OPS_PER_THREAD - 1;

        upper = upper > N - 1 ? N - 1 : upper;

        for (i = lower; i <= upper; i++) {
            a[i * N + k] = a[i * N + i] * a[k * N + i];
            //printf("[%d][%d] = [%d][%d] * [%d][%d]\n", i, k, i, i, i, k);

            for (j = i + 1; j < N; j++) {
                // use upper matrix as buffer for multiplication
                a[i * N + k] += a[i * N + j] * a[k * N + j];
                //printf("[%d][%d] += [%d][%d] * [%d][%d]\n", i, k, j, i, j, k);
            }
        }

        __syncthreads(); // compute row by row
    }

    __syncthreads();


    // set back to lower matrix
    int lower = tid * OPS_PER_THREAD;
    int upper = lower + OPS_PER_THREAD - 1;
    upper = upper > N - 1 ? N - 1 : upper;
    for (i = lower; i <= upper; i++) {
        for (j = 0; j < i; j++) {
            a[j * N + i] = a[i * N + j];
        }
    }
}

void multiplyLowerGPU(Array a, int N) {
    int threads = N * N / OPS_PER_THREAD;
    multiplyLowerKernel<<< 1, threads >>>(a, N);
}

extern "C" void inverse_chol_gpu(Array a, int n) {
    Array a_dev;

    size_t matrixSize = n*n * sizeof(ELEMENT_TYPE);
    gpuErrchk( cudaMalloc(&a_dev, matrixSize) );
    gpuErrchk( cudaMemcpy(a_dev, a, matrixSize, cudaMemcpyHostToDevice) );

    decomposeCholeskyGPU(a_dev, n);
    inverseLowerGPU(a_dev, n);
    multiplyLowerGPU(a_dev, n);

    gpuErrchk( cudaMemcpy(a, a_dev, matrixSize, cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(a_dev) );
}

// int main(int argc, char *argv[]) {
//     const char *directory = "tests/simpleMean";
//     char filePath[1024];
//     int numMatrices, m, n, matrixSize;
//     Array a, a_dev;

//     snprintf(filePath, 1024, "%s/chol2.mats", directory);
//     readMatricesFile(filePath, &numMatrices, &m, &n, &a);
//     printMatrix(a, m, n);

//     matrixSize = m * n * sizeof(ELEMENT_TYPE);
//     gpuErrchk( cudaMalloc(&a_dev, matrixSize) );
//     gpuErrchk( cudaMemcpy(a_dev, a, matrixSize, cudaMemcpyHostToDevice) );

//     decomposeCholeskyGPU(a_dev, n);
//     inverseLowerGPU(a_dev, n);
//     multiplyLowerGPU(a_dev, n);

//     cudaMemcpy(a, a_dev, matrixSize, cudaMemcpyDeviceToHost);
//     printMatrix(a, m, n);

//     gpuErrchk( cudaPeekAtLastError() );
//     gpuErrchk( cudaDeviceSynchronize() );

//     return 0;
// }


