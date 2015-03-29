#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <cuda.h>
#include "cublas_v2.h"

#include "../include/helper.h"

#define N 30

int main(int argc, char *argv[]) {
    float a[N];
    float *dev_a;
    int pivot, i;
    cublasHandle_t handle;

    /* Pre-processing steps */
    gpuErrchk( cudaMalloc((void **) &dev_a, N * sizeof(float)) );

    /* Input column major matrix */
    for(i = 0; i < N; i++)
        scanf("%f", &a[N]);
    gpuErrchk( cudaMemcpy(dev_a, a, N * sizeof(float), cudaMemcpyHostToDevice) );

    /* Invert the matrix */
    cublasErrchk( cublasCreate(&handle) );
    cublasErrchk( cublasIsamax(handle,
                    N,          // Number of elements to be searched
                    dev_a,      // Starting position
                    1,          // Increment in words (NOT BYTES)
                    &pivot)     // Maximum element in the col
    );

    printf("%d\n", pivot);

    /* Cleanup the mess */
    cublasErrchk( cublasDestroy(handle) );
    gpuErrchk( cudaFree(dev_a) );

    return 0;
}
