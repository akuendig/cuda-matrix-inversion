#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "../include/types.h"
#include "../include/helper_cpu.h"
#include "../include/helper_gpu.h"

// Reads `numMatrices` matrices from file at `path` where each matrix has dimension `m` x `n`.
// Memory is allocated in one big block, so `matrices` points to the beginning of all matrices.
extern "C" void readMatricesFile(const char *path, int *numMatrices, int *m, int *n, Array *matrices) {
    int ret;
    int _numMatrices, _m, _n;

    FILE* fp = fopen(path, "r");
    ensure(fp, "could not open matrix file %s", path);

    ret = fscanf(fp, "%d %d %d", &_numMatrices, &_m, &_n);
    ensure(3 == ret, "could not read number of matrices from file %s", path);

    *numMatrices = _numMatrices;
    *m = _m;
    *n = _n;

    size_t arraySize = sizeof(DataType) * (_numMatrices) * (_m) * (_n);
    ensure(arraySize <= MAX_MATRIX_BYTE_READ, "cannot read file %s because "
        "the allocated array would be bigger than 0x%lX bytes", path, arraySize);

    *matrices = (Array)malloc(arraySize);
    ensure(*matrices, "could not allocate 0x%lX bytes of memory for file %s", arraySize, path);

    int k, i, j;

    for (k = 0; k < _numMatrices; ++k) {
        Array firstElement = *matrices;
        Array currentMatrix = firstElement + k*_m*_n;

        // Read row by row
        for (i = 0; i < _m; ++i) {
            for (j = 0; j < _n; ++j) {
                ret = fscanf(fp, "%f", &currentMatrix[j*_m + i]);
                ensure(ret, "could not read matrix from file %s, stuck at matrix %d element %d, %d", path, k, i, j);
            }
        }
    }

    fclose(fp);
}

extern "C" void replicateMatrices(Array *matrices, const int M, const int N, const int numMatrices, const int numReplications) {
    const size_t ArraySize = M*N*sizeof(DataType);
    const size_t ArrayListSize = ArraySize*numMatrices;
    const size_t ArrayFinalSize = ArrayListSize*numReplications;

    char *replicated = (char*)malloc(ArrayFinalSize);
    ensure(replicated, "Could not allocate memory for the replicated array (%lu bytes).", ArrayFinalSize);

    int i;
    char *currentHead = replicated;

    for (i = 0; i < numReplications; ++i, currentHead += ArrayListSize) {
        memcpy(currentHead, *matrices, ArrayListSize);
    }

    free(*matrices);

    *matrices = (Array)replicated;
}

extern "C" void printMatrix(Array a, int M, int N) {
    int i, j;

    for(i = 0; i < M; i++) {
        for(j = 0; j < N; j++)
            printf("%f\t", a[j * M + i]);
        printf("\n");
    }

    printf("\n");
}

// Prints matrix a stored in column major format
extern "C" void printMatrixList(Array a, int N, int batchSize) {
    int i, j, k;

    for(k = 0; k < batchSize; k++) {
        printf("=============== <%d> ===============\n", k + 1);
        for(i = 0; i < N; i++) {
            for(j = 0; j < N; j++)
                printf("%f\t", a[k * N * N + j * N + i]);
            printf("\n");
        }
    }
    printf("\n");
}

// Allocates one continous array of memory of size arraySize*batchSize and writes the
// pointers of all subarrays into the array of pointers located at devArrayPtr.
cudaError_t batchedCudaMalloc(Array* devArrayPtr, size_t *pitch, size_t arraySize, int batchSize) {
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
