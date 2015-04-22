#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <time.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#include <cblas.h>
#include <lapacke.h>

#ifdef __cplusplus
}
#endif // __cplusplus

#include "../include/types.h"
#include "../include/helper_cpu.h"
#include "../include/helper_gpu.h"
#include "../include/timer.h"
#include "../include/inverse_cpu.h"
#include "../include/inverse_gpu.h"

#define MAX_MATRIX_BYTE_READ 67108864

// b -= a
static void vec_diff(const Array a, Array b, const int N) {
    cblas_saxpy(N, -1.f, a, 1, b, 1);
}

static DataType vec_sum(const Array a, const int N) {
    return cblas_sasum(N, a, 1);
}

#define BENCH_VAR(name) \
    float error_##name = 0; \
    double total_error_##name = 0; \
    TIMER_INIT(name) \
    TIMER_ACC_INIT(name)

#define BENCH_SETUP(name)

#define BENCH_CLEANUP(name) \
    vec_diff(inv, aInv, N*N*numMatrices); \
    total_error_##name += vec_sum(aInv, N*N*numMatrices);

#ifndef DETAILED_LOGGING
#define BENCH_REPORT(name) \
    if (csv) { \
        if (numReps > 1) { \
            printf("%d %d %d " #name " %e %e %e %e\n", \
                numMatrices, N, numReps, TIMER_TOTAL(name), TIMER_MEAN(name), TIMER_VARIANCE(name), total_error_##name/numMatrices/numReps); \
        } else { \
            printf("%d %d %d " #name " %e %e\n", \
                numMatrices, N, numReps, TIMER_TOTAL(name), total_error_##name/numMatrices/numReps); \
        } \
    } else { \
        if (numReps > 1) { \
            printf(#name " - %d %dx%d matrices, replicated %d times, runtime %.4f ms (%.4f ms average, %.4f ms variance), average error %.4e\n", \
                numMatrices, N, N, numReps, TIMER_TOTAL(name), TIMER_MEAN(name), TIMER_VARIANCE(name), total_error_##name/numMatrices/numReps); \
        } else { \
            printf(#name " - %d %dx%d matrices, replicated %d times, runtime %.4f ms, average error %.4e\n", \
                numMatrices, N, N, numReps, TIMER_TOTAL(name), total_error_##name/numMatrices/numReps); \
        } \
    }
#else
#define BENCH_REPORT(name)
#endif

void bench_parallel(int numMatrices, int numReps, int N, const Array a, Array aInv, bool csv) {
    cublasHandle_t handle;

    Array inv = (Array)malloc(numMatrices*N*N*sizeof(DataType));
    Array workspace = (Array)malloc(numMatrices*N*N*sizeof(DataType));

    int i, rep;

    BENCH_VAR(lu_blas_cpu)
    BENCH_VAR(lu_blas_omp_cpu)
    BENCH_VAR(chol_gpu)
    BENCH_VAR(chol_mm2_gpu)
    BENCH_VAR(gauss_batched_gpu)
    BENCH_VAR(lu_cuda_batched_gpu)

    // CPU Benchmark 1
    ////////////////
    BENCH_SETUP(lu_blas_cpu)

    for (rep = 0; rep < numReps; ++rep) {
        cblas_scopy(numMatrices*N*N, a, 1, inv, 1);

        TIMER_START(lu_blas_cpu)
        for (i = 0; i < numMatrices; ++i) {
            Array current_inv = inv + (i * N * N);

            inverse_lu_blas(current_inv, workspace, N);
        }
        TIMER_STOP(lu_blas_cpu)
#ifdef DETAILED_LOGGING
        TIMER_LOG(lu_blas_cpu, numMatrices, N)
#endif // DETAILED_LOGGING
        TIMER_ACC(lu_blas_cpu)
    }

    BENCH_CLEANUP(lu_blas_cpu);

    // CPU Benchmark 2
    ////////////////
    BENCH_SETUP(lu_blas_omp_cpu)

    for (rep = 0; rep < numReps; ++rep) {
        cblas_scopy(numMatrices*N*N, a, 1, inv, 1);

        TIMER_START(lu_blas_omp_cpu)
        inverse_lu_blas_omp(inv, N, numMatrices);
        TIMER_STOP(lu_blas_omp_cpu)
#ifdef DETAILED_LOGGING
        TIMER_LOG(lu_blas_omp_cpu, numMatrices, N)
#endif // DETAILED_LOGGING
        TIMER_ACC(lu_blas_omp_cpu)
    }

    BENCH_CLEANUP(lu_blas_omp_cpu);

    // Create handle after CPU benchmarks to allow testing on non-nvidia host
    cublasErrchk( cublasCreate(&handle) );

    // GPU Benchmark 1
    //////////////////
    // Build benchmark data
    BENCH_SETUP(chol_gpu)

    // Compute inverses
    for (rep = 0; rep < numReps; ++rep) {
        cblas_scopy(numMatrices*N*N, a, 1, inv, 1);

        TIMER_START(chol_gpu)
        inverse_cholesky_batched_gpu(handle, N, a, inv, numMatrices);
        TIMER_STOP(chol_gpu)
#ifdef DETAILED_LOGGING
        TIMER_LOG(chol_gpu, numMatrices, N)
#endif // DETAILED_LOGGING
        TIMER_ACC(chol_gpu)

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    // calculate error
    BENCH_CLEANUP(chol_gpu)

    // GPU Benchmark 2
    //////////////////
    // Build benchmark data
    BENCH_SETUP(chol_mm2_gpu)

    // Compute inverses
    for (rep = 0; rep < numReps; ++rep) {
        cblas_scopy(numMatrices*N*N, a, 1, inv, 1);

        TIMER_START(chol_mm2_gpu)
        inverse_cholesky_mm2_batched_gpu(handle, N, a, inv, numMatrices);
        TIMER_STOP(chol_mm2_gpu)
#ifdef DETAILED_LOGGING
        TIMER_LOG(chol_mm2_gpu, numMatrices, N)
#endif // DETAILED_LOGGING
        TIMER_ACC(chol_mm2_gpu)

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    // calculate error
    BENCH_CLEANUP(chol_mm2_gpu)

    // GPU Benchmark 3
    //////////////////
    // Build benchmark data
    BENCH_SETUP(gauss_batched_gpu)

    for (rep = 0; rep < numReps; ++rep) {
        cblas_scopy(numMatrices*N*N, a, 1, workspace, 1);

        TIMER_START(gauss_batched_gpu)
        inverse_gauss_batched_gpu(handle, N, workspace, inv, numMatrices);
        TIMER_STOP(gauss_batched_gpu)
#ifdef DETAILED_LOGGING
        TIMER_LOG(gauss_batched_gpu, numMatrices, N)
#endif // DETAILED_LOGGING
        TIMER_ACC(gauss_batched_gpu)

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    // calculate error
    BENCH_CLEANUP(gauss_batched_gpu)

    // GPU Benchmark 4
    //////////////////
    // Build benchmark data
    BENCH_SETUP(lu_cuda_batched_gpu)

    for (rep = 0; rep < numReps; ++rep) {
        cblas_scopy(numMatrices*N*N, a, 1, workspace, 1);

        TIMER_START(lu_cuda_batched_gpu)
        inverse_lu_cuda_batched_gpu(handle, N, workspace, inv, numMatrices);
        TIMER_STOP(lu_cuda_batched_gpu)
#ifdef DETAILED_LOGGING
        TIMER_LOG(lu_cuda_batched_gpu, numMatrices, N)
#endif // DETAILED_LOGGING
        TIMER_ACC(lu_cuda_batched_gpu)

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    BENCH_CLEANUP(lu_cuda_batched_gpu)

    BENCH_REPORT(lu_blas_cpu);
    BENCH_REPORT(lu_blas_omp_cpu);
    BENCH_REPORT(chol_gpu);
    BENCH_REPORT(chol_mm2_gpu);
    BENCH_REPORT(gauss_batched_gpu);
    BENCH_REPORT(lu_cuda_batched_gpu);

    cublasErrchk( cublasDestroy(handle) );

    free(workspace);
    free(inv);
}

static void readTest(const char *directory, int *numMatrices, int *n,
        Array *a, Array *aInv) {
    char filePath[1024];

    int numMatricesA, numMatricesAInv;
    int mA, mAInv;
    int nA, nAInv;

    snprintf(filePath, 1024, "%s/a.mats", directory);
    readMatricesFile(filePath, &numMatricesA, &mA, &nA, a);

    snprintf(filePath, 1024, "%s/aInv.mats", directory);
    readMatricesFile(filePath, &numMatricesAInv, &mAInv, &nAInv, aInv);

    ensure(
        numMatricesA == numMatricesAInv,
        "test in directory %s invalid, number of matrices in files not matching\r\n"
        "numMatricesA(%d) numMatricesAInv(%d)\r\n",
        directory,
        numMatricesA, numMatricesAInv
    );

    ensure(
        mA == mAInv && nA == nAInv,
        "test in directory %s invalid, dimensions not matching\r\n"
        "mA(%d) mAInv(%d)\r\n"
        "nA(%d) nAInv(%d)\r\n",
        directory,
        mA, mAInv,
        nA, nAInv
    );

    *numMatrices = numMatricesA;
    *n = mA;
}

int main(int argc, char const *argv[]) {
    ensure(argc >= 4, "Usage: inverse_bench TEST_FOLDER TEST_REPLICATIONS MATRIX_DUPLICATES [-csv]");

    int numMatrices, numReps, numDuplicates;
    int n;

    Array a, aInv;

    bool csv = (argc >= 5) && !strncmp("-csv", argv[4], 4);

    numReps = atoi(argv[2]);
    numDuplicates = atoi(argv[3]);

    readTest(argv[1], &numMatrices, &n, &a, &aInv);

    replicateMatrices(&a, n, n, numMatrices, numDuplicates);
    replicateMatrices(&aInv, n, n, numMatrices, numDuplicates);

    numMatrices *= numDuplicates;

    bench_parallel(numMatrices, numReps, n, a, aInv, csv);

    free(a); free(aInv);

    cudaDeviceReset();

    return 0;
}
