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
#ifdef __APPLE__
    #include <lapacke.h>
#else
    #include <clapack.h>
#endif // __APPLE__
#ifdef __cplusplus
}
#endif // __cplusplus

#include "../include/types.h"
#include "../include/timer.h"
#include "../include/helper_cpu.h"
#include "../include/helper_gpu.h"
#include "../include/inverse_cpu.h"
#include "../include/inverse_gpu.h"

#define MAX_MATRIX_BYTE_READ 67108864
#define BENCH_REPS 10

static void fill_sym(Array a, int M, int N) {
    int i, j;

    for (i = 0; i < N-1; ++i) {
        for (j = i+1; j < M; ++j) {
            a[i*M + j] = a[j*M + i];
        }
    }
}

static void mat_sum(Array a, int M, int N, DataType *total) {
    *total = cblas_sasum(M*N, a, 1);
}

#define BENCH_VAR(name) \
    float error_##name = 0; \
    double total_error_##name = 0; \
    TIMER_INIT(name) \
    TIMER_ACC_INIT(name)

#define BENCH_SETUP(name) \
    for (i = 0; i < numMatrices; ++i) { \
        Array current_a = a + (i * M * N); \
        Array current_atra = atra + (i * N * N); \
\
        cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans, \
            N, M, 1, current_a, M, 0, current_atra, N); \
        fill_sym(current_atra, N, N); \
    }

#define BENCH_CLEANUP(name) \
    for (i = 0; i < numMatrices; ++i) { \
        Array current_atra = atra + (i * N * N); \
        Array current_inv = inv + (i * N * N); \
        Array current_rec = reconstr + (i * N * N);\
\
        cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper, \
            M, N, 1.f, current_inv, N, current_atra, N, 0, current_rec, N); \
        mat_sum(current_rec, M, N, &error_##name); \
\
        total_error_##name += error_##name; \
        if (detailed) { printf("L1 error for " #name  ": %f\n", error_##name); } \
    }

#define BENCH_REPORT_ERROR(name) \
    printf("Total error for %d %dx%d matrices of " #name ": %.2e (%.2e average)\n", \
        numMatrices, N, N, total_error_##name, total_error_##name/numMatrices)

#ifdef __APPLE__
#define BENCH_REPORT_TIME(name) \
    printf("Total execution time for %d %dx%d matrices and %d replications of " #name ": %lu cycles (%lu cycles average)\n", \
        numMatrices, N, N, BENCH_REPS, timer_total_##name, timer_total_##name/numMatrices/BENCH_REPS)
#else
#define BENCH_REPORT_TIME(name) \
    printf("Total execution time for %d %dx%d matrices and %d replications of " #name ": %.4f ms (%.4f ms average)\n", \
        numMatrices, N, N, BENCH_REPS, time_to_ms(&timer_total_##name), time_to_ms(&timer_total_##name)/numMatrices/BENCH_REPS)
#endif // __APPLE__


void bench_parallel(int numMatrices, int M, int N, Array a, bool detailed) {
    cublasHandle_t handle;

    Array atra = (Array)malloc(numMatrices*N*N*sizeof(DataType));
    Array inv = (Array)malloc(numMatrices*N*N*sizeof(DataType));
    Array reconstr = (Array)malloc(numMatrices*N*N*sizeof(DataType));
    Array workspace = (Array)malloc(N*N*sizeof(DataType));

    int i, rep;

    BENCH_VAR(lu_blas_cpu)
    BENCH_VAR(lu_blas_omp_cpu)
    BENCH_VAR(chol_gpu)
    BENCH_VAR(gauss_kernel_gpu)
    BENCH_VAR(gauss_batched_gpu)
    BENCH_VAR(lu_cuda_batched_gpu)

    // CPU Benchmark 1
    ////////////////
    BENCH_SETUP(lu_blas_cpu)

    for (rep = 0; rep < BENCH_REPS; ++rep) {
        cblas_scopy(numMatrices*N*N, atra, 1, inv, 1);

        TIMER_START(lu_blas_cpu)
        for (i = 0; i < numMatrices; ++i) {
            Array current_atra = atra + (i * N * N);
            Array current_inv = inv + (i * N * N);

            inverse_lu_blas(current_inv, workspace, N);
        }
        TIMER_STOP(lu_blas_cpu)
        TIMER_ACC(lu_blas_cpu)
    }

    BENCH_CLEANUP(lu_blas_cpu);

    // CPU Benchmark 2
    ////////////////
    BENCH_SETUP(lu_blas_omp_cpu)

    for (rep = 0; rep < BENCH_REPS; ++rep) {
        cblas_scopy(numMatrices*N*N, atra, 1, inv, 1);

        TIMER_START(lu_blas_omp_cpu)
        inverse_lu_blas_omp(inv, N, numMatrices);
        TIMER_STOP(lu_blas_omp_cpu)
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
    for (rep = 0; rep < BENCH_REPS; ++rep) {
        cblas_scopy(numMatrices*N*N, atra, 1, inv, 1);

        TIMER_START(chol_gpu)
        // inverse_chol_gpu(inv, N, numMatrices);
        TIMER_STOP(chol_gpu)
        TIMER_ACC(chol_gpu)

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    // calculate error
    BENCH_CLEANUP(chol_gpu)

    // GPU Benchmark 2
    //////////////////
    // Build benchmark data
    BENCH_SETUP(gauss_kernel_gpu)

    // Compute inverses
    //gpuErrchk( cudaProfilerStart() );
    for (rep = 0; rep < BENCH_REPS; ++rep) {
        cblas_scopy(numMatrices*N*N, atra, 1, reconstr, 1);

        TIMER_START(gauss_kernel_gpu)
        inverse_gauss_kernel_gpu(handle, N, reconstr, inv, numMatrices);
        TIMER_STOP(gauss_kernel_gpu)
        TIMER_ACC(gauss_kernel_gpu)

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
    //gpuErrchk( cudaProfilerStop() );

    // calculate error
    BENCH_CLEANUP(gauss_kernel_gpu)

    // GPU Benchmark 3
    //////////////////
    // Build benchmark data
    BENCH_SETUP(gauss_batched_gpu)

    for (rep = 0; rep < BENCH_REPS; ++rep) {
        cblas_scopy(numMatrices*N*N, atra, 1, reconstr, 1);

        TIMER_START(gauss_batched_gpu)
        inverse_gauss_batched_gpu(handle, N, reconstr, inv, numMatrices);
        TIMER_STOP(gauss_batched_gpu)
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

    for (rep = 0; rep < BENCH_REPS; ++rep) {
        cblas_scopy(numMatrices*N*N, atra, 1, reconstr, 1);

        TIMER_START(lu_cuda_batched_gpu)
        inverse_lu_cuda_batched_gpu(handle, N, reconstr, inv, numMatrices);
        TIMER_STOP(lu_cuda_batched_gpu)
        TIMER_ACC(lu_cuda_batched_gpu)

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    BENCH_CLEANUP(lu_cuda_batched_gpu)

    BENCH_REPORT_ERROR(lu_blas_cpu);
    BENCH_REPORT_ERROR(lu_blas_omp_cpu);
    BENCH_REPORT_ERROR(chol_gpu);
    BENCH_REPORT_ERROR(gauss_kernel_gpu);
    BENCH_REPORT_ERROR(gauss_batched_gpu);
    BENCH_REPORT_ERROR(lu_cuda_batched_gpu);

    BENCH_REPORT_TIME(lu_blas_cpu);
    BENCH_REPORT_TIME(lu_blas_omp_cpu);
    BENCH_REPORT_TIME(chol_gpu);
    BENCH_REPORT_TIME(gauss_kernel_gpu);
    BENCH_REPORT_TIME(gauss_batched_gpu);
    BENCH_REPORT_TIME(lu_cuda_batched_gpu);

    cublasErrchk( cublasDestroy(handle) );

    free(workspace);
    free(reconstr);
    free(inv);
    free(atra);
}

int main(int argc, char const *argv[]) {
    ensure(argc >= 2, "Usage: inverse_bench TEST_FILE [-d]");

    bool detailedReporting = (argc >= 3) && !strncmp("-d", argv[2], 2);

    int numMatrices;
    int M;
    int N;

    Array a;

    readMatricesFile(argv[1], &numMatrices, &M, &N, &a);

    bench_parallel(numMatrices, M, N, a, detailedReporting);

    cudaDeviceReset();

    return 0;
}
