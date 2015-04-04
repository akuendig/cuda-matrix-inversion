#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <errno.h>
#include <time.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #ifdef __cplusplus
    extern "C" {
    #endif
    #include <cblas.h>
    #include <clapack.h>
    #ifdef __cplusplus
    }
    #endif
#endif

#include "../include/types.h"
#include "../include/helper.h"
#include "../include/inverse.h"

#define MAX_MATRIX_BYTE_READ 67108864
#define BENCH_REPS 1

void mean(Array a, Array mean, const int M, const int N) {
    int i;

    for (i = 0; i < N; ++i) {
        mean[i] = cblas_sasum(M, &a[i*M], 1);
    }

    cblas_sscal(N, 1.0f/((float)M), mean, 1);
}

void sub_each(Array a, Array vec, const int M, const int N) {
    int i;

    for (i = 0; i < M; ++i) {
        cblas_saxpy(N, -1.f, vec, 1, &a[i], M);
    }
}

void covariance(Array a, Array cov, Array mu, int M, int N) {
    mean(a, mu, M, N);
    sub_each(a, mu, M, N);
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans, N, M, 1, a, M, 0, cov, N);
}

/*
 * Source: http://stackoverflow.com/questions/3519959/computing-the-inverse-of-a-matrix-using-lapack-in-c
 *
 */

void inverse_lu_blas(Array a, Array workspace, int N) {
    int *pivot = (int*)malloc(N*sizeof(int));
    int workspace_size = N*N;
    int error;

    sgetrf_(&N, &N, a, &N, pivot, &error);
    ensure(!error, "Error code %d in LU-decomposition", error);
    sgetri_(&N, a, &N, pivot, workspace, &workspace_size, &error);
    ensure(!error, "Error code %d in LU-inversion", error);

    free(pivot);
}

// Result is stored in the lower triangular part of a.
void inverse_chol_blas(Array a, int N) {
    int error;

    spotrf_("U", &N, a, &N, &error);
    // printMatrix(a, N, N);
    ensure(!error, "Error code %d in cholesky factorization", error);
    spotri_("U", &N, a, &N, &error);
    // printMatrix(a, N, N);
    ensure(!error, "Error code %d in cholesky inversion", error);
}

void fill_sym(Array a, int M, int N) {
    int i, j;

    for (i = 0; i < N-1; ++i) {
        for (j = i+1; j < M; ++j) {
            a[i*M + j] = a[j*M + i];
        }
    }
}

void mat_sum(Array a, int M, int N, DataType *total) {
    *total = cblas_sasum(M*N, a, 1);
}

#define BILLION 1000000000
void time_add(struct timespec *t1, const struct timespec *t2) {
    t1->tv_sec += t2->tv_sec;
    t1->tv_nsec += t2->tv_nsec;

    if (t1->tv_nsec >= BILLION) {
        t1->tv_nsec -= BILLION;
        t1->tv_sec++;
    }
}

void time_sub(struct timespec *t1, const struct timespec *t2) {
    if (t1->tv_nsec < t2->tv_nsec) {
        ensure(t1->tv_sec >= 1, "No negative time possible");

        t1->tv_sec -= 1;
        t1->tv_nsec += BILLION;
    }

    ensure(t1->tv_sec >= t2->tv_sec, "No negative time possible");
    t1->tv_nsec -= t2->tv_nsec;
    t1->tv_sec -= t2->tv_sec;
}

void time_div(struct timespec *t1, double div) {
    double sec = t1->tv_sec / div;
    double nsec = (sec - floor(sec))*BILLION + t1->tv_nsec / div;

    t1->tv_sec = floor(sec);
    t1->tv_nsec = floor(nsec);
}

double time_to_ms(struct timespec *t1) {
    return t1->tv_sec*1000.0 + t1->tv_nsec/1000.0/1000.0;
}

#ifdef __APPLE__
#define TIMER_START(name) start = clock();
#define TIMER_STOP(name) \
    diff = clock() - start; \
    cycle_sum_##name += diff; \
    if (detailed) { printf("Execution time of " #name ": %lucycles \n", diff); }
#else
#define TIMER_START(name) clock_gettime(CLOCK_MONOTONIC, &ts_start);
#define TIMER_STOP(name) \
    clock_gettime(CLOCK_MONOTONIC, &ts_end); \
    time_sub(&ts_end, &ts_start); \
    time_add(&ts_sum_##name, &ts_end); \
    if (detailed) { printf("Execution time of " #name ": %.4fms \n", time_to_ms(&ts_end)); }
#endif

void bench_parallel(int numMatrices, int M, int N, Array a, bool detailed) {
    cublasHandle_t handle;

    Array atra = (Array)malloc(numMatrices*N*N*sizeof(DataType));
    Array inv = (Array)malloc(numMatrices*N*N*sizeof(DataType));
    Array reconstr = (Array)malloc(numMatrices*N*N*sizeof(DataType));
    Array workspace = (Array)malloc(N*N*sizeof(DataType));

    DataType
        total_chol_cpu,
        total_chol_gpu,
        total_gauss_gpu,
        total_gauss_batched_gpu,
        total_lu_cuda_batched_gpu;
    int i, rep;
#ifdef __APPLE__
    clock_t start, diff,
        cycle_sum_chol_cpu = 0,
        cycle_sum_chol_cpu_naive = 0,
        cycle_sum_chol_gpu = 0,
        cycle_sum_gauss_gpu = 0,
        cycle_sum_gauss_batched_gpu = 0,
        cycle_sum_lu_cuda_batched_gpu = 0;
#else
    struct timespec ts_start, ts_end,
        ts_sum_chol_cpu = { 0 },
        ts_sum_chol_cpu_naive = { 0 },
        ts_sum_chol_gpu = { 0 },
        ts_sum_gauss_gpu = { 0 },
        ts_sum_gauss_batched_gpu = { 0 },
        ts_sum_lu_cuda_batched_gpu = { 0 };
#endif

    // CPU Benchmark
    ////////////////
    for (i = 0; i < numMatrices; ++i) {
        Array current_a = a + (i * M * N);
        Array current_atra = atra + (i * N * N);

        cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans,
            N, M, 1, current_a, M, 0, current_atra, N);
        fill_sym(current_atra, N, N);
    }

    for (rep = 0; rep < BENCH_REPS; ++rep) {
        cblas_scopy(numMatrices*N*N, atra, 1, inv, 1);

        TIMER_START()
        for (i = 0; i < numMatrices; ++i) {
            Array current_atra = atra + (i * N * N);
            Array current_inv = inv + (i * N * N);

            inverse_lu_blas(current_inv, workspace, N);
        }
        TIMER_STOP(chol_cpu)
    }

    for (i = 0; i < numMatrices; ++i) {
        Array current_atra = atra + (i * N * N);
        Array current_inv = inv + (i * N * N);
        Array current_rec = reconstr + (i * N * N);

        // Try to get identity matrix
        cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper,
            M, N, 1.f, current_inv, N, current_atra, N, 0, current_rec, N);
        // Calculate the distance to real identity matrix
        mat_sum(current_rec, M, N, &total_chol_cpu);

        printf("L1 error for chol_cpu: %f\n", total_chol_cpu);
    }

    // GPU Benchmark 1
    //////////////////
    // Build benchmark data
    for (i = 0; i < numMatrices; ++i) {
        Array current_a = a + (i * M * N);
        Array current_atra = atra + (i * N * N);

        cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans,
            N, M, 1, current_a, M, 0, current_atra, N);
        fill_sym(current_atra, N, N);
    }

    cublasErrchk( cublasCreate(&handle) );

    // Compute inverses
    for (rep = 0; rep < BENCH_REPS; ++rep) {
        cblas_scopy(numMatrices*N*N, atra, 1, inv, 1);

        TIMER_START()
        // inverse_chol_gpu(inv, N, numMatrices);
        TIMER_STOP(chol_gpu)

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    // calculate error
    for (i = 0; i < numMatrices; ++i) {
        Array current_atra = atra + (i * N * N);
        Array current_inv = inv + (i * N * N);
        Array current_rec = reconstr + (i * N * N);

        cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper,
            M, N, 1.f, current_inv, N, current_atra, N, 0, current_rec, N);
        mat_sum(current_rec, M, N, &total_chol_gpu);

        printf("L1 error for chol_gpu: %f\n", total_chol_gpu);
    }

    // GPU Benchmark 2
    //////////////////
    // Build benchmark data
    for (i = 0; i < numMatrices; ++i) {
        Array current_a = a + (i * M * N);
        Array current_atra = atra + (i * N * N);

        cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans,
            N, M, 1, current_a, M, 0, current_atra, N);
        fill_sym(current_atra, N, N);
    }

    // Compute inverses
    //gpuErrchk( cudaProfilerStart() );
    for (rep = 0; rep < BENCH_REPS; ++rep) {
        cblas_scopy(numMatrices*N*N, atra, 1, reconstr, 1);

        TIMER_START()
        inverse_gauss_kernel_gpu(handle, N, reconstr, inv, numMatrices);
        TIMER_STOP(gauss_gpu)

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
    //gpuErrchk( cudaProfilerStop() );

    // calculate error
    for (i = 0; i < numMatrices; ++i) {
        Array current_atra = atra + (i * N * N);
        Array current_inv = inv + (i * N * N);
        Array current_rec = reconstr + (i * N * N);

        cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper,
            M, N, 1.f, current_inv, N, current_atra, N, 0, current_rec, N);
        mat_sum(current_rec, M, N, &total_gauss_gpu);

        printf("L1 error for gauss_gpu: %f\n", total_gauss_gpu);
    }

    // GPU Benchmark 3
    //////////////////
    // Build benchmark data
    for (i = 0; i < numMatrices; ++i) {
        Array current_a = a + (i * M * N);
        Array current_atra = atra + (i * N * N);

        cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans,
            N, M, 1, current_a, M, 0, current_atra, N);
        fill_sym(current_atra, N, N);
    }

    for (rep = 0; rep < BENCH_REPS; ++rep) {
        cblas_scopy(numMatrices*N*N, atra, 1, reconstr, 1);

        TIMER_START()
        inverse_gauss_batched_gpu(handle, N, reconstr, inv, numMatrices);
        TIMER_STOP(gauss_batched_gpu)

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    // calculate error
    for (i = 0; i < numMatrices; ++i) {
        Array current_atra = atra + (i * N * N);
        Array current_inv = inv + (i * N * N);
        Array current_rec = reconstr + (i * N * N);

        cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper,
            M, N, 1.f, current_inv, N, current_atra, N, 0, current_rec, N);
        mat_sum(current_rec, M, N, &total_gauss_batched_gpu);

        printf("L1 error for gauss_batched_gpu: %f\n", total_gauss_batched_gpu);
    }

    // GPU Benchmark 4
    //////////////////
    // Build benchmark data
    for (i = 0; i < numMatrices; ++i) {
        Array current_a = a + (i * M * N);
        Array current_atra = atra + (i * N * N);

        cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans,
            N, M, 1, current_a, M, 0, current_atra, N);
        fill_sym(current_atra, N, N);
    }

    for (rep = 0; rep < BENCH_REPS; ++rep) {
        cblas_scopy(numMatrices*N*N, atra, 1, reconstr, 1);

        TIMER_START()
        inverse_lu_cuda_batched_gpu(handle, N, reconstr, inv, numMatrices);
        TIMER_STOP(lu_cuda_batched_gpu)

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }

    // calculate error
    for (i = 0; i < numMatrices; ++i) {
        Array current_atra = atra + (i * N * N);
        Array current_inv = inv + (i * N * N);
        Array current_rec = reconstr + (i * N * N);

        cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper,
            M, N, 1.f, current_inv, N, current_atra, N, 0, current_rec, N);
        mat_sum(current_rec, M, N, &total_lu_cuda_batched_gpu);

        printf("L1 error for lu_cuda_batched_gpu: %f\n", total_lu_cuda_batched_gpu);
    }

#ifdef __APPLE__
    printf("Total execution time for %d matrices and %d replications of chol_cpu: %lu cycles (%lu cycles average)\n",
        numMatrices, BENCH_REPS, cycle_sum_chol_cpu, cycle_sum_chol_cpu/numMatrices/BENCH_REPS);
    printf("Total execution time for %d matrices and %d replications of chol_gpu: %lu cycles (%lu cycles average)\n",
        numMatrices, BENCH_REPS, cycle_sum_chol_gpu, cycle_sum_chol_gpu/numMatrices/BENCH_REPS);
    printf("Total execution time for %d matrices and %d replications of gauss_gpu: %lu cycles (%lu cycles average)\n",
        numMatrices, BENCH_REPS, cycle_sum_gauss_gpu, cycle_sum_gauss_gpu/numMatrices/BENCH_REPS);
    printf("Total execution time for %d matrices and %d replications of gauss_batched_gpu: %lu cycles (%lu cycles average)\n",
        numMatrices, BENCH_REPS, cycle_sum_gauss_batched_gpu, cycle_sum_gauss_batched_gpu/numMatrices/BENCH_REPS);
    printf("Total execution time for %d matrices and %d replications of lu_cuda_batched_gpu: %lu cycles (%lu cycles average)\n",
        numMatrices, BENCH_REPS, cycle_sum_lu_cuda_batched_gpu, cycle_sum_lu_cuda_batched_gpu/numMatrices/BENCH_REPS);
#else
    printf("Total execution time for %d matrices and %d replications of chol_cpu: %.4f ms (%.4f ms average)\n",
        numMatrices, BENCH_REPS, time_to_ms(&ts_sum_chol_cpu), time_to_ms(&ts_sum_chol_cpu)/numMatrices/BENCH_REPS);
    printf("Total execution time for %d matrices and %d replications of chol_gpu: %.4f ms (%.4f ms average)\n",
        numMatrices, BENCH_REPS, time_to_ms(&ts_sum_chol_gpu), time_to_ms(&ts_sum_chol_gpu)/numMatrices/BENCH_REPS);
    printf("Total execution time for %d matrices and %d replications of gauss_gpu: %.4f ms (%.4f ms average)\n",
        numMatrices, BENCH_REPS, time_to_ms(&ts_sum_gauss_gpu), time_to_ms(&ts_sum_gauss_gpu)/numMatrices/BENCH_REPS);
    printf("Total execution time for %d matrices and %d replications of gauss_batched_gpu: %.4f ms (%.4f ms average)\n",
        numMatrices, BENCH_REPS, time_to_ms(&ts_sum_gauss_batched_gpu), time_to_ms(&ts_sum_gauss_batched_gpu)/numMatrices/BENCH_REPS);
    printf("Total execution time for %d matrices and %d replications of lu_cuda_batched_gpu: %.4f ms (%.4f ms average)\n",
        numMatrices, BENCH_REPS, time_to_ms(&ts_sum_lu_cuda_batched_gpu), time_to_ms(&ts_sum_lu_cuda_batched_gpu)/numMatrices/BENCH_REPS);
#endif

    cublasErrchk( cublasDestroy(handle) );

    free(workspace);
    free(reconstr);
    free(inv);
    free(atra);
}

void bench_sequencial(int numMatrices, int M, int N, Array a, bool detailed) {
    cublasHandle_t handle;
    cublasErrchk( cublasCreate(&handle) );

    Array atra = (Array)malloc(N*N*sizeof(DataType));
    Array inv = (Array)malloc(N*N*sizeof(DataType));
    Array reconstr = (Array)malloc(N*N*sizeof(DataType));
    Array workspace = (Array)malloc(N*N*sizeof(DataType));

    DataType total_chol_cpu, total_chol_gpu, total_gauss_gpu, total_gauss_batched_gpu;
    int i, rep;
#ifdef __APPLE__
    clock_t start, diff,
        cycle_sum_chol_cpu = 0,
        cycle_sum_chol_cpu_naive = 0,
        cycle_sum_chol_gpu = 0,
        cycle_sum_gauss_gpu = 0,
        cycle_sum_gauss_batched_gpu = 0;
#else
    struct timespec ts_start, ts_end,
        ts_sum_chol_cpu = { 0 },
        ts_sum_chol_cpu_naive = { 0 },
        ts_sum_chol_gpu = { 0 },
        ts_sum_gauss_gpu = { 0 },
        ts_sum_gauss_batched_gpu = { 0 };
#endif

    for (i = 0; i < numMatrices; ++i) {
        Array current_a = a + (i * M * N);

        cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans, N, M, 1, current_a, M, 0, atra, N);
        fill_sym(atra, N, N);

        // CPU Benchmark
        ////////////////
        for (rep = 0; rep < BENCH_REPS; ++rep) {
            cblas_scopy(N*N, atra, 1, inv, 1);

            TIMER_START()
            inverse_lu_blas(inv, workspace, N);
            TIMER_STOP(chol_cpu)
        }

        cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper, M, N, 1.f, inv, N, atra, N, 0, reconstr, N);
        mat_sum(reconstr, M, N, &total_chol_cpu);

        printf("Inversion using BLAS LU-decomposition L1 error %f\n", total_chol_cpu);

        // GPU Benchmark 1
        //////////////////
        for (rep = 0; rep < BENCH_REPS; ++rep) {
            cblas_scopy(N*N, atra, 1, inv, 1);

            TIMER_START()
            inverse_chol_gpu(inv, N);
            TIMER_STOP(chol_gpu)

            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
        }

        cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper, M, N, 1.f, inv, N, atra, N, 0, reconstr, N);
        mat_sum(reconstr, M, N, &total_chol_gpu);

        printf("Inversion using GPU cholesky L1 error %f\n", total_chol_gpu);

        ensure(abs(total_chol_gpu-total_chol_cpu) < 2*total_chol_cpu,
            "Error of GPU cholesky (%f) should not be higher than twice the error of CPU (%f)",
            total_chol_gpu,
            total_chol_cpu);

        // GPU Benchmark 2
        //////////////////
        //gpuErrchk( cudaProfilerStart() );
        for (rep = 0; rep < BENCH_REPS; ++rep) {
            cblas_scopy(N*N, atra, 1, reconstr, 1);

            TIMER_START()
            inverse_gauss_kernel_gpu(handle, N, reconstr, inv, 1);
            TIMER_STOP(gauss_gpu)

            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
        }
        //gpuErrchk( cudaProfilerStop() );

        cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper, M, N, 1.f, inv, N, atra, N, 0, reconstr, N);
        mat_sum(reconstr, M, N, &total_gauss_gpu);

        printf("Inversion using GPU gauss kernel batched L1 error %f\n", total_gauss_gpu);

        ensure(abs(total_gauss_gpu-total_chol_cpu) < 100*total_chol_cpu,
            "Error of GPU gauss (%f) should not be higher than 100 times the error of CPU (%f)",
            total_gauss_gpu,
            total_chol_cpu);

        // GPU Benchmark 3
        //////////////////
        for (rep = 0; rep < BENCH_REPS; ++rep) {
            cblas_scopy(N*N, atra, 1, reconstr, 1);

            TIMER_START()
            inverse_gauss_batched_gpu(handle, N, reconstr, inv, 1);
            TIMER_STOP(gauss_batched_gpu)

            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
        }

        cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper, M, N, 1.f, inv, N, atra, N, 0, reconstr, N);
        mat_sum(reconstr, M, N, &total_gauss_batched_gpu);

        printf("Inversion using GPU gauss batched L1 error %f\n", total_gauss_batched_gpu);

        ensure(abs(total_gauss_batched_gpu-total_chol_cpu) < 100*total_chol_cpu,
            "Error of GPU gauss (%f) should not be higher than 100 times the error of CPU (%f)",
            total_gauss_batched_gpu,
            total_chol_cpu);
    }

#ifdef __APPLE__
    printf("Execution time for BLAS cholesky on average:\t%lu cycles\n", cycle_sum_chol_cpu/numMatrices/BENCH_REPS);
    printf("Execution time for GPU cholesky on average:\t%lu cycles\n", cycle_sum_chol_gpu/numMatrices/BENCH_REPS);
    printf("Execution time for GPU gauss on average:\t%lu cycles\n", cycle_sum_gauss_gpu/numMatrices/BENCH_REPS);
    printf("Execution time for GPU gauss batched on average:\t%lu cycles\n", cycle_sum_gauss_batched_gpu/numMatrices/BENCH_REPS);
#else
    time_div(&ts_sum_chol_cpu, numMatrices*BENCH_REPS);
    time_div(&ts_sum_chol_gpu, numMatrices*BENCH_REPS);
    time_div(&ts_sum_gauss_gpu, numMatrices*BENCH_REPS);
    time_div(&ts_sum_gauss_batched_gpu, numMatrices*BENCH_REPS);
    printf("Execution time for BLAS cholesky on average:\t%lu seconds and %lu nanoseconds (%.3f ms)\n",
        ts_sum_chol_cpu.tv_sec, ts_sum_chol_cpu.tv_nsec, ts_sum_chol_cpu.tv_nsec/1000000.f);
    printf("Execution time for GPU cholesky on average:\t%lu seconds and %lu nanoseconds (%.3f ms)\n",
        ts_sum_chol_gpu.tv_sec, ts_sum_chol_gpu.tv_nsec, ts_sum_chol_gpu.tv_nsec/1000000.f);
    printf("Execution time for GPU gauss on average:\t%lu seconds and %lu nanoseconds (%.3f ms)\n",
        ts_sum_gauss_gpu.tv_sec, ts_sum_gauss_gpu.tv_nsec, ts_sum_gauss_gpu.tv_nsec/1000000.f);
    printf("Execution time for GPU gauss batched on average:\t%lu seconds and %lu nanoseconds (%.3f ms)\n",
        ts_sum_gauss_batched_gpu.tv_sec, ts_sum_gauss_batched_gpu.tv_nsec, ts_sum_gauss_batched_gpu.tv_nsec/1000000.f);
#endif

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
