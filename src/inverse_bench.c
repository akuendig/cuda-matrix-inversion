#include <stdio.h>
#include <stdlib.h>
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

int main(int argc, char const *argv[]) {
    ensure(argc >= 2, "Usage: inverse_bench TEST_FILE");

    int numMatrices;
    int M;
    int N;

    Array a;

    readMatricesFile(argv[1], &numMatrices, &M, &N, &a);

    Array mu = (Array)malloc(M*sizeof(DataType));
    Array atra = (Array)malloc(N*N*sizeof(DataType));
    Array inv = (Array)malloc(N*N*sizeof(DataType));
    Array reconstr = (Array)malloc(N*N*sizeof(DataType));

    DataType total;
    int i, rep;
#ifdef __APPLE__
    clock_t start, diff, cycle_sum_chol_cpu, cycle_sum_chol_gpu = 0;
#else
    struct timespec ts_start, ts_end, ts_sum_chol_cpu = { 0 }, ts_sum_chol_gpu = { 0 };
#endif

    for (i = 0; i < numMatrices; ++i) {
        Array current_a = a + (i * M * N);

        cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans, N, M, 1, current_a, M, 0, atra, N);
        fill_sym(atra, N, N);

        // CPU Benchmark
        ////////////////
        for (rep = 0; rep < 10; ++rep) {
            cblas_scopy(N*N, atra, 1, inv, 1);

#ifdef __APPLE__
            start = clock();
#else
            clock_gettime(CLOCK_MONOTONIC, &ts_start);
#endif

            inverse_chol_blas(inv, N);

#ifdef __APPLE__
            diff = clock() - start;
            cycle_sum_chol_cpu += diff;
            printf("Inversion using BLAS cholesky took %lu cycles\n", diff);
#else
            clock_gettime(CLOCK_MONOTONIC, &ts_end);
            time_sub(&ts_end, &ts_start);
            time_add(&ts_sum_chol_cpu, &ts_end);
            printf("Inversion using BLAS cholesky took %lu seconds and %lu nanoseconds\n", ts_end.tv_sec, ts_end.tv_nsec);
#endif
        }

        cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper, M, N, 1.f, inv, N, atra, N, 0, reconstr, N);
        mat_sum(reconstr, M, N, &total);

        printf("Inversion using BLAS cholesky L1 error %f\n", total);

        // GPU Benchmark
        ////////////////
        for (rep = 0; rep < 10; ++rep) {
            cblas_scopy(N*N, atra, 1, inv, 1);

#ifdef __APPLE__
            start = clock();
#else
            clock_gettime(CLOCK_MONOTONIC, &ts_start);
#endif

            inverse_chol_gpu(inv, N);

#ifdef __APPLE__
            diff = clock() - start;
            cycle_sum_chol_gpu += diff;
            printf("Inversion using GPU cholesky took %lu cycles\n", diff);
#else
            clock_gettime(CLOCK_MONOTONIC, &ts_end);
            time_sub(&ts_end, &ts_start);
            time_add(&ts_sum_chol_gpu, &ts_end);
            printf("Inversion using GPU cholesky took %lu seconds and %lu nanoseconds\n", ts_end.tv_sec, ts_end.tv_nsec);
#endif
        }

        cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper, M, N, 1.f, inv, N, atra, N, 0, reconstr, N);
        mat_sum(reconstr, M, N, &total);

        printf("Inversion using GPU cholesky L1 error %f\n", total);
    }

#ifdef __APPLE__
    printf("Execution time for BLAS cholesky on average:\t%lu cycles\n", cycle_sum_chol_cpu/numMatrices/rep);
    printf("Execution time for GPU cholesky on average:\t%lu cycles\n", cycle_sum_chol_gpu/numMatrices/rep);
#else
    time_div(&ts_sum_chol_cpu, numMatrices*rep);
    time_div(&ts_sum_chol_gpu, numMatrices*rep);
    printf("Execution time for BLAS cholesky on average:\t%lu seconds and %lu nanoseconds (%lu ms)\n",
        ts_sum_chol_cpu.tv_sec, ts_sum_chol_cpu.tv_nsec, ts_sum_chol_cpu.tv_nsec/1000000);
    printf("Execution time for GPU cholesky on average:\t%lu seconds and %lu nanoseconds (%lu ms)\n",
        ts_sum_chol_gpu.tv_sec, ts_sum_chol_gpu.tv_nsec, ts_sum_chol_gpu.tv_nsec/1000000);
#endif

    return 0;
}
