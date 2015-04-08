#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>
#include <errno.h>
#include <time.h>
#include <omp.h>

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
#include "../include/helper_cpu.h"
#include "../include/inverse_cpu.h"


static void mean(Array a, Array mean, const int M, const int N) {
    int i;

    for (i = 0; i < N; ++i) {
        mean[i] = cblas_sasum(M, &a[i*M], 1);
    }

    cblas_sscal(N, 1.0f/((float)M), mean, 1);
}

static void sub_each(Array a, Array vec, const int M, const int N) {
    int i;

    for (i = 0; i < M; ++i) {
        cblas_saxpy(N, -1.f, vec, 1, &a[i], M);
    }
}

static void covariance(Array a, Array cov, Array mu, int M, int N) {
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

void inverse_lu_blas_omp(Array as, int N, int batchSize) {
    int i;

    #pragma omp parallel
    {
        const int ArraySize = sizeof(DataType)*N*N;
        Array workspace = (Array)malloc(ArraySize);
        ensure(workspace, "Could not allocate workspace for matrix inversion");

        #pragma omp parallel for
        for (i = 0; i < batchSize; ++i) {
            inverse_lu_blas(as+(i*ArraySize), workspace, N);
        }

        free(workspace);
    }
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
