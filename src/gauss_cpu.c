#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>
#include <omp.h>

#include <cblas.h>
#ifdef __APPLE__
    #include <lapacke.h>
#else
    #include <clapack.h>
#endif // __APPLE__

#include "../include/types.h"
#include "../include/helper_cpu.h"
#include "../include/inverse_cpu.h"
#include "../include/gauss_cpu.h"

// Calculates the mean of the matrix set {A, B, C, D}.
// Mean = A*(B+C)^{-1}*D
// As       batchSize x n x 1
// Bs       batchSize x n x n
// Cs       batchSize x n x 1
// Ds       batchSize x n x 1
// Means    batchSize x n x 1
// Means is assumed to be already allocated.
void calcluateMeanCPU(
    const int n,
    Array As,
    Array Bs,
    Array Cs,
    Array Ds,
    Array Means,
    const int batchSize) {

    int i, j;

    #pragma omp parallel shared(As, Bs, Cs, Ds, Means) private(j)
    {
        Array workspace = (Array)malloc(sizeof(DataType)*n*n);
        ensure(workspace, "Could not allocate workspace for matrix inversion");

        #pragma omp for schedule(dynamic, 8)
        for (i = 0; i < batchSize; ++i) {
            Array currentA = As+(i*n);
            Array currentB = Bs+(i*n*n);
            Array currentC = Cs+(i*n);
            Array currentD = Ds+(i*n);

            // Update diagonal
            for (j = 0; j < n; ++j) {
                currentB[j + j*n] += currentC[j];
            }

            // inverse_lu_blas(currentB, workspace, n);
            inverse_chol_blas(currentB, n);

            cblas_ssymv (CblasColMajor, CblasUpper,
                n, // rows in A
                1, // alpha
                currentB, // A
                n, // LDA
                currentD, // x
                1, // inc x
                0, // beta
                currentC, // y
                1 // inc y
            );

            Means[i] = cblas_sdot (
                n, // rows in x
                currentA, // x
                1, // inc x
                currentC, // y
                1 // inc y
            );
        }

        free(workspace);
    }
}

// Calculates the variance of the matrix set {A, B, C, E}.
// Var = E-AT*(B+C)^{-1}*A
// As       batchSize x n x 1
// Bs       batchSize x n x n
// Cs       batchSize x n x 1
// Es       batchSize x 1 x 1
// Variances    batchSize x 1 x 1
// Variances is assumed to be already allocated.
//
// Bs and Cs are destroyed
void calcluateVarianceCPU(
    const int n,
    Array As,
    Array Bs,
    Array Cs,
    Array Es,
    Array Variances,
    const int batchSize) {

    int i, j;

    #pragma omp parallel shared(As, Bs, Cs, Es, Variances) private(j)
    {
        Array workspace = (Array)malloc(sizeof(DataType)*n*n);
        ensure(workspace, "Could not allocate workspace for matrix inversion");

        #pragma omp for schedule(dynamic, 16)
        for (i = 0; i < batchSize; ++i) {
            Array currentA = As+(i*n);
            Array currentB = Bs+(i*n*n);
            Array currentC = Cs+(i*n);

            // Update diagonal
            for (j = 0; j < n; ++j) {
                currentB[j + j*n] += currentC[j];
            }

            // inverse_lu_blas(currentB, workspace, n);
            inverse_chol_blas(currentB, n);

            cblas_ssymv (CblasColMajor, CblasUpper,
                n, // rows in A
                1, // alpha
                currentB, // A
                n, // LDA
                currentA, // x
                1, // inc x
                0, // beta
                currentC, // y
                1 // inc y
            );

            Variances[i] = Es[i] + cblas_sdot (
                n, // rows in x
                currentA, // x
                1, // inc x
                currentC, // y
                1 // inc y
            );
        }

        free(workspace);
    }
}