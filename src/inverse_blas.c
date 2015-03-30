#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #include <clapack.h>
#endif

#include "../include/types.h"
#include "../include/helper.h"

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
    ensure(!error, "Error code %d in cholesky factorization", error);
    spotri_("U", &N, a, &N, &error);
    ensure(!error, "Error code %d in cholesky inversion", error);
}
