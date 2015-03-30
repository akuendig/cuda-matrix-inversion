#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#ifdef __APPLE__
    #include <Accelerate/Accelerate.h>
#else
    #include <cblas.h>
    #include <clapack.h>
#endif

#include "../include/types.h"
#include "../include/helper.h"
#include "../include/inverse.h"

#define MAX_MATRIX_BYTE_READ 67108864

void mean(Array a, Array mean, const int m, const int n) {
    for (int i = 0; i < n; ++i) {
        mean[i] = cblas_sasum(m, &a[i*m], 1);
    }

    cblas_sscal(n, 1.0f/float(m), mean, 1);
}

void sub_each(Array a, Array vec, const int m, const int n) {
    for (int i = 0; i < m; ++i) {
        cblas_saxpy(n, -1.f, vec, 1, &a[i], m);
    }
}

void covariance(Array a, Array cov, Array mu, int m, int n) {
    printf("m(%d) n(%d)\n", m, n);

    printf("Matrix\n");
    printMatrix(a, m, n);

    mean(a, mu, m, n);
    printf("Mean\n");
    printMatrix(mu, 1, n);

    sub_each(a, mu, m, n);

    cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans, n, m, 1, a, m, 0, cov, n);

    printf("A matrix\n");
    printMatrix(a, m, n);

    printf("Covariance matrix\n");
    printMatrix(cov, n, n);
}

int main(int argc, char const *argv[]) {
    const char *directory = "tests/simpleMean";
    char filePath[1024];

    int numMatrices;
    int m;
    int n;

    Array a;

    snprintf(filePath, 1024, "%s/mean.mats", directory);
    readMatricesFile(filePath, &numMatrices, &m, &n, &a);

    Array mu = (Array)malloc(m*sizeof(ELEMENT_TYPE));
    Array cov = (Array)malloc(n*n*sizeof(ELEMENT_TYPE));

    for (int i = 0; i < numMatrices; ++i) {
        Array current_a = a + (i * m * n);

        covariance(current_a, cov, mu, m, n);
    }

    return 0;
}