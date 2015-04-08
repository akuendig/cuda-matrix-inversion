#ifndef HEADER_INVERSE_CPU_INCLUDED
#define HEADER_INVERSE_CPU_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

void inverse_lu_blas(Array a, Array workspace, int N);
void inverse_lu_blas_omp(Array as, Array workspaces, int N, int batchSize);
void inverse_chol_blas(Array a, int N);

void inverse_lu_blas(Array a, Array workspace, int n);
void inverse_chol_blas(Array a, int n);
void inverse_chol_gpu(Array a, int n);
#ifdef __cplusplus
}
#endif // __cplusplus

#endif