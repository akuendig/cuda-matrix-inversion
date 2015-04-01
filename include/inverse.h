#ifndef HEADER_INVERSE_INCLUDED
#define HEADER_INVERSE_INCLUDED

void inverse_lu_blas(Array a, Array workspace, int n);
void inverse_chol_blas(Array a, int n);
void inverse_chol_gpu(Array a, int n);
void inverse_gauss_gpu(Array a, int n);
void inverse_gauss_batched_gpu(cublasHandle_t handle, int n, Array *a, Array *aInv, int batchSize);

#endif