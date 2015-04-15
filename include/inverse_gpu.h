#ifndef HEADER_INVERSE_GPU_INCLUDED
#define HEADER_INVERSE_GPU_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
void inverse_gauss_kernel_gpu(cublasHandle_t handle, int n, Array As, Array aInvs, int batchSize);
void inverse_gauss_batched_gpu(cublasHandle_t handle, int n, Array As, Array aInvs, int batchSize);
void inverse_lu_cuda_batched_gpu(cublasHandle_t handle, int n, Array As, Array aInvs, int batchSize);

void inverse_gauss_batched_device(cublasHandle_t handle, int n, Array *devAs, Array *devAInvs, int batchSize);
void inverse_lu_cuda_batched_device(cublasHandle_t handle, int n, Array *devAs, Array *devAInvs, int batchSize);
#ifdef __cplusplus
}
#endif // __cplusplus

#endif