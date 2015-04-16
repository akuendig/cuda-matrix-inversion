#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda.h>
#include "cublas_v2.h"

#include "../../include/types.h"
#include "../../include/helper_cpu.h"
#include "../../include/helper_gpu.h"
#include "../../include/timer.h"
#include "../../include/inverse_cpu.h"
#include "../../include/inverse_gpu.h"

// Inverts `a` by inplace factorizing `a` and then inverting it into `aInv`.
extern "C" void inverse_lu_cuda_batched_device(cublasHandle_t handle, int N, Array *devAs, Array *devAInvs, int batchSize) {
    int i;
    int *PivotArray;
    int *infoArray;

    gpuErrchk( cudaHostAlloc((void**)&PivotArray, sizeof(int)*batchSize*N, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc((void**)&infoArray, sizeof(int)*batchSize, cudaHostAllocDefault) );

    cublasErrchk( cublasSgetrfBatched(handle,
        N, // number of rows and columns of Aarray[i].
        devAs, // array of pointers to <type> array, with each array of dimension n*n with lda>=max(1,n).
        N, // leading dimension of two-dimensional array used to store each matrix Aarray[i].
        PivotArray, // array of size n*batchSize that contains the pivoting sequence of each factorization of Aarray[i] stored in a linear fashion. If PivotArray is nil, pivoting is disabled.
        infoArray, // array of size batchSize that info(=infoArray[i]) contains the information of inversion of A[i].
                   // If info=0, the execution is successful.
                   // If info = k, U(k,k) is 0. The U is exactly singular and the inversion failed.
        batchSize) // number of pointers contained in A
    );

    for (i = 0; i < batchSize; ++i) {
        ensure(!infoArray[i], "Error during lu decomposition of batched matrix %d", i);
    }

    cublasErrchk( cublasSgetriBatched(handle,
        N, // number of rows and columns of Aarray[i].
        const_cast<const float**>(devAs),  // array of pointers to <type> array, with each array of dimension n*n with lda>=max(1,n).
        N, // leading dimension of two-dimensional array used to store each matrix Aarray[i].
        PivotArray,  // array of size n*batchSize that contains the pivoting sequence of each factorization of Aarray[i] stored in a linear fashion. If PivotArray is nil, pivoting is disabled.
        devAInvs, // array of pointers to <type> array, with each array of dimension n*n with ldc>=max(1,n).
        N, // leading dimension of two-dimensional array used to store each matrix Carray[i].
        infoArray, // array of size batchSize that info(=infoArray[i]) contains the information of inversion of A[i].
                   // If info=0, the execution is successful.
                   // If info = k, U(k,k) is 0. The U is exactly singular and the inversion failed.
        batchSize)  // number of pointers contained in A
    );

    for (i = 0; i < batchSize; ++i) {
        ensure(!infoArray[i], "Error during inversion of batched matrix %d", i);
    }

    gpuErrchk( cudaFreeHost((void*)infoArray) );
    gpuErrchk( cudaFreeHost((void*)PivotArray) );
}

extern "C" void inverse_lu_cuda_batched_gpu(
        cublasHandle_t handle,
        int n,
        Array As,
        Array aInvs,
        int batchSize) {

    Array *devAs;
    size_t pitchAs;
    Array *devAInvs;
    size_t pitchAInvs;

#ifdef DETAILED_LOGGING
    TIMER_INIT(inverse_lu_cuda_batched_gpu_mem_htod)
    TIMER_INIT(inverse_lu_cuda_batched_gpu_ker)
    TIMER_INIT(inverse_lu_cuda_batched_gpu_mem_dtoh)
#endif // DETAILED_LOGGING

    const size_t ArraySize = sizeof(DataType) * n * n;

    gpuErrchk( cudaHostAlloc((void**)&devAs, sizeof(Array)*batchSize, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc((void**)&devAInvs, sizeof(Array)*batchSize, cudaHostAllocDefault) );

    gpuErrchk( batchedCudaMalloc(devAs, &pitchAs, ArraySize, batchSize) );
    gpuErrchk( batchedCudaMalloc(devAInvs, &pitchAInvs, ArraySize, batchSize) );

#ifdef DETAILED_LOGGING
    TIMER_START(inverse_lu_cuda_batched_gpu_mem_htod)
#endif // DETAILED_LOGGING

    gpuErrchk( cudaMemcpy2D(devAs[0], pitchAs, As, ArraySize, ArraySize, batchSize,
                cudaMemcpyHostToDevice) );

#ifdef DETAILED_LOGGING
    TIMER_STOP(inverse_lu_cuda_batched_gpu_mem_htod)
    TIMER_START(inverse_lu_cuda_batched_gpu_ker)
#endif // DETAILED_LOGGING

    inverse_lu_cuda_batched_device(handle, n, devAs, devAInvs, batchSize);

#ifdef DETAILED_LOGGING
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    TIMER_STOP(inverse_lu_cuda_batched_gpu_ker)
    TIMER_START(inverse_lu_cuda_batched_gpu_mem_dtoh)
#endif // DETAILED_LOGGING

    gpuErrchk( cudaMemcpy2D(aInvs, ArraySize, devAInvs[0], pitchAInvs, ArraySize, batchSize,
                cudaMemcpyDeviceToHost) );

#ifdef DETAILED_LOGGING
    TIMER_STOP(inverse_lu_cuda_batched_gpu_mem_dtoh)

    TIMER_LOG(inverse_lu_cuda_batched_gpu_mem_htod, batchSize, n)
    TIMER_LOG(inverse_lu_cuda_batched_gpu_ker, batchSize, n)
    TIMER_LOG(inverse_lu_cuda_batched_gpu_mem_dtoh, batchSize, n)
#endif // DETAILED_LOGGING

    gpuErrchk( cudaFree((void*)devAs[0]) );
    gpuErrchk( cudaFree((void*)devAInvs[0]) );
    gpuErrchk( cudaFreeHost((void*)devAs) );
    gpuErrchk( cudaFreeHost((void*)devAInvs) );
}
