#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "../include/types.h"
#include "../include/helper_cpu.h"
#include "../include/helper_gpu.h"
#include "../include/inverse_cpu.h"
#include "../include/inverse_gpu.h"
#include "../include/timer.h"

#define MAX_THREADS_PER_BLOCK 256
#define MIN_OPS 64
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define LBOUND(start,ops,threadId) (threadId * ops + start)
#define UBOUND(end, start,ops,threadId) MIN((threadId + 1) * ops + start, end)


#ifdef __MACH__
#include <mach/clock.h>
#include <mach/mach.h>
#endif

void current_utc_time(struct timespec *ts) {
 
#ifdef __MACH__ // OS X does not have clock_gettime, use clock_get_time
  clock_serv_t cclock;
  mach_timespec_t mts;
  host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
  clock_get_time(cclock, &mts);
  mach_port_deallocate(mach_task_self(), cclock);
  ts->tv_sec = mts.tv_sec;
  ts->tv_nsec = mts.tv_nsec;
#else
  clock_gettime(CLOCK_REALTIME, ts);
#endif
 
}


__global__
void decompose_cholesky_kernel_device_ops(Array *aInv, int N, int ops) {
    int i, j, row;

    for (row = 0; row < N; row++) {


        if (threadIdx.x == 0) { // computing diagonal elements in the first thread

            aInv[blockIdx.x][row * N + row] = sqrt(aInv[blockIdx.x][row * N + row]);
            //printf("[%d][%d] = sqrt([%d][%d])\n", row, row, row, row);

            for (j = (row + 1); j < N; j++) {
                aInv[blockIdx.x][row * N + j] /= aInv[blockIdx.x][row * N + row]; // divide by diagonal elemnents
                //printf("[%d][%d] /= [%d][%d]\n", j, row, row, row);
            }
        }

        __syncthreads(); // all diagonal elemnents need to be computed

        for (i = LBOUND(row + 1, ops, threadIdx.x); i < UBOUND(N, row + 1, ops, threadIdx.x); i++) {
            for (j = i; j < N; j++) {
                aInv[blockIdx.x][i * N + j] -= aInv[blockIdx.x][row * N + i] * aInv[blockIdx.x][row * N + j];
                //printf("[%d][%d] -= [%d][%d] * [%d][%d]\n", j, i, i, row, j, row);
            }
        }

        __syncthreads(); // compute row by row
    }

    // set zeroes
    for (i = LBOUND(0, ops, threadIdx.x); i < UBOUND(N, 0, ops, threadIdx.x); i++) {
        for (j = 0; j < i; j++) {
            aInv[blockIdx.x][i * N + j] = 0;
        }
    }

}


extern "C" 
void decompose_cholesky_batched_device_ops(cublasHandle_t handle, int N, Array *devAs, Array *devAInvs, int batchSize) {
    int ops = N; //MAX(N * N / MIN(N * N, MAX_THREADS_PER_BLOCK), N);
    int threads = N;
    decompose_cholesky_kernel_device_ops<<< batchSize, threads>>>(devAInvs, N, ops);
}

__global__
void inverse_upper_kernel_device_ops(Array *aInv, int N, int ops) {

    int i, j, row;

    for (row = 0; row < N; row++) {

        if (threadIdx.x == 0) { // re-compute diagonal elements in the first thread
            aInv[blockIdx.x][row * N + row] = 1.0 / aInv[blockIdx.x][row * N + row];
            //printf("[%d][%d] = 1.0 / [%d][%d] = %f\n", row, row, row, row, aInv[blockIdx.x][row * N + row]);

            for (j = (row + 1); j < N; j++) {
                //printf("[%d][%d] = -[%d][%d] * [%d][%d] / [%d][%d] = -%f * %f / %f", j, row, j, row, row, row, j, j, aInv[blockIdx.x][row * N + j], aInv[blockIdx.x][row * N + row], aInv[blockIdx.x][j * N + j]);
                aInv[blockIdx.x][row * N + j] = 0 - aInv[blockIdx.x][row * N + j] * aInv[blockIdx.x][row * N + row] / aInv[blockIdx.x][j * N + j];
                //printf("= %f\n", aInv[blockIdx.x][row * N + j]);
            }
        }

        __syncthreads(); // all diagonal elemnents need to be re-computed

        for (i = LBOUND(row + 2, ops, threadIdx.x); i < UBOUND(N, row + 2, ops, threadIdx.x); i++) {

            for (j = row + 1; j < i; j++) {
                aInv[blockIdx.x][row * N + i] -= aInv[blockIdx.x][j * N + i] * aInv[blockIdx.x][row * N + j] / aInv[blockIdx.x][i * N + i];
                //printf("[%d][%d] -= [%d][%d] * [%d][%d] / [%d][%d]\n", i, row, i, j, j, row, i, i);
            }

            __syncthreads();
        }

        __syncthreads(); // compute row by row
    }
}

extern "C" 
void inverse_upper_batched_device_ops(cublasHandle_t handle, int N, Array *devAs, Array *devAInvs, int batchSize) {
    int ops = N; //MAX(N * N / MIN(N * N, MAX_THREADS_PER_BLOCK), N);
    int threads = N;
    inverse_upper_kernel_device_ops<<< batchSize, threads>>>(devAInvs, N, ops);
}


__global__
void multiply_upper_kernel_device_ops(Array *aInv, int N, int ops) {
    int i, j, row;

    for (row = 0; row < N; row++) {

        for (i = LBOUND(row, ops, threadIdx.x); i < UBOUND(N, row, ops, threadIdx.x); i++) {
            aInv[blockIdx.x][i * N + row] = aInv[blockIdx.x][i * N + i] * aInv[blockIdx.x][row * N + i];
            //printf("[%d][%d] = [%d][%d] * [%d][%d]\n", i, row, i, i, i, row);

            for (j = i + 1; j < N; j++) {
                // use upper matrix as buffer for multiplication
                aInv[blockIdx.x][i * N + row] += aInv[blockIdx.x][i * N + j] * aInv[blockIdx.x][row * N + j];
                //printf("[%d][%d] += [%d][%d] * [%d][%d]\n", i, row, j, i, j, row);
            }
        }

        __syncthreads(); // compute row by row
    }

    __syncthreads();

    // set back to lower matrix
    for (i = LBOUND(0, ops, threadIdx.x); i < UBOUND(N, 0, ops, threadIdx.x); i++) {
        for (j = 0; j < i; j++) {
            aInv[blockIdx.x][j * N + i] = aInv[blockIdx.x][i * N + j];
        }
    }
}

extern "C" 
void multiply_upper_batched_device_ops(cublasHandle_t handle, int N, Array *devAs, Array *devAInvs, int batchSize) {
    int ops = N; //MAX(N * N / MIN(N * N, MAX_THREADS_PER_BLOCK), N);
    int threads = N;
    multiply_upper_kernel_device_ops<<< batchSize, threads>>>(devAInvs, N, ops);
}

extern "C" 
void inverse_cholesky_batched_device_ops(cublasHandle_t handle, int N, Array *devAs, Array *devAInvs, int batchSize) {
    decompose_cholesky_batched_device_ops(handle, N, devAs, devAInvs, batchSize);
    inverse_upper_batched_device_ops(handle, N, devAs, devAInvs, batchSize);
    multiply_upper_batched_device_ops(handle, N, devAs, devAInvs, batchSize);
}

extern "C" 
void inverse_cholesky_batched_gpu_ops(cublasHandle_t handle, int n, Array As, Array aInvs, int batchSize) {

    Array *devAs;
    size_t pitchAs;
    Array *devAInvs;
    size_t pitchAInvs;

#ifdef DETAILED_LOGGING
    TIMER_INIT(inverse_cholesky_batched_gpu_ops_mem_htod)
    TIMER_INIT(inverse_cholesky_batched_gpu_ops_ker)
    TIMER_INIT(inverse_cholesky_batched_gpu_ops_mem_dtoh)
#endif // DETAILED_LOGGING

    const size_t ArraySize = sizeof(DataType) * n * n;

    gpuErrchk( cudaHostAlloc((void**)&devAs, sizeof(Array)*batchSize, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc((void**)&devAInvs, sizeof(Array)*batchSize, cudaHostAllocDefault) );

    gpuErrchk( batchedCudaMalloc(devAs, &pitchAs, ArraySize, batchSize) );
    gpuErrchk( batchedCudaMalloc(devAInvs, &pitchAInvs, ArraySize, batchSize) );

#ifdef DETAILED_LOGGING
    TIMER_START(inverse_cholesky_batched_gpu_ops_mem_htod)
#endif // DETAILED_LOGGING

    gpuErrchk( cudaMemcpy2D(devAInvs[0], pitchAs, As, ArraySize, ArraySize, batchSize,
                cudaMemcpyHostToDevice) );

#ifdef DETAILED_LOGGING
    TIMER_STOP(inverse_cholesky_batched_gpu_ops_mem_htod)
    TIMER_START(inverse_cholesky_batched_gpu_ops_ker)
#endif // DETAILED_LOGGING

    inverse_cholesky_batched_device_ops(handle, n, devAs, devAInvs, batchSize);

#ifdef DETAILED_LOGGING
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    TIMER_STOP(inverse_cholesky_batched_gpu_ops_ker)
    TIMER_START(inverse_cholesky_batched_gpu_ops_mem_dtoh)
#endif // DETAILED_LOGGING

    gpuErrchk( cudaMemcpy2D(aInvs, ArraySize, devAInvs[0], pitchAInvs, ArraySize, batchSize,
                cudaMemcpyDeviceToHost) );

#ifdef DETAILED_LOGGING
    TIMER_STOP(inverse_cholesky_batched_gpu_ops_mem_dtoh)

    TIMER_LOG(inverse_cholesky_batched_gpu_ops_mem_htod)
    TIMER_LOG(inverse_cholesky_batched_gpu_ops_ker)
    TIMER_LOG(inverse_cholesky_batched_gpu_ops_mem_dtoh)
#endif // DETAILED_LOGGING

    gpuErrchk( cudaFree((void*)devAs[0]) );
    gpuErrchk( cudaFree((void*)devAInvs[0]) );
    gpuErrchk( cudaFreeHost((void*)devAs) );
    gpuErrchk( cudaFreeHost((void*)devAInvs) );
}


__global__
void pivot_cholesky_kernel_device(Array *a, int N, int row) {

    //printf("%d\n", row);
    if (threadIdx.x == 0)
        a[blockIdx.x][row * N + row] = sqrt(a[blockIdx.x][row * N + row]); // update diagonal elemnets
    __syncthreads();

    //printf("[%d][%d] = sqrt([%d][%d])\n", row, row, row, row);

    int j = threadIdx.x + row + 1;
    if (j < N) {
        a[blockIdx.x][j * N + row] = 0;
        a[blockIdx.x][row * N + j] /= a[blockIdx.x][row * N + row]; // divide by diagonal elemnents
    }
}


__global__
void decompose_cholesky_kernel_device(Array *a, int N, int row) {

    int i = threadIdx.x + row + 1;

    for (int j = i; j < N; j++) {
        //printf("[%d][%d] -= [%d][%d] x [%d][%d] \n", j, i, i, row, j, row);
        a[blockIdx.x][i * N + j] = a[blockIdx.x][i * N + j] - a[blockIdx.x][row * N + i] * a[blockIdx.x][row * N + j];
    }
}

__global__
void inverse_upper_kernel_device(Array *a, Array *aInv, int N, int row) {
    int i = threadIdx.x;

    aInv[blockIdx.x][i * N + row] = 0 - a[blockIdx.x][i * N + row] *  aInv[blockIdx.x][i * N + i] / a[blockIdx.x][row * N + row]; 
    for (int j = i + 1; j < row; j++) {
        aInv[blockIdx.x][i * N + row] -= a[blockIdx.x][j * N + row] * aInv[blockIdx.x][i * N + j] / a[blockIdx.x][row * N + row];
    }

    if (row == i) {
        //printf("%d\n %f", row, a[blockIdx.x][row * N + row]);
        aInv[blockIdx.x][row * N + row] = 1.0 / a[blockIdx.x][row * N + row];
    }
}

__global__
void multiply_upper_kernel_device(Array *a, Array *aInv, int N, int row) {
    int i = threadIdx.x;

    for (int j = 0; j <= i; j++) {
            aInv[blockIdx.x][i * N + j] += a[blockIdx.x][i * N + row] *  a[blockIdx.x][j * N + row];
            aInv[blockIdx.x][j * N + i] = aInv[blockIdx.x][i * N + j];
            //printf("[%d][%d] += [%d][%d] * [%d][%d] %f\n", i, j, row, i, row, j, a[blockIdx.x][i * N + row] *  a[blockIdx.x][j * N + row]);
    }
}

__global__
void intialize_array(Array *a, int N) {
    for (int j = 0; j < N; j++) {
        a[blockIdx.x][threadIdx.x * N + j] = 0.0;
    }
}


extern "C" 
void inverse_cholesky_batched_device(cublasHandle_t handle, int N, Array *devAs, Array *devAInvs, int batchSize) {
    Array *devATmp;
    size_t pitchATmp;

    gpuErrchk( cudaHostAlloc((void**)&devATmp, sizeof(Array)*batchSize, cudaHostAllocDefault) );
    gpuErrchk( batchedCudaMalloc(devATmp, &pitchATmp, sizeof(DataType) * N * N, batchSize) );


    // Set to zeroes
    intialize_array<<<batchSize, N>>>(devAInvs, N);

    for (int row = 0; row < N; row++) { // loop through each row
        
        // computing diagonal elements
        pivot_cholesky_kernel_device<<<batchSize, (N - row)>>>(devAs, N, row);
        gpuErrchk( cudaPeekAtLastError() );

        // cholesky decomposition
        decompose_cholesky_kernel_device<<<batchSize, (N - row)>>>(devAs, N, row);
        gpuErrchk( cudaPeekAtLastError() );

        // invert the upper
        inverse_upper_kernel_device<<<batchSize, row + 1>>>(devAs, devATmp, N, row);
        gpuErrchk( cudaPeekAtLastError() );

        multiply_upper_kernel_device<<<batchSize, row + 1>>>(devATmp, devAInvs, N, row);
        gpuErrchk( cudaPeekAtLastError() );
    }
    
    gpuErrchk( cudaFree((void*)devATmp[0]) );
    gpuErrchk( cudaFreeHost((void*)devATmp) );
}

extern "C" 
void decompose_cholesky_batched_device(cublasHandle_t handle, int N, Array *devAs, Array *devAInvs, int batchSize) {
    for (int row = 0; row < N; row++) { // loop through each row
        
        // computing diagonal elements
        pivot_cholesky_kernel_device<<<batchSize, (N - row)>>>(devAs, N, row);
        gpuErrchk( cudaPeekAtLastError() );

        // cholesky decomposition
        decompose_cholesky_kernel_device<<<batchSize, (N - row)>>>(devAs, N, row);
        gpuErrchk( cudaPeekAtLastError() );

    }
}



extern "C" 
void inverse_cholesky_batched_gpu(cublasHandle_t handle, int n, Array As, Array aInvs, int batchSize) {

    Array *devAs;
    size_t pitchAs;
    Array *devAInvs;
    size_t pitchAInvs;

#ifdef DETAILED_LOGGING
    TIMER_INIT(decompose_cholesky_batched_gpu_mem_htod)
    TIMER_INIT(decompose_cholesky_batched_gpu_ker)
    TIMER_INIT(decompose_cholesky_batched_gpu_mem_dtoh)
#endif // DETAILED_LOGGING

    const size_t ArraySize = sizeof(DataType) * n * n;

    gpuErrchk( cudaHostAlloc((void**)&devAs, sizeof(Array)*batchSize, cudaHostAllocDefault) );
    gpuErrchk( cudaHostAlloc((void**)&devAInvs, sizeof(Array)*batchSize, cudaHostAllocDefault) );

    gpuErrchk( batchedCudaMalloc(devAs, &pitchAs, ArraySize, batchSize) );
    gpuErrchk( batchedCudaMalloc(devAInvs, &pitchAInvs, ArraySize, batchSize) );

#ifdef DETAILED_LOGGING
    TIMER_START(decompose_cholesky_batched_gpu_mem_htod)
#endif // DETAILED_LOGGING

    gpuErrchk( cudaMemcpy2D(devAs[0], pitchAs, As, ArraySize, ArraySize, batchSize,
                cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy2D(devAInvs[0], pitchAs, As, ArraySize, ArraySize, batchSize,
                cudaMemcpyHostToDevice) );

#ifdef DETAILED_LOGGING
    TIMER_STOP(decompose_cholesky_batched_gpu_mem_htod)
    TIMER_START(decompose_cholesky_batched_gpu_ker)
#endif // DETAILED_LOGGING

    inverse_cholesky_batched_device(handle, n, devAs, devAInvs, batchSize);

#ifdef DETAILED_LOGGING
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    TIMER_STOP(decompose_cholesky_batched_gpu_ker)
    TIMER_START(decompose_cholesky_batched_gpu_mem_dtoh)
#endif // DETAILED_LOGGING

    gpuErrchk( cudaMemcpy2D(As, ArraySize, devAs[0], pitchAInvs, ArraySize, batchSize,
                cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaMemcpy2D(aInvs, ArraySize, devAInvs[0], pitchAInvs, ArraySize, batchSize,
                cudaMemcpyDeviceToHost) );

#ifdef DETAILED_LOGGING
    TIMER_STOP(decompose_cholesky_batched_gpu_mem_dtoh)

    TIMER_LOG(decompose_cholesky_batched_gpu_mem_htod)
    TIMER_LOG(decompose_cholesky_batched_gpu_ker)
    TIMER_LOG(decompose_cholesky_batched_gpu_mem_dtoh)
#endif // DETAILED_LOGGING

    gpuErrchk( cudaFree((void*)devAs[0]) );
    gpuErrchk( cudaFree((void*)devAInvs[0]) );
    gpuErrchk( cudaFreeHost((void*)devAs) );
    gpuErrchk( cudaFreeHost((void*)devAInvs) );
}




extern "C" void inverse_chol_gpu(Array a, int n) {
    /*
    Array a_dev;

    size_t matrixSize = n*n * sizeof(DataType);
    gpuErrchk( cudaMalloc(&a_dev, matrixSize) );
    gpuErrchk( cudaMemcpy(a_dev, a, matrixSize, cudaMemcpyHostToDevice) );

    decomposeCholeskyGPU(a_dev, n);
    inverseLowerGPU(a_dev, n);
    multiplyLowerGPU(a_dev, n);

    gpuErrchk( cudaMemcpy(a, a_dev, matrixSize, cudaMemcpyDeviceToHost) );
    gpuErrchk( cudaFree(a_dev) );
    */
}



int main2(int argc, char const *argv[]) {
    cublasHandle_t handle;
    const char *directory = "tests/inverse_100_128x128";
    char filePath[1024];
    int numMatrices, m, n;
    Array a, ainv, atest;

    //Array *devAs;
    //size_t pitchAs;
    //Array *devAInvs;
    //size_t pitchAInvs;

    //struct timespec timer_start, timer_end;

    //int batchSize = 1;

    cublasCreate(&handle);


    snprintf(filePath, 1024, "%s/a.mats", directory);
    readMatricesFile(filePath, &numMatrices, &m, &n, &a);
    readMatricesFile(filePath, &numMatrices, &m, &n, &ainv);
    //printMatrix(a, m, n);


    snprintf(filePath, 1024, "%s/ainv.mats", directory);
    readMatricesFile(filePath, &numMatrices, &m, &n, &atest);
    //printMatrix(a, m, n);

    //current_utc_time(&timer_start);
    //inverse_cholesky_batched_gpu(handle, n, a, ainv, 1);
    //printMatrix(ainv, m, n);

    inverse_cholesky_batched_gpu(handle, n, a, ainv, 1);
    //current_utc_time(&timer_end);
    //printf("New Cholesky: %fms\n", (double) (timer_end.tv_nsec - timer_start.tv_nsec) / 1000);

    //printMatrix(a, m, n);
    //printMatrix(ainv, m, n);
    /*
    // old code
    matrixSize = m * n * sizeof(DataType);

    gpuErrchk( cudaMalloc(&a_dev, matrixSize) );
    gpuErrchk( cudaMemcpy(a_dev, a, matrixSize, cudaMemcpyHostToDevice) );

    current_utc_time(&timer_start);
    decomposeCholeskyGPU(a_dev, n);
    current_utc_time(&timer_end);
    printf("Cholesky: %fms\n", (double) (timer_end.tv_nsec - timer_start.tv_nsec) / 1000);
    //printMatrix(ainv, m, n);


    // inverse new cholesky
    //gpuErrchk( cudaMemcpy(a_dev, ainv, matrixSize, cudaMemcpyHostToDevice) );    

    current_utc_time(&timer_start);
    inverseLowerGPU(a_dev, n);
    current_utc_time(&timer_end);
    printf("Substitution: %fms\n", (double) (timer_end.tv_nsec - timer_start.tv_nsec) / 1000);
    
    current_utc_time(&timer_start);
    multiplyLowerGPU(a_dev, n);
    current_utc_time(&timer_end);
    printf("Multiplication: %fms\n", (double) (timer_end.tv_nsec - timer_start.tv_nsec) / 1000);

    gpuErrchk( cudaMemcpy(ainv, a_dev, matrixSize, cudaMemcpyDeviceToHost) );
    //printMatrix(ainv, m, n);

    //gpuErrchk( cudaPeekAtLastError() );
    //gpuErrchk( cudaDeviceSynchronize() );

    */

    double error = 0;
    
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            error += ainv[i * n + j] - atest[i * n + j];
        }
    }   

    printf("Error: %f\n", error);
    //printf("%fms\n", (double) (timer_end.tv_nsec - timer_start.tv_nsec) / 1000);

    return 0;
}


