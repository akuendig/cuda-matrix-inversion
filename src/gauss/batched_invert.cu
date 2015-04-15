#include <stdio.h>
#include <errno.h>
#include <stdlib.h>

#include <cuda.h>
#include "cublas_v2.h"

#include "../../include/types.h"
#include "../../include/helper_cpu.h"
#include "../../include/helper_gpu.h"
#include "../../include/inverse_cpu.h"
#include "../../include/inverse_gpu.h"

#define SWAP(x, y, z)	((z) = (x),(x) = (y),(y) = (z))

__global__
void pivotRow(Array *a, Array *a_inv, int n, int col) {
	if (a[blockIdx.x][col * n + col] != 0)
		return;
	int i;
	__shared__ int row;
	if(threadIdx.x == 0) {
		for (i = 1; i < (n - col); ++i) {
			if (a[blockIdx.x][(col * n) + col + i] != 0)
				break;
		}
	
		if (i == (n - col)) {
			//Handle Error: Matrix is not invertible
			// Do something, maybe quit the code
		} else {
			row = i + col;
		}
	}
	__syncthreads();
	
	float temp1 = a[blockIdx.x][threadIdx.x * n + col];
        a[blockIdx.x][threadIdx.x * n + col] = a[blockIdx.x][threadIdx.x * n + row];
        a[blockIdx.x][threadIdx.x * n + row] = temp1;

        temp1 = a_inv[blockIdx.x][threadIdx.x * n + col];
        a_inv[blockIdx.x][threadIdx.x * n + col] = a_inv[blockIdx.x][threadIdx.x * n + row];
        a_inv[blockIdx.x][threadIdx.x * n + row] = temp1;
}

__global__
void normalizeRow(Array *a, Array *a_inv, int n, int row) {
	__shared__ DataType scalar;

	if(threadIdx.x == 0)
		scalar = 1 / a[blockIdx.x][row * n + row];
	__syncthreads();

	a[blockIdx.x][threadIdx.x * n + row] *= scalar;
	a_inv[blockIdx.x][threadIdx.x * n + row] *= scalar;
}

__global__
void transform_matrix(Array *a, Array *a_inv, int n, int row) {
	extern __shared__ DataType shared[];

	DataType *scalars = &shared[0];
	DataType *currRowA = &shared[n];
	DataType *currRowI = &shared[2 * n];

	// store the scalars corresponding to the column 'row'
	scalars[threadIdx.x] = a[blockIdx.x][row * n + threadIdx.x];
	currRowA[threadIdx.x] = a[blockIdx.x][threadIdx.x * n + row];
	currRowI[threadIdx.x] = a_inv[blockIdx.x][threadIdx.x * n + row];
	__syncthreads();

	// no need to transform 'row'th row
	if(threadIdx.x == row)
		return;

	// Each thread transforms row
	for(int i = 0; i < n; i++) {
		a[blockIdx.x][i * n + threadIdx.x] -= (scalars[threadIdx.x] * currRowA[i]);
		a_inv[blockIdx.x][i * n + threadIdx.x] -= (scalars[threadIdx.x] * currRowI[i]);
	}
}

void invert(cublasHandle_t &handle, int n, Array *a, Array *a_inv, int batchSize) {
	for(int i = 0; i < n; i++) {
		// Pivot the matrix
		pivotRow<<<batchSize, n>>>(a, a_inv, n, i);

		// Make column entry to be one
		normalizeRow<<<batchSize, n>>>(a, a_inv, n, i);

		// number of threads equals number of rows
		transform_matrix<<<batchSize, n, 3*n*sizeof(DataType)>>>(a, a_inv, n, i);
	}
}

void inverse_gauss_batched_device(cublasHandle_t handle, int n, Array devAs, Array devAInvs, int batchSize);

extern "C" void inverse_gauss_batched_gpu(
		cublasHandle_t handle,
		int n,
		Array As,
		Array aInvs,
		int batchSize) {

	int k, i;
	Array *devAs;
	size_t pitchAs;
	Array *devAInvs;
	size_t pitchAInvs;

	const size_t ArraySize = sizeof(DataType) * n * n;

	gpuErrchk( cudaHostAlloc((void**)&devAs, sizeof(Array)*batchSize, cudaHostAllocDefault) );
	gpuErrchk( cudaHostAlloc((void**)&devAInvs, sizeof(Array)*batchSize, cudaHostAllocDefault) );

	gpuErrchk( batchedCudaMalloc(devAs, &pitchAs, ArraySize, batchSize) );
	gpuErrchk( batchedCudaMalloc(devAInvs, &pitchAInvs, ArraySize, batchSize) );

    memset(aInvs, 0, batchSize*ArraySize);

	for (k = 0; k < batchSize; ++k) {
	    for (i = 0; i < n; ++i) {
	    	aInvs[k*n*n + i*n + i] = 1.f;
    	}
	}

	gpuErrchk( cudaMemcpy2D(devAs[0], pitchAs, As, ArraySize, ArraySize, batchSize,
				cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy2D(devAInvs[0], pitchAInvs, aInvs, ArraySize, ArraySize, batchSize,
				cudaMemcpyHostToDevice) );

	// Calculate Minv = Madd^-1, store result in Bs
	invert(handle, n, devAs, devAInvs, batchSize);
	// devAs: As
	// devAs: Minv
	// devAInvs: Madd

	gpuErrchk( cudaMemcpy2D(aInvs, ArraySize, devAInvs[0], pitchAInvs, ArraySize, batchSize,
				cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaFree((void*)devAs[0]) );
	gpuErrchk( cudaFree((void*)devAInvs[0]) );
	gpuErrchk( cudaFreeHost((void*)devAs) );
	gpuErrchk( cudaFreeHost((void*)devAInvs) );
}

// int main(int argc, char *argv[]) {
// 	cublasHandle_t handle;
// 	int numMatrices, n;
// 	Array a, a_inv;

// 	cublasErrchk( cublasCreate(&handle) );

// 	readMatricesFile(argv[1], &numMatrices, &n, &n, &a);
// 	a_inv = (Array) malloc(sizeof(DataType) * numMatrices * n * n);
// 	printMatrixList(a, n, numMatrices);
// 	for(int i = 0; i < numMatrices; i++)
// 		for(int j = 0; j < n; j++)
// 			for(int k = 0; k < n; k++)
// 				if(j == k)
// 					a_inv[i * n * n + j * n + k] = 1;
// 				else
// 					a_inv[i * n * n + j * n + k] = 0;
// 	batchedInverse(handle, n, a, a_inv, numMatrices);
// 	printMatrixList(a_inv, n, numMatrices);

// 	gpuErrchk( cudaPeekAtLastError() );
// 	gpuErrchk( cudaDeviceSynchronize() );

// 	return 0;
// }
