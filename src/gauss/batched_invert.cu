#include <stdio.h>	
#include <errno.h>
#include <stdlib.h>
#include <cuda.h>
#include "cublas_v2.h"
#include "../include/types.h"
#include "../include/helper.h"

#define SWAP(x, y, z)	((z) = (x),(x) = (y),(y) = (z))

void printMatrix(Array a, int n, int batchSize) {
	int i, j, k;

	for(k = 0; k < batchSize; k++) {
		printf("=============== <%d> ===============\n", k + 1);
		for(i = 0; i < n; i++) {
			for(j = 0; j < n; j++)
				printf("%f\t", a[k * n * n + j * n + i]);
			printf("\n");
		}
	}
	printf("\n");
}

void pivotRow(cublasHandle_t &handle, int n, Array *a, Array *a_inv, int col, int batchSize) {
	cudaStream_t *streams = (cudaStream_t *) malloc(sizeof(cudaStream_t) * batchSize);
	for(int i = 0; i < batchSize; i++)
		cudaStreamCreate(&streams[i]);

	int *pivot = (int *) malloc(sizeof(int) * batchSize);
	for(int i = 0; i < batchSize; i++) {
		cublasSetStream(handle, streams[i]);
		cublasIsamax(handle,
				n - col,			// Number of elements to be searched
				a[i] + (col * n) + col,		// Starting position
				1,				// Increment in words (NOT BYTES)
				&pivot[i]);			// Maximum element in the col
	}
	cudaDeviceSynchronize();

	for(int i = 0; i < batchSize; i++) {
		int row = pivot[i] - 1 + col;		// Row number with maximum element (starts with 1)
		if(row == col)
			return;
		cublasSetStream(handle, streams[i]);
		cublasSswap(handle,
				n,				// Nuber of elements to be swapped
				a[i] + col,			// Current row
				n,				// Increment (becuase of column major)
				a[i] + row,			// Row with max pivot
				n);
		cublasSswap(handle, n, a_inv[i] + col, n, a_inv[i] + row, n);
	}
	cudaDeviceSynchronize();

	for(int i = 0; i < batchSize; i++)
		cudaStreamDestroy(streams[i]);
	free(pivot);
	free(streams);
}

void normalizeRow(cublasHandle_t &handle, int n, Array *a, Array *a_inv, int row, int batchSize) {
	cudaStream_t *streams = (cudaStream_t *) malloc(sizeof(cudaStream_t) * batchSize);
	for(int i = 0; i < batchSize; i++)
		cudaStreamCreate(&streams[i]);
	DataType *scalar = (DataType *) malloc(sizeof(DataType) * batchSize);

	for(int i = 0; i < batchSize; i++) {
		cublasSetStream(handle, streams[i]);
		cudaMemcpy(&scalar[i], &a[i][row * n + row], sizeof(DataType), cudaMemcpyDeviceToHost);
	}
	for(int i = 0; i < batchSize; i++) {
		scalar[i] = 1 / scalar[i];
		cublasSetStream(handle, streams[i]);
		cublasSscal(handle, n, &scalar[i], a[i] + row, n);
		cublasSscal(handle, n, &scalar[i], a_inv[i] + row, n);
	}

	for(int i = 0; i < batchSize; i++)
		cudaStreamDestroy(streams[i]);
	free(scalar);
	free(streams);
}

__global__
void transform_matrix(Array *a, Array *a_inv, int row, int n, int batchSize) {
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
		pivotRow(handle, n, a, a_inv, i, batchSize);

		// Make column entry to be one
		normalizeRow(handle, n, a, a_inv, i, batchSize);

		// number of threads equals number of rows
		transform_matrix<<<batchSize, n, 3 * n>>>(a, a_inv, i, n, batchSize);
	}
}

static cudaError_t batchedCudaMalloc(Array* devArrayPtr, size_t *pitch, size_t arraySize, int batchSize) {
	char *devPtr;

	cudaError_t result = cudaMallocPitch((void**)&devPtr, pitch, arraySize, batchSize);

	if (cudaSuccess != result) {
		return result;
	}

	for (int i = 0; i < batchSize; ++i) {
		devArrayPtr[i] = (Array)devPtr;
		devPtr += *pitch;
	}

	return cudaSuccess;
}
static void batchedInverse(
		cublasHandle_t handle,
		int n,
		Array As,
		Array A_invs,
		int batchSize) {

	Array *devBs;
	size_t pitchBs;
	Array *devCs;
	size_t pitchCs;

	const size_t sizeOfMatrixB = sizeof(DataType) * n * n;

	gpuErrchk( cudaHostAlloc((void**)&devBs, sizeof(Array)*batchSize, cudaHostAllocDefault) );
	gpuErrchk( cudaHostAlloc((void**)&devCs, sizeof(Array)*batchSize, cudaHostAllocDefault) );

	gpuErrchk( batchedCudaMalloc(devBs, &pitchBs, sizeOfMatrixB, batchSize) );
	gpuErrchk( batchedCudaMalloc(devCs, &pitchCs, sizeOfMatrixB, batchSize) );

	gpuErrchk( cudaMemcpy2D(devBs[0], pitchBs, As, sizeOfMatrixB, sizeOfMatrixB, batchSize,
				cudaMemcpyHostToDevice) );
	gpuErrchk( cudaMemcpy2D(devCs[0], pitchCs, A_invs, sizeOfMatrixB, sizeOfMatrixB, batchSize,
				cudaMemcpyHostToDevice) );


	// Calculate Minv = Madd^-1, store result in Bs
	invert(handle, n, devBs, devCs, batchSize);
	// devAs: As
	// devBs: Minv
	// devCs: Madd

	gpuErrchk( cudaMemcpy2D(A_invs, sizeOfMatrixB, devCs[0], pitchCs, sizeOfMatrixB, batchSize,
				cudaMemcpyDeviceToHost) );
	gpuErrchk( cudaFree((void*)devBs[0]) );
	gpuErrchk( cudaFree((void*)devCs[0]) );
	gpuErrchk( cudaFreeHost((void*)devBs) );
	gpuErrchk( cudaFreeHost((void*)devCs) );
}

int main(int argc, char *argv[]) {
	cublasHandle_t handle;
	int numMatrices, n;
	Array a, a_inv;

	cublasErrchk( cublasCreate(&handle) );

	readMatricesFile(argv[1], &numMatrices, &n, &n, &a);
	a_inv = (Array) malloc(sizeof(DataType) * numMatrices * n * n);
	printMatrix(a, n, numMatrices);
	for(int i = 0; i < numMatrices; i++)
		for(int j = 0; j < n; j++)
			for(int k = 0; k < n; k++)
				if(j == k)
					a_inv[i * n * n + j * n + k] = 1;
				else
					a_inv[i * n * n + j * n + k] = 0;
	batchedInverse(handle, n, a, a_inv, numMatrices);
	printMatrix(a_inv, n, numMatrices);

	gpuErrchk( cudaPeekAtLastError() );
	gpuErrchk( cudaDeviceSynchronize() );

	return 0;
}
