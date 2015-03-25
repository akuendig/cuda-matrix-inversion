#include <stdio.h>	
#include <cuda.h>
#include <cublas.h>

#define SWAP(x, y, z)	((z) = (x),(x) = (y),(y) = (z))

#define N 5
#define	DataType	float
#define ArraySize	(N * N * sizeof(DataType))

void printMatrix(float a[N][N]) {
	int i, j;

	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++)
			printf("%f\t", a[i][j]);
		printf("\n");
	}
	printf("\n");
}

void pivotRow(cublasHandle_t &handle, DataType *a, DataType *a_inv, int col) {
	int pivot = -1;

	cublasIsamax(handle,
		N - col,					// Number of elements to be searched
		a + (col * N) + col,		// Starting position
		1,							// Increment in words (NOT BYTES)
		&pivot);					// Maximum element in the col
	int row = pivot - (col * N);	// Row number with maximum element

	cublasSswap(handle,
		N,							// Nuber of elements to be swapped
		a + col,					// Current row
		N,							// Increment (becuase of column major)
		a + row,					// Row with max pivot
		N);
	cublasSswap(handle, N, a_inv + col, N, a_inv + row, N);
}

void normalizeRow(cublasHandle_t &handle, DataType *a, DataType *a_inv, int row) {
	DataType scalar = a[row * N + row];

	cublasSscal(handle, N, &scalar, a + row, N);
	cublasSscal(handle, N, &scalar, a_inv + row, N);
}

__global__
void transform_matrix(DataType *a DataType *a_inv, int row) {
	__shared__ DataType scalars[N];

	// store the scalars corresponding to the column 'row'
	scalars[threadIdx.x] = a[row * N + threadIdx.x];
	__syncthreads();

	// No need to transform 'row'th row
	if(threadIdx.x == row)
		return;

	// Each thread transforms column
	DataType pivot_elem = a[threadIdx.x * N + row];
	for(int i = 0; i < N; i++) {
		a[threadIdx.x * N + i] -= (scalars[threadIdx.x] * pivot_elem);
	}
}

void invert(cublasHandle_t &handle, DataType *a, DataType *a_inv) {
	for(int i = 0; i < N; i++) {
		// Pivot the matrix
		pivotRow(handle, a, a_inv, i);

		// Make column entry to be one
		normalizeRow(a, a_inv, i);

		// Number of threads equals number of rows
		transform_matrix<<<1, N>>>(a, a_inv, i);
	}
}

int main(int argc, char *argv[]) {
	DataType *a, *a_inv;
	DataType *dev_a, **dev_a_inv;
	cublasHandle_t handle;

	/* Pre-processing steps */
	if(!(a = (DataType *)malloc(ArraySize))) {
		perror("");
		return errno;
	}
	if(!(a_inv = (DataType *)malloc(ArraySize))) {
		perror("");
		return errno;
	}
	cudaMalloc(&dev_a, ArraySize);
	cudaMalloc(&dev_a_inv, ArraySize);
	cublasCreate(&handle);

	/* Input column major matrix */
	for(int i = 0; i < N; i++)
		for(int j = 0; j < N; j++) {
			scanf("%f", &a[i * N + j])
			if(i == j)
				a_inv[i * N + j] = 1;
			else
				a_inv[i * N + j] = 0;
		}
	cudaMemcpy(dev_a, a, ArraySize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_a_inv, a_inv, ArraySize, cudaMemcpyHostToDevice);

	/* Invert the matrix */
	invert(handle, dev_a, dev_a_inv);

	/* Display the result */
	cudaMemcpy(a_inv, dev_a_inv, ArraySize, cudaMemcpyDeviceToHost);
	printf("Inverse is:\n");
	printMatrix(a_inv);

	/* Cleanup the mess */
	free(a);
	free(a_inv);
	cudaFree(dev_a);
	cudaFree(dev_a_inv);
	cublasDestroy(handle);

	return 0;
}
