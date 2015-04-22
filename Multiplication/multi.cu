#include <stdio.h>
#include <cuda_runtime.h>
// #include <helper_cuda.h>
// #include <helper_functions.h>


#define cutoff_thres			12
static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}

void HostMemcpy(float *des, float *src, float s){
	if(des != NULL){
		free(des);
		des=NULL;
	}
	des=(float*)malloc(s*sizeof(float));
	memcpy(des,src,s*sizeof(float));
}

#define HANDLE_ERROR( err ) 				(HandleError( err, __FILE__, __LINE__ ))
#define THREADS_PER_BLOCK					256
#define hostMemcpy(des, src, s)				(HostMemcpy(des,src,s))
#define BLOCK_THREAD(m,n)					((m)*(n)+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK,(THREADS_PER_BLOCK<(m)*(n)?THREADS_PER_BLOCK:(m)*(n))


void multiply(float *a, float *b, float *c, int m, int k, int n);
bool cutoff(int m, int k, int n);
void strassen_multiply(float *a, float *b, float *c, int m, int k, int n);
void memCopy2D(float *src,int sc, int soffsetx, int soffsety, float *dest,int dc,  int doffsetx, int doffsety, int width, int height);
void add(float *a, float *b, float *c, int m, int n);
__global__ void kernel_add(float *a, float *b, float *c, int m, int n);
void subtract(float *a, float *b, float *c, int m, int n);
__global__ void kernel_subtract(float *a, float *b, float *c, int m, int n);

void standard_multiply(float *a, float *b, float *c, int m, int k, int n);
__global__ void kernel_multiply(float *a, float *b,float *c, int m, int k, int n);

void test(){
	float a[304] = {7,68,11,11,10,31,18,16,51,34,27,9,14,79,57,19,
12,41,94,29,56,74,39,45,70,96,86,28,80,35,38,61,
16,72,48,73,85,94,46,56,83,46,95,19,42,74,56,32,
1,80,58,97,5,77,5,48,62,20,90,80,63,38,85,37,
44,57,84,92,35,3,95,58,5,89,87,2,33,34,49,77,
97,21,28,82,25,4,33,42,60,68,39,5,58,83,47,67,
3,27,54,75,43,96,61,42,45,59,5,72,29,82,27,9,
20,27,20,28,37,7,26,59,1,46,83,74,98,52,31,91,
76,59,68,93,93,77,46,85,87,57,79,58,45,67,72,9,
100,70,97,5,55,74,93,13,94,50,85,94,16,78,23,11,
15,42,67,70,1,33,92,14,56,57,19,50,59,16,57,6,
21,23,83,58,69,57,17,40,35,39,25,64,10,99,24,92,
33,64,13,79,13,43,12,19,80,5,82,60,38,12,100,2,
95,59,78,69,88,56,78,93,22,79,69,66,78,55,25,34,
81,13,99,69,79,15,95,68,80,96,21,9,32,62,53,50,
45,26,96,12,94,11,1,62,35,28,38,99,75,70,14,3,
94,33,91,41,29,79,34,29,93,39,27,69,89,65,76,34,
67,10,96,30,20,10,6,77,57,100,27,94,35,20,61,67,
59,8,30,47,29,11,88,69,19,54,15,84,58,21,8,9};
	float b[272] = {25,78,23,8,51,29,39,3,13,90,43,71,85,42,47,88,34,
96,32,59,28,22,27,75,56,36,72,51,21,51,5,59,22,23,
58,29,72,14,58,55,60,11,53,60,35,6,59,63,84,66,100,
37,54,96,74,67,24,92,78,37,97,14,66,32,19,75,57,88,
94,69,16,14,12,75,72,80,61,31,53,7,8,54,17,44,75,
35,20,47,18,41,71,36,2,23,19,53,16,12,60,88,55,35,
28,97,23,27,14,21,96,77,56,99,33,83,21,48,68,46,58,
44,99,56,17,92,89,18,32,91,40,69,75,39,89,98,60,83,
56,1,89,79,83,16,28,91,91,99,71,94,45,94,19,23,87,
98,25,34,30,87,96,51,23,41,49,79,41,49,87,87,39,1,
1,26,20,31,77,19,76,37,31,23,26,21,16,27,95,75,40,
33,67,7,38,74,30,70,78,15,98,60,4,68,21,84,79,10,
30,51,78,81,35,46,37,7,21,78,100,46,69,49,68,11,65,
27,98,60,4,64,18,84,40,9,24,34,99,45,30,37,93,93,
49,74,84,38,54,91,71,55,59,91,22,94,70,96,19,60,62,
29,68,42,3,60,27,3,19,55,83,16,39,44,28,40,95,72};
	
	float *c;
	c = (float*)malloc(19*17*sizeof(float));
	multiply(a,b,c,19,16,17);

	for (int i = 0; i < 19; i ++){
		printf("\n");
		for (int j = 0; j < 17; j++){
			printf("%f ",c[i*17+j]);
		}
	}
}

void test2(){
		float a[9] = {10,58,36,
53,6,24,
45,99,36};
	float b[9] = {15,31,81,
97,91,24,
81,48,40};
	
	float *c;
	c = (float*)malloc(3*3*sizeof(float));
	multiply(a,b,c,3,3,3);

	for (int i = 0; i < 3; i ++){
		printf("\n");
		for (int j = 0; j < 3; j++){
			printf("%f ",c[i*3+j]);
		}
	}
}

int main(){
	printf("\nMultiplication");
	test();
}

void multiply(float *a, float *b, float *c, int m, int k, int n){
	printf("\nStart multiplication");
	
	if (cutoff(m,k,n)){
		printf("\nCut off satisfied");
		standard_multiply(a,b,c,m,k,n);
	} else {
		if (m % 2 == 0 && k % 2 == 0 && n % 2 == 0){
			printf("\nStart Strassen");
			strassen_multiply(a,b,c,m,k,n);
		} else {
			printf("\nStart Dynamic Peeling");
			int x,y,z;
			float *a11, *a12, *a21, *a22, *b11, *b12, *b21, *b22, *c11, *c12, *c21, *c22, *r1;
			x = m % 2;
			y = k % 2;
			z = n % 2;
			
			a11 = (float*)malloc((m-x)*(k-y)*sizeof(float));
			a12 = (float*)malloc((m-x)*sizeof(float));
			a21 = (float*)malloc((k-y)*sizeof(float));
			a22 = (float*)malloc(sizeof(float));
			b11 = (float*)malloc((k-y)*(n-z)*sizeof(float));
			b12 = (float*)malloc((k-y)*sizeof(float));
			b21 = (float*)malloc((n-z)*sizeof(float));
			b22 = (float*)malloc(sizeof(float));
			c11 = (float*)malloc((m-x)*(n-z)*sizeof(float));
			c12 = (float*)malloc((m-x)*sizeof(float));	
			c21 = (float*)malloc((n-z)*sizeof(float));
			c22 = (float*)malloc(x*z*sizeof(float));
			memCopy2D(a,k,0,0,a11,k-y,0,0,k-y,m-x);
			memCopy2D(b,n,0,0,b11,n-z,0,0,n-z,k-y);
			multiply(a11,b11,c11,m-x,k-y,n-z);

			if (y == 1){
				

				memCopy2D(a,k,k-y,0,a12,y,0,0,y,m-x);
				memCopy2D(b,n,0,k-y,b21,n-z,0,0,n-z,y);
				
				r1 = (float*)malloc((m-x)*(n-z)*sizeof(float));
				multiply(a12,b21,r1,m-x,y,n-z);

				add(c11,r1,c11,m-x,n-z);

				free(r1);

				
			}
			memCopy2D(c11,n-z,0,0,c,n,0,0,n-z,m-x);

			if (z == 1){
				memCopy2D(b,n,n-z,0,b12,z,0,0,z,k-y);
				multiply(a11,b12,c12,(m-x),(k-y),1);
				if (y == 1 && z == 1){
					b22[0] = b[k*n-1];
					r1 = (float*)malloc((m-x)*z*sizeof(float));
					multiply(a12,b22,r1,m-x,y,z);
					
					add(c12,r1,c12,m-x,z);
					free(r1);

				}
				memCopy2D(c12,z,0,0,c,n,n-z,0,z,m-x);	
			}

			if (x == 1){

				memCopy2D(a,k,0,m-x,a21,k-y,0,0,k-y,x);
				multiply(a21,b11,c21,x,k-y,n-z);
				if (y == 1){
					memCopy2D(b,n,0,k-y,b21,n-z,0,0,n-z,1);
					a22[0] = a[m*k-1];
					r1 = (float*)malloc((n-z)*sizeof(float));
					multiply(a22,b21,r1,1,1,n-z);
					add(c21,r1,c21,x,n-z);
					free(r1);
				}
				memCopy2D(c21,n-z,0,0,c,n,0,m-x,n-z,1);

				if (z == 1){
					memCopy2D(b,n,n-z,0,b12,z,0,0,z,k-y);
					multiply(a21,b12,c22,x,k-y,z);
					if (y == 1){
						r1 = (float*)malloc(x*z*sizeof(float));
						multiply(a22,b22,r1,x,y,z);
						add(c22,r1,c22,x,z);
						free(r1);
					}
					memCopy2D(c22,z,0,0,c,n,n-z,m-x,1,1);			
				}
			}


			free(c11);
			free(c12);
			free(c21);
			free(c22);
		}
	}
}

bool cutoff(int m, int k, int n){
	if ((m < cutoff_thres && k < cutoff_thres && n < cutoff_thres) || (m*k*n<cutoff_thres*(n*k+m*n+m*k)/3)){
		return true;
	}
	return false;
}

void strassen_multiply(float *a, float *b, float *c, int m, int k, int n){
	printf("\nStrassen multiplication");
	float *a11, *a12, *a21, *a22, *b11, *b12, *b21, *b22, *c11, *c12, *c21, *c22, *r1, *r2, *r3, *r4, *r5;
	a11 = (float*)malloc((m*k)/4*sizeof(float));
	a12 = (float*)malloc((m*k)/4*sizeof(float));
	a21 = (float*)malloc((m*k)/4*sizeof(float));
	a22 = (float*)malloc((m*k)/4*sizeof(float));
	b11 = (float*)malloc((n*k)/4*sizeof(float));
	b12 = (float*)malloc((n*k)/4*sizeof(float));
	b21 = (float*)malloc((n*k)/4*sizeof(float));
	b22 = (float*)malloc((n*k)/4*sizeof(float));
	c11 = (float*)malloc((m*n)/4*sizeof(float));
	c21 = (float*)malloc((m*n)/4*sizeof(float));
	c12 = (float*)malloc((m*n)/4*sizeof(float));
	c22 = (float*)malloc((m*n)/4*sizeof(float));

	r1 = (float*)malloc((m*k)/4*sizeof(float));
	r2 = (float*)malloc((k*n)/4*sizeof(float));
	r3 = (float*)malloc((m*n)/4*sizeof(float));
	r4 = (float*)malloc((m*n)/4*sizeof(float));
	r5 = (float*)malloc((m*n)/4*sizeof(float));

	memCopy2D(a,k,0,0,a11,k/2,0,0,k/2,m/2);
	memCopy2D(a,k,k/2,0,a12,k/2,0,0,k/2,m/2);
	memCopy2D(a,k,0,m/2,a21,k/2,0,0,k/2,m/2);
	memCopy2D(a,k,k/2,m/2,a22,k/2,0,0,k/2,m/2);

	memCopy2D(b,n,0,0,b11,n/2,0,0,n/2,k/2);
	memCopy2D(b,n,n/2,0,b12,n/2,0,0,n/2,k/2);
	memCopy2D(b,n,0,k/2,b21,n/2,0,0,n/2,k/2);
	memCopy2D(b,n,n/2,k/2,b22,n/2,0,0,n/2,k/2);


	////////////////////////////////////////////////////////////////////////////////
	add(a21,a22,r1,m/2,k/2);
	subtract(b12,b11,r2,k/2,n/2);
	multiply(r1,r2,r3,m/2,k/2,n/2);
	// add(c12,r3,c12,m/2,n/2);
	hostMemcpy(c12,r3,m*n/4);
	// add(c22,r3,c22,m/2,n/2);
	hostMemcpy(c22,r3,m*n/4);
	subtract(r1,a11,r1,m/2,k/2);
	subtract(b22,r2,r2,k/2,n/2);
	multiply(a11,b11,r3,m/2,k/2,n/2);
	// add(c11,r3,c11,m/2,n/2);
	hostMemcpy(c11,r3,m*n/4);
	multiply(r1,r2,r4,m/2,k/2,n/2);
	add(r3,r4,r3,m/2,n/2);
	multiply(a12,b21,r5,m/2,k/2,n/2);
	add(c11,r5,c11,m/2,n/2);
	subtract(a12,r1,r1,m/2,k/2);
	subtract(b21,r2,r2,k/2,n/2);
	multiply(r1,b22,r5,m/2,k/2,n/2);
	add(c12,r5,c12,m/2,n/2);
	add(c12,r3,c12,m/2,n/2);
	multiply(a22,r2,r5,m/2,k/2,n/2);
	// add(c21,r5,c21,m/2,n/2);
	hostMemcpy(c21,r5,m*n/4);
	subtract(a11,a21,r1,m/2,k/2);
	subtract(b22,b12,r2,k/2,n/2);
	multiply(r1,r2,r4,m/2,k/2,n/2);
	add(r3,r4,r3,m/2,n/2);
	add(c21,r3,c21,m/2,n/2);
	add(c22,r3,c22,m/2,n/2);

	//Gather C11, C12, C21 and C22 to C 
	memCopy2D(c11,n/2,0,0,c,n,0,0,n/2,m/2);
	memCopy2D(c12,n/2,0,0,c,n,n/2,0,n/2,m/2);
	memCopy2D(c21,n/2,0,0,c,n,0,m/2,n/2,m/2);
	memCopy2D(c22,n/2,0,0,c,n,n/2,m/2,n/2,m/2);

	//Free mem
	free(a11);
	free(a12);
	free(a21);
	free(a22);
	free(b11);
	free(b12);
	free(b21);
	free(b22);
	free(c11);
	free(c12);
	free(c21);
	free(c22);
	free(r1);
	free(r2);
	free(r3);
	free(r4);
	free(r5);

}


void memCopy2D(float *src,int sc, int soffsetx, int soffsety, float *dest,int dc,  int doffsetx, int doffsety, int width, int height){
	printf("\nStart copying 2D mem");
	for (int i = 0; i < width; i ++){
		for (int j = 0; j < height; j++){
			dest[(doffsety+j)*dc + doffsetx + i] = src[(soffsety+j)*sc + soffsetx + i];
		}
	}
}

void add(float *a, float *b, float *c, int m, int n){
	float *dev_a, *dev_b,*dev_c;
	HANDLE_ERROR(cudaMalloc(&dev_a,m*n*sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&dev_b,m*n*sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&dev_c,m*n*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(dev_a,a,m*n*sizeof(float),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b,b,m*n*sizeof(float),cudaMemcpyHostToDevice));
	kernel_add<<<BLOCK_THREAD(m,n)>>>(dev_a,dev_b,dev_c,m,n);
	HANDLE_ERROR(cudaMemcpy(c,dev_c,m*n*sizeof(float),cudaMemcpyDeviceToHost));
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}

__global__ void kernel_add(float *a, float *b, float *c, int m, int n){
	// cudaMalloc(&c, m*n*sizeof(float));
	int idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (idx < m*n){
		int x = idx / n;
		int y = idx % n;
		c[x*n+y] = a[x*n+y] + b[x*n+y];	
	}
	
}

void subtract(float *a, float *b, float *c, int m, int n){
	float *dev_a, *dev_b,*dev_c;
	HANDLE_ERROR(cudaMalloc(&dev_a,m*n*sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&dev_b,m*n*sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&dev_c,m*n*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(dev_a,a,m*n*sizeof(float),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b,b,m*n*sizeof(float),cudaMemcpyHostToDevice));
	kernel_subtract<<<BLOCK_THREAD(m,n)>>>(dev_a,dev_b,dev_c,m,n);
	HANDLE_ERROR(cudaMemcpy(c,dev_c,m*n*sizeof(float),cudaMemcpyDeviceToHost));
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}
__global__ void kernel_subtract(float *a, float *b, float *c, int m, int n){
	// cudaMalloc(&c, m*n*sizeof(float));
	int idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (idx < m*n){
		int x = idx / n;
		int y = idx % n;
		c[x*n+y] = a[x*n+y] - b[x*n+y];	
	}
}



void standard_multiply(float *a, float *b, float *c, int m, int k, int n){
	printf("\nM: %d, K: %d, N: %d",m,k,n);
	float *dev_a, *dev_b, *dev_c;
 	HANDLE_ERROR(cudaMalloc(&dev_a,m*k*sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&dev_b,k*n*sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&dev_c,m*n*sizeof(float)));
	HANDLE_ERROR(cudaMemcpy(dev_a,a,m*k*sizeof(float),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(dev_b,b,k*n*sizeof(float),cudaMemcpyHostToDevice));
	kernel_multiply<<<BLOCK_THREAD(m,n)>>>(dev_a, dev_b, dev_c, m, k, n);

	HANDLE_ERROR(cudaMemcpy(c,dev_c,m*n*sizeof(float),cudaMemcpyDeviceToHost));
	
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
}

__global__ void kernel_multiply(float *a, float *b,float *c, int m, int k, int n){
	int idx = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if (idx < m*n){
		int x = idx / n;
		int y = idx % n;
		c[x*n+y] = 0;
		for (int i = 0; i < k; i++){
			c[x*n+y] += a[x*k+i]*b[i*n+y];
		}
	}
}