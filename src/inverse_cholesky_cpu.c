#include <stdio.h>
#include <math.h>

#define N 4

void printMatrix(float a[N][N]) {
	int i, j;

	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++)
			printf("%f\t", a[i][j]);
		printf("\n");
	}
	printf("\n");
}

void choleskyDecomposition(float a[N][N], float l[N][N]) {

	for(int i = 0; i< N; i++){
		for(int j = 0; j < (i+1); j++){
			float sum = 0;

			for(int k = 0; k < j; k++){
				sum += l[i][k] * l[j][k];
			}

			l[i][j] = (i == j) ? sqrt(a[i][i] - sum) : (1.0 / l[j][j] * (a[i][j] - sum));

			//printf("= %f", l[i][j]);

			//printf("\n");
		}
	}

}

void inverseLower(float l[N][N], float l_inv[N][N]) {

	for (int j = 0; j < N; j++) {
		l_inv[j][j] = 1.0 / l[j][j];


		for (int i = j + 1; i < N; i++) {
			//printf("[%d][%d]: ",i,j);
			float sum = 0;

			for (int k = j; k < i; k++) {
				//printf("- [%d][%d]*[%d][%d] ",i,k,k,j);
				sum -= l[i][k] * l_inv[k][j];
			}
			
			l_inv[i][j] = sum / l[i][i];

			//printf("/ [%d][%d]",i,i);
			//printf("\n");
		}
	}
}

void inverse(float a[N][N], float a_inv[N][N]) {
	float l[N][N], l_inv[N][N];

	choleskyDecomposition(a, l);
	inverseLower(l, l_inv);

	printMatrix(l_inv);  
	
	for (int j = 0; j < N; j++) {
		for (int i = j; i < N; i++) {
			float sum = 0;
			printf("[%d][%d]: ",i,j);

			for (int k = i; k < N; k++) {
				printf("+[%d][%d] * [%d][%d] ",k,i,k,j);
				sum += l_inv[k][i]*l_inv[k][j];
			}

            a_inv[i][j] = sum;
            a_inv[j][i] = sum;
            printf("\n");
         }

    }
    
}


int main(int argc, char *argv[]) {
	float a[N][N], a_inv[N][N];

	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			scanf("%f", &a[i][j]);
			a_inv[i][j] = 0;
		}
	}

	inverse(a, a_inv);
	printf("Inverse is:\n"); 
	printMatrix(a_inv); 

	return 0;
}
