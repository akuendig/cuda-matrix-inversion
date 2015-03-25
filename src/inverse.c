#include <stdio.h>	

#define N 5
#define SWAP(x, y, z)	((z) = (x),(x) = (y),(y) = (z))

void printMatrix(float a[N][N]) {
	int i, j;

	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++)
			printf("%f\t", a[i][j]);
		printf("\n");
	}
	printf("\n");
}

void pivotRow(float a[N][N], float a_inv[N][N], int col) {
	int row, temp, j;

	for(row = col; row < N; row++)
		if(a[row][col] != 0)
			break;

	if(row == col)
		return;
	for(j = 0; j < N; j++) {
		SWAP(a[row][j], a[col][j], temp);
		SWAP(a_inv[row][j], a_inv[col][j], temp);
	}
}

void normalizeRow(float a[N][N], float a_inv[N][N], int row) {
	int j;
	float scalar = a[row][row];

	for(j = 0; j < N; j++) {
		a[row][j] /= scalar;
		a_inv[row][j] /= scalar;
	}
}

void verifyInverse(float a[N][N], float a_inv[N][N]) {
	float c[N][N];
	int i, j, k;
	float sum;

	for(i = 0; i < N; i++)
		for(j = 0; j < N; j++) {
			sum = 0;
			for(k = 0; k < N; k++)
				sum += (a[i][k] * a_inv[k][j]);
			c[i][j] = sum;
		}

	printf("Verification\n");
	printMatrix(c);
}

int main(int argc, char *argv[]) {
	float a[N][N], a_inv[N][N];
	int i, j, k;
	float scalar;

	for(i = 0; i < N; i++)
		for(j = 0; j < N; j++) {
			scanf("%f", &a[i][j]);
			if(i == j)
				a_inv[i][j] = 1;
			else
				a_inv[i][j] = 0;
		}

	for(i = 0; i < N; i++) {
		// Pivot the matrix
		pivotRow(a, a_inv, i);

		// Make column entry to be one
		normalizeRow(a, a_inv, i);

		// Transform all other rows
		for(k = 0; k < N; k++) {
			if(i == k)
				continue;
			scalar = a[k][i];
			if(scalar == 0)
				continue;
			for(j = 0; j < N; j++) {
				a[k][j] -= (scalar * a[i][j]);
				a_inv[k][j] -= (scalar * a_inv[i][j]);
				printf("Transform [%d][%d]:\n", k, j);
				printMatrix(a);
			}
		}
	}

	//verifyInverse(a, a_inv);

	printf("Inverse is:\n");
	printMatrix(a_inv);

	return 0;
}
