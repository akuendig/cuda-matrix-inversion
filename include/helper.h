#ifndef HEADER_HELPER_INCLUDED
#define HEADER_HELPER_INCLUDED

#define MAX_MATRIX_BYTE_READ 67108864

#define fail(...) \
  fprintf(stderr, "%s:%d\t", __FILE__, __LINE__); \
  fprintf(stderr, __VA_ARGS__); \
  fprintf(stderr, "\r\n"); \
  exit(EXIT_FAILURE);

#define ensure(condition, ...) \
  do { \
    if (! (condition)) { \
      fprintf(stderr, "ENSURE FAILED %s:%d\r\n", __FILE__, __LINE__); \
      fprintf(stderr, __VA_ARGS__); \
      fprintf(stderr, "\r\n"); \
      if (errno) { perror("possible reason for failure from ERRNO"); } \
      exit(EXIT_FAILURE); \
    } \
  } while(0)

// Reads `numMatrices` matrices from file at `path` where each matrix has dimension `m` x `n`.
// Memory is allocated in one big block, so `matrices` points to the beginning of all matrices.
inline static void readMatricesFile(const char *path, int *numMatrices, int *m, int *n, Array *matrices) {
    int ret;
    int _numMatrices, _m, _n;

    FILE* fp = fopen(path, "r");
    ensure(fp, "could not open matrix file %s", path);

    ret = fscanf(fp, "%d %d %d", &_numMatrices, &_m, &_n);
    ensure(3 == ret, "could not read number of matrices from file %s", path);

    *numMatrices = _numMatrices;
    *m = _m;
    *n = _n;

    size_t arraySize = sizeof(ELEMENT_TYPE) * (_numMatrices) * (_m) * (_n);
    ensure(arraySize <= MAX_MATRIX_BYTE_READ, "cannot read file %s because "
        "the allocated array would be bigger than 0x%lX bytes", path, arraySize);

    *matrices = (Array)malloc(arraySize);
    ensure(*matrices, "could not allocate 0x%lX bytes of memory for file %s", arraySize, path);

    int k, i, j;

    for (k = 0; k < _numMatrices; ++k) {
        Array firstElement = *matrices;
        Array currentMatrix = firstElement + k*_m*_n;

        // Read row by row
        for (i = 0; i < _m; ++i) {
            for (j = 0; j < _n; ++j) {
                ret = fscanf(fp, "%f", &currentMatrix[j*_m + i]);
                ensure(ret, "could not read matrix from file %s, stuck at matrix %d element %d, %d", path, k, i, j);
            }
        }
    }

    fclose(fp);
}

// Prints matrix a stored in column major format
inline static void printMatrix(Array a, int M, int N) {
    int i, j;

    for(i = 0; i < M; i++) {
        for(j = 0; j < N; j++)
            printf("%f\t", a[j * M + i]);
        printf("\n");
    }
    printf("\n");
}

#ifdef __CUDACC__
/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline static void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s:%d\n", cudaGetErrorString(code), file, line);
        cudaDeviceReset();
        if (abort) { exit(code); }
    }
}
#endif

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}

#define cublasErrchk(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline static void cublasAssert(cublasStatus_t code, const char *file, int line)
{
    if (code != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr,"cuBLASassert: %s %s:%d\n", _cudaGetErrorEnum(code), file, line);
        cudaDeviceReset();
        exit(code);
    }
}

#endif

#endif
