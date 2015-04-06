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

#define div_ceil(x, y) (1 + (((x) - 1) / (y)))
  // {
  // ensure(x > 0, "Ceiling only works for positive division");
  // ensure(y > 0, "Ceiling only works for positive division");
  // return 1 + ((x - 1) / y); // if x != 0
// }

void printMatrix(Array a, int M, int N);
void printMatrixList(Array a, int N, int batchSize);
void readMatricesFile(const char *path, int *numMatrices, int *m, int *n, Array *matrices);

cudaError_t batchedCudaMalloc(Array* devArrayPtr, size_t *pitch, size_t arraySize, int batchSize);

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline static void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s:%d\n", cudaGetErrorString(code), file, line);
        cudaDeviceReset();
        exit(code);
    }
}

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
