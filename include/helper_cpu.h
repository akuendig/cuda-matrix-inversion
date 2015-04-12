#ifndef HEADER_HELPER_CPU_INCLUDED
#define HEADER_HELPER_CPU_INCLUDED

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

#ifdef __cplusplus
extern "C" {
#endif
void printMatrix(Array a, int M, int N);
void printMatrixList(Array a, int N, int batchSize);
void readMatricesFile(const char *path, int *numMatrices, int *m, int *n, Array *matrices);
void replicateMatrices(Array *matrices, const int M, const int N, const int numMatrices, const int numReplications);
#ifdef __cplusplus
}
#endif

#endif
