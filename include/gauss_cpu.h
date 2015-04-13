#ifndef HEADER_GAUSS_CPU_INCLUDED
#define HEADER_GAUSS_CPU_INCLUDED

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Calculates the mean of the matrix set {A, B, C, D}.
// Mean = A*(B+C)^{-1}*D
// As       batchSize x n x 1
// Bs       batchSize x n x n
// Cs       batchSize x n x 1
// Ds       batchSize x n x 1
// Means    batchSize x n x 1
// Means is assumed to be already allocated.
void calcluateMeanCPU(
    int n,
    Array As,
    Array Bs,
    Array Cs,
    Array Ds,
    Array Means,
    int batchSize);
void calcluateMeanSolveCPU(
    int n,
    Array As,
    Array Bs,
    Array Cs,
    Array Ds,
    Array Means,
    int batchSize);

// Calculates the variance of the matrix set {A, B, C, E}.
// Var = E-AT*(B+C)^{-1}*A
// As       batchSize x n x 1
// Bs       batchSize x n x n
// Cs       batchSize x n x 1
// Es       batchSize x 1 x 1
// Variances    batchSize x 1 x 1
// Variances is assumed to be already allocated.
//
// Bs and Cs are destroyed
void calcluateVarianceCPU(
    int n,
    Array As,
    Array Bs,
    Array Cs,
    Array Es,
    Array Variances,
    int batchSize);
void calcluateVarianceSolveCPU(
    int n,
    Array As,
    Array Bs,
    Array Cs,
    Array Es,
    Array Variances,
    int batchSize);

#ifdef __cplusplus
}
#endif

#endif