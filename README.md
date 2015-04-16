# cuda-matrix-inversion

To build in debug mode ```make dbg=1```

To run ```./gauss_bench TEST_FOLDER NUM_REPLICATIONS [-d]```

export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
make clean inverse_bench gauss_bench log=0 solve=0 dbg=0
BENCH_REPS=10 BENCH_MAX_DUPS=10 BENCH_NUM_THREADS=4 make run-gauss-bench

### Notes on execution

Lets first start with the basic sequence of operations to perform a *mean* computation. The calculated formula is ```A*(B+C)^{-1}*D```

In short ```add -> inv -> gemv -> dot```

1. Allocate space for
  - B       batchSize x n x n
  - C       batchSize x n x n
  - D       batchSize x n x 1

  C is diagonal so could also be only batchSize x n x 1.
2. Copy the matrices B, C, and D
3. Add B and C (this could be done on CPU to save transfer time of C)
4. Invert result
5. Multiply D with result to get a vector
6. Load A into D
7. Scalar product of A with the result vector
8. Read back the resulting scalar

The most expensive step is the inversion of the matrix and the multiplication of the two matrices.

### Notes on parallelism
There are two different ways of parallelizing: Parallelism within the computation by batching and data parallelism inside kernels AND parallelism between matrices of different dimensions.
- Add: Can only batch but needs to be implemented. Could be implemented by having each kernel working on one matrix (diagonal) addition.
- Inv: Can be batched and blocked. Depends on implementation.
- Gemv: Can be batch using gemm_batched. Could also parallelise on element.
- Dot: Can only batch but needs to be implemented, see Add.

#### Multiple stream queues
During discussions, the requirement of changing matrix dimensions came up. The reason for this is that the matrices represent sensor data. As soon as new sensor data arrives, it gets appended to the current data and the matrix dimension increases. This means, that during program execution the matrices can have varying numbers of dimensions (need to clarify of all have the same dimension). If not all the matrices have the same dimension we can split the calculation up into multiple streams and calculate them separately. To partition them we could create queues of maximum sizes, e.g. 32, 128, 512, 1024, and put the matrix into the corresponding queue. Then after one data collection round we could execute each queue in one batch in parallel.

Having multiple queues only makes sense, if the batch calculation does **not** exhaust the computing resources of the GPU. I can not imagine this happening a lot. This area definitely needs more profiling.

### Memory Layout

```
________________________________________________________________
| A_1: col 1, col 2, col 3, .., col n | padding | A_2: col 1, ..
----------------------------------------------------------------
```
