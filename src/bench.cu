#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>

#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "../include/helper.h"

#define ELEMENT_TYPE float

// int benchmarkMalloc(const int numAllocs, const int numElems) {

// }

void benchmarkTransfer(const int numReplications, const int numElems) {
    const size_t sizeOfData = sizeof(ELEMENT_TYPE)*numElems;
    const ELEMENT_TYPE *data = (ELEMENT_TYPE *)malloc(sizeOfData);
    float timeToDeviceSum, timeFromDeviceSum;

    printf("Benchmark TRANSFER - Replications: %d Elements: %d\n", numReplications, numElems);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float *devData;

    gpuErrchk( cudaMalloc((void**)&devData, sizeOfData) );

    for (int i = 0; i < numReplications; ++i) {
        cudaEventRecord(start);
        cudaMemcpy(devData, data, sizeOfData, cudaMemcpyHostToDevice);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float timeToDevice = 0;
        cudaEventElapsedTime(&timeToDevice, start, stop);

        cudaEventRecord(start);
        cudaMemcpy(devData, data, sizeOfData, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop);

        cudaEventSynchronize(stop);
        float timeFromDevice = 0;
        cudaEventElapsedTime(&timeFromDevice, start, stop);

        printf("Benchmark TRANSFER - To: %f From: %f\n", timeToDevice, timeFromDevice);

        timeToDeviceSum += timeToDevice;
        timeFromDeviceSum += timeFromDevice;
    }

    printf("Benchmark TRANSFER - Bandwidth to Device (GB/s): %f\n", sizeOfData/(timeToDeviceSum/float(numReplications))/1e6);
    printf("Benchmark TRANSFER - Bandwidth from Device (GB/s): %f\n", sizeOfData/(timeFromDeviceSum/float(numReplications))/1e6);

    gpuErrchk( cudaFree(devData) );
    free((void*)data);
};

int main(int argc, char const *argv[])
{
    benchmarkTransfer(3, 1024*1024*200);
    /* code */
    return 0;
}
