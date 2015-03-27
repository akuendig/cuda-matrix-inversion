#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <errno.h>

#include <iostream>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include "../include/helper.h"

#define ELEMENT_TYPE float

void fillRandom(char *s, const int len) {
    static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";

    for (int i = 0; i < len; ++i) {
        s[i] = alphanum[rand() % (sizeof(alphanum) - 1)];
    }

    s[len-1] = 0;
}

void benchmarkMalloc(const int numReplications, const int numElems) {
    const size_t sizeOfData = sizeof(ELEMENT_TYPE)*numElems;
    float timeOfMallocSum, timeOfFreeSum;
    ELEMENT_TYPE *devData;
    cudaEvent_t start, stop;

    printf("Benchmark MALLOC - Replications: %d Elements: %d\n", numReplications, numElems);

    gpuErrchk( cudaEventCreate(&start) );
    gpuErrchk( cudaEventCreate(&stop) );

    for (int i = 0; i < numReplications; ++i) {
        gpuErrchk( cudaEventRecord(start) );
        gpuErrchk( cudaMalloc((void**)&devData, sizeOfData) );
        gpuErrchk( cudaEventRecord(stop) );

        gpuErrchk( cudaEventSynchronize(stop) );
        float timeOfMalloc = 0;
        gpuErrchk( cudaEventElapsedTime(&timeOfMalloc, start, stop) );

        gpuErrchk( cudaEventRecord(start) );
        gpuErrchk( cudaFree(devData) );
        gpuErrchk( cudaEventRecord(stop) );

        gpuErrchk( cudaEventSynchronize(stop) );
        float timeOfFree = 0;
        gpuErrchk( cudaEventElapsedTime(&timeOfFree, start, stop) );

        printf("Benchmark MALLOC - malloc: %fms free: %fms\n", timeOfMalloc, timeOfFree);

        timeOfMallocSum += timeOfMalloc;
        timeOfFreeSum += timeOfFree;
    }

    printf("Benchmark MALLOC - malloc: %f\n", (timeOfMallocSum/float(numReplications)));
    printf("Benchmark MALLOC - free: %f\n", (timeOfFreeSum/float(numReplications)));
}

void benchmarkTransfer(const int numReplications, const int numElems) {
    const size_t sizeOfData = sizeof(ELEMENT_TYPE)*numElems;
    float timeToDeviceSum, timeFromDeviceSum;
    ELEMENT_TYPE *data, *devData;
    cudaEvent_t start, stop;

    printf("Benchmark TRANSFER - Replications: %d Elements: %d\n", numReplications, numElems);

    data = (ELEMENT_TYPE *)malloc(sizeOfData);
    ensure(data, "Could not allocate host memory");

    gpuErrchk( cudaEventCreate(&start) );
    gpuErrchk( cudaEventCreate(&stop) );

    gpuErrchk( cudaMalloc((void**)&devData, sizeOfData) );

    fillRandom((char*)data, sizeOfData);

    for (int i = 0; i < numReplications; ++i) {
        gpuErrchk( cudaEventRecord(start) );
        gpuErrchk( cudaMemcpy(devData, data, sizeOfData, cudaMemcpyHostToDevice) );
        gpuErrchk( cudaEventRecord(stop) );

        gpuErrchk( cudaEventSynchronize(stop) );
        float timeToDevice = 0;
        gpuErrchk( cudaEventElapsedTime(&timeToDevice, start, stop) );

        gpuErrchk( cudaEventRecord(start) );
        gpuErrchk( cudaMemcpy(data, devData, sizeOfData, cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaEventRecord(stop) );

        gpuErrchk( cudaEventSynchronize(stop) );
        float timeFromDevice = 0;
        gpuErrchk( cudaEventElapsedTime(&timeFromDevice, start, stop) );

        printf("Benchmark TRANSFER - To: %fms From: %fms\n", timeToDevice, timeFromDevice);

        timeToDeviceSum += timeToDevice;
        timeFromDeviceSum += timeFromDevice;
    }

    printf("Benchmark TRANSFER - Bandwidth to Device (GB/s): %f\n", sizeOfData/(timeToDeviceSum/float(numReplications))/1e6);
    printf("Benchmark TRANSFER - Bandwidth from Device (GB/s): %f\n", sizeOfData/(timeFromDeviceSum/float(numReplications))/1e6);

    gpuErrchk( cudaFree(devData) );
    free(data);
};

void benchmarkTransferPinned(const int numReplications, const int numElems) {
    const size_t sizeOfData = sizeof(ELEMENT_TYPE)*numElems;
    float timeToDeviceSum, timeFromDeviceSum;
    ELEMENT_TYPE *data, *devData;
    cudaEvent_t start, stop;

    printf("Benchmark TRANSFER PINNED - Replications: %d Elements: %d\n", numReplications, numElems);

    gpuErrchk( cudaEventCreate(&start) );
    gpuErrchk( cudaEventCreate(&stop) );

    gpuErrchk( cudaHostAlloc((void**)&data, sizeOfData, cudaHostAllocDefault) );
    gpuErrchk( cudaMalloc((void**)&devData, sizeOfData) );

    fillRandom((char*)data, sizeOfData);

    for (int i = 0; i < numReplications; ++i) {
        gpuErrchk( cudaEventRecord(start) );
        gpuErrchk( cudaMemcpy(devData, data, sizeOfData, cudaMemcpyHostToDevice) );
        gpuErrchk( cudaEventRecord(stop) );

        gpuErrchk( cudaEventSynchronize(stop) );
        float timeToDevice = 0;
        gpuErrchk( cudaEventElapsedTime(&timeToDevice, start, stop) );

        gpuErrchk( cudaEventRecord(start) );
        gpuErrchk( cudaMemcpy(data, devData, sizeOfData, cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaEventRecord(stop) );

        gpuErrchk( cudaEventSynchronize(stop) );
        float timeFromDevice = 0;
        gpuErrchk( cudaEventElapsedTime(&timeFromDevice, start, stop) );

        printf("Benchmark TRANSFER PINNED - To: %fms From: %fms\n", timeToDevice, timeFromDevice);

        timeToDeviceSum += timeToDevice;
        timeFromDeviceSum += timeFromDevice;
    }

    printf("Benchmark TRANSFER PINNED - Bandwidth to Device (GB/s): %f\n", sizeOfData/(timeToDeviceSum/float(numReplications))/1e6);
    printf("Benchmark TRANSFER PINNED - Bandwidth from Device (GB/s): %f\n", sizeOfData/(timeFromDeviceSum/float(numReplications))/1e6);

    gpuErrchk( cudaFree(devData) );
    gpuErrchk( cudaFree(data) );
};

void benchmarkTransfer2D(const int numReplications, const int numElems, const int numArrays) {
    const size_t sizeOfData = sizeof(ELEMENT_TYPE)*numElems;
    float timeToDeviceSum, timeFromDeviceSum;
    ELEMENT_TYPE *data, *devData;
    size_t pitch;
    cudaEvent_t start, stop;

    printf("Benchmark TRANSFER 2D - Replications: %d Elements: %d Arrays: %d\n", numReplications, numElems, numArrays);

    gpuErrchk( cudaEventCreate(&start) );
    gpuErrchk( cudaEventCreate(&stop) );

    gpuErrchk( cudaHostAlloc((void**)&data, sizeOfData*numArrays, cudaHostAllocDefault) );
    gpuErrchk( cudaMallocPitch((void**)&devData, &pitch, sizeOfData, numArrays) );

    fillRandom((char*)data, sizeOfData*numArrays);

    for (int i = 0; i < numReplications; ++i) {
        gpuErrchk( cudaEventRecord(start) );
        gpuErrchk( cudaMemcpy2D(devData, pitch, data, sizeOfData, sizeOfData, numArrays,
               cudaMemcpyHostToDevice) );
        gpuErrchk( cudaEventRecord(stop) );

        gpuErrchk( cudaEventSynchronize(stop) );
        float timeToDevice = 0;
        gpuErrchk( cudaEventElapsedTime(&timeToDevice, start, stop) );

        gpuErrchk( cudaEventRecord(start) );
        gpuErrchk( cudaMemcpy2D(data, sizeOfData, devData, pitch, sizeOfData, numArrays,
               cudaMemcpyDeviceToHost) );
        gpuErrchk( cudaEventRecord(stop) );

        gpuErrchk( cudaEventSynchronize(stop) );
        float timeFromDevice = 0;
        gpuErrchk( cudaEventElapsedTime(&timeFromDevice, start, stop) );

        printf("Benchmark TRANSFER 2D - To: %fms From: %fms\n", timeToDevice, timeFromDevice);

        timeToDeviceSum += timeToDevice;
        timeFromDeviceSum += timeFromDevice;
    }

    printf("Benchmark TRANSFER 2D - Bandwidth to Device (GB/s): %f\n", sizeOfData/(timeToDeviceSum/float(numReplications))/1e6);
    printf("Benchmark TRANSFER 2D - Bandwidth from Device (GB/s): %f\n", sizeOfData/(timeFromDeviceSum/float(numReplications))/1e6);

    gpuErrchk( cudaFree(devData) );
    gpuErrchk( cudaFree(data) );
};

int main(int argc, char const *argv[])
{
    ensure(argc >= 4, "Usage: %s NUM_REPLICATIONS NUM_ELEMENTS NUM_ARRAYS", argv[0]);

    int numReplications = atoi(argv[1]);
    int numElems = atoi(argv[2]);
    int numArrays = atoi(argv[3]);

    benchmarkMalloc(numReplications, numElems);
    benchmarkTransfer(numReplications, numElems);
    benchmarkTransferPinned(numReplications, numElems);
    benchmarkTransfer2D(numReplications, numElems, numArrays);

    /* code */
    return 0;
}
