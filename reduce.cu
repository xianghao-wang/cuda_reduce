#include <cstdio>
#include <clocale>
#include <cuda.h>

#include "get_time.h"

#ifndef __OPT__
#define __OPT__ 0
#endif

#ifndef TB_SIZE
#define TB_SIZE 1024
#endif

#ifndef NITERS
#define NITERS 8
#endif

#define CONCAT(a, b) a ## b
#define REDUCE(opt) CONCAT(reduce_opt, opt) 

using namespace std;

__global__ void reduce_opt0(double *A, double *blockSums, int n)
{
    unsigned int tid, idx, nThreads;
    unsigned int j, offset;
    __shared__ double cached[TB_SIZE];

    tid = threadIdx.x;
    idx = blockIdx.x * blockDim.x + threadIdx.x;
    nThreads = gridDim.x * blockDim.x;

    // Reduce elements to each threads
    cached[tid] = 0.0;
    j = idx;
    while (j < n)
    {
        cached[tid] += A[j];
        j += nThreads;
    }
    __syncthreads();

    // Reduce threads to a block
    for (offset = 1; offset < blockDim.x; offset *= 2)
    {
        if (tid % (2 * offset) == 0)
        {
            cached[tid] += cached[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
        blockSums[blockIdx.x] = cached[0];
}

__global__ void reduce_opt1(double *A, double *blockSums, int n)
{
    unsigned int tid, idx, nThreads;
    unsigned int j, offset, s;
    __shared__ double cached[TB_SIZE];

    tid = threadIdx.x;
    idx = blockIdx.x * blockDim.x + threadIdx.x;
    nThreads = gridDim.x * blockDim.x;

    // Reduce elements to each threads
    cached[tid] = 0.0;
    j = idx;
    while (j < n)
    {
        cached[tid] += A[j];
        j += nThreads;
    }
    __syncthreads();

    // Reduce threads to a block
    s = blockDim.x / 2;
    for (offset = 1; offset < blockDim.x; offset *= 2)
    {
        if (tid < s)
        {
            cached[(offset * 2) * tid] += cached[(offset * 2) * tid + offset];
        }
        s /= 2;
        __syncthreads();
    }

    if (tid == 0)
        blockSums[blockIdx.x] = cached[0];
}

__global__ void reduce_opt2(double *A, double *blockSums, int n)
{
    unsigned int tid, idx, nThreads;
    unsigned int j, offset;
    __shared__ double cached[TB_SIZE];

    tid = threadIdx.x;
    idx = blockIdx.x * blockDim.x + threadIdx.x;
    nThreads = gridDim.x * blockDim.x;

    // Reduce elements to each threads
    cached[tid] = 0.0;
    j = idx;
    while (j < n)
    {
        cached[tid] += A[j];
        j += nThreads;
    }
    __syncthreads();

    // Reduce threads to a block
    for (offset = blockDim.x / 2; offset > 0; offset /= 2)
    {
        if (tid < offset)
        {
            cached[tid] += cached[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0)
        blockSums[blockIdx.x] = cached[0];
}


int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <size>\n", argv[0]);
        exit(1);
    }

    int n, numBlocks;
    double *A, *blockSums;
    double *A_dev, *blockSums_dev;
    double sum;
    int i;

    timer t;
    double tms;

    cudaDeviceReset();
    setlocale(LC_NUMERIC, "");

    // Get dimensions
    n = atoi(argv[1]);
    numBlocks = (n + TB_SIZE -1) / TB_SIZE;

    // Prepare data
    A = new double[n];
    blockSums = new double[numBlocks];
    for (i = 0; i < n; ++ i)
        A[i] = (double) i;
    cudaMalloc(&A_dev, n * sizeof(double));
    cudaMalloc(&blockSums_dev, numBlocks * sizeof(double));
    cudaMemcpy(A_dev, A, n * sizeof(double), cudaMemcpyHostToDevice);

    // Print status
    printf("=========================================\n");
    printf("= Running on kernel with optimization %d =\n", __OPT__);
    printf("=========================================\n\n");
    printf("Total %'d threads are launched\n", numBlocks * TB_SIZE);
    printf("Total %'d blocks are launched with %d block size\n", numBlocks, TB_SIZE);

    // Warmup
    for (i = 0; i < 5; ++ i)
    {
        REDUCE(__OPT__)<<<numBlocks, TB_SIZE>>>(A_dev, blockSums_dev, n);
        cudaDeviceSynchronize();
    }

    // Bench
    t.start();
    for (i = 0; i < NITERS; ++ i) {
        REDUCE(__OPT__)<<<numBlocks, TB_SIZE>>>(A_dev, blockSums_dev, n);
        cudaDeviceSynchronize();
    }
    tms = t.next_time() * 1e3;


    cudaMemcpy(blockSums, blockSums_dev, numBlocks * sizeof(double), cudaMemcpyDeviceToHost);

    // Calculate sum
    for (i = 0; i < numBlocks; ++ i)
        sum += blockSums[i];
    printf("Result:     %10e\n", sum);
    printf("Expected:   %10e\n", (n - 1.0) * n / 2.0);
    printf("Average kernel time: %5.3lfms\n", tms / NITERS);

    // Release memory
    cudaFree(A_dev);
    cudaFree(blockSums_dev);
    delete [] A;
    delete [] blockSums;
}
