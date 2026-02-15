# The program multiplies every element of a large array by a constant using thousands of GPU threads in parallel.
# Data is copied from CPU RAM → GPU VRAM, processed by a kernel, then copied back.
# Each thread computes one index using blockIdx * blockDim + threadIdx.
# if (i < n) prevents extra threads from accessing invalid memory.
# It teaches that GPU speed often depends on memory transfer/bandwidth, not just computation.

#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 10000000   // 10 million elements

// GPU kernel
__global__ void vectorScale(float *A, float alpha, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        A[i] = alpha * A[i];
}

int main()
{
    size_t bytes = N * sizeof(float);

    // Host memory
    float *h_A = new float[N];

    for (int i = 0; i < N; i++)
        h_A[i] = 1.0f;

    // Device memory
    float *d_A;
    cudaMalloc(&d_A, bytes);

    // Copy to GPU
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

    // Launch configuration
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // ---- Time GPU ----
    auto start = std::chrono::high_resolution_clock::now();

    vectorScale<<<blocks, threads>>>(d_A, 2.5f, N);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end - start;

    // Copy back
    cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost);

    std::cout << "GPU time: " << gpu_time.count() << " seconds\n";
    std::cout << "First element: " << h_A[0] << std::endl;

    cudaFree(d_A);
    delete[] h_A;

    return 0;
}

