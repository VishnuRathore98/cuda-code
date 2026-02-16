#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 10000000

__global__ void saxpy(float a, float *X, float *Y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        Y[i] = a * X[i] + Y[i];
}

int main()
{
    size_t bytes = N * sizeof(float);

    // Host memory
    float *h_X = new float[N];
    float *h_Y = new float[N];

    for (int i = 0; i < N; i++)
    {
        h_X[i] = 1.0f;
        h_Y[i] = 2.0f;
    }

    // Device memory
    float *d_X, *d_Y;
    cudaMalloc(&d_X, bytes);
    cudaMalloc(&d_Y, bytes);

    cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, h_Y, bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    auto start = std::chrono::high_resolution_clock::now();

    saxpy<<<blocks, threads>>>(2.5f, d_X, d_Y, N);
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end - start;

    cudaMemcpy(h_Y, d_Y, bytes, cudaMemcpyDeviceToHost);

    std::cout << "GPU time: " << gpu_time.count() << " seconds\n";
    std::cout << "First element: " << h_Y[0] << std::endl;

    cudaFree(d_X);
    cudaFree(d_Y);
    delete[] h_X;
    delete[] h_Y;

    return 0;
}
