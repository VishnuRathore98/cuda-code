#include <stdio.h>

// __global__ means this function runs on the GPU (device)
// and can be launched from the CPU (host)
__global__ void cuda_hello()
{
    // Unique thread ID across the whole grid
    // (block index * threads per block) + thread index inside block
    // int id = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("Hello World from GPU! %d \n", id);

    // Index of the thread inside its block (0 → blockDim.x - 1)
    printf("threadIdx.x %d \n", threadIdx.x);

    // Index of the block inside the grid (0 → gridDim.x - 1)
    printf("blockIdx.x %d \n", blockIdx.x);

    // Total number of threads inside each block
    printf("blockDim.x %d \n", blockDim.x);

    // Total number of blocks inside the grid
    printf("gridDim.x %d \n", gridDim.x);
}

int main()
{
    // Kernel launch configuration <<<number_of_blocks, threads_per_block>>>
    // This launches:
    // 2 blocks × 4 threads = 8 GPU threads total
    cuda_hello<<<2, 4>>>();

    // Wait for GPU to finish execution before CPU exits
    // (Otherwise program may terminate before printf executes)
    cudaDeviceSynchronize();

    return 0;
}
