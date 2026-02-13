#include <stdio.h>

// GPU kernel (runs on the device, launched by CPU)
__global__ void cuda_hello()
{
    // Each GPU thread executes this line
    printf("Hello World from GPU!");

    // Kernel does NOT "return control" to CPU.
    // It simply finishes its work. The CPU was never paused.
}

int main()
{
    // CPU launches a kernel: schedules work on GPU
    // 1 block, 1 thread
    cuda_hello<<<1, 1>>>();

    // CPU waits here until GPU finishes all previously launched work
    cudaDeviceSynchronize();

    // Program ends on CPU
    return 0;
}
