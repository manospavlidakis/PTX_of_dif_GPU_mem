#include <stdio.h>
__global__ void myKernel(int n) {
    float* d_data = (float*)malloc(n * sizeof(float));

    // Do some computation with the allocated memory
    d_data[n-1] = 1024;

    // Free the memory when done
    free(d_data);
}

int main() {
    int n = 1000;
    size_t limit = 1024*1024*256;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit);
    //limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
    printf("cudaLimitMallocHeapSize: %u\n", (unsigned)limit);
    // Launch the kernel with 1 block and 1 thread per block
    myKernel<<<1, 1>>>(n);

    // Wait for the kernel to finish
    cudaDeviceSynchronize();
    return 0;
}

