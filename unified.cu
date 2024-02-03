#include <stdio.h>
#include <cuda_runtime.h>

#define N 100

__global__ void addKernel(int *a, int *b, int *c) {
    int tid = threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int *a, *b, *c;

    // Allocate managed memory with cudaMallocManaged()
    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&b, N * sizeof(int));
    cudaMallocManaged(&c, N * sizeof(int));

    // Initialize data in the CPU
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = N - i;
    }

    // Launch kernel on the GPU
    addKernel<<<1, N>>>(a, b, c);

    // Wait for the kernel to complete
    cudaDeviceSynchronize();

    // Print the results
    for (int i = 0; i < N; i++) {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    // Free the managed memory with cudaFree()
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}

