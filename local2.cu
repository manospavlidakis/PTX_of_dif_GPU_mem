#include <cuda_runtime.h>

__global__ void localLoadKernel(float *input, float *output) {
    __shared__ float local_buffer[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // Copy data from global to local memory
    local_buffer[tid] = input[i];

    // Wait for all threads to finish copying data to local memory
    __syncthreads();

    // Use __ldg to load data from local memory
    float result = 0.0f;
    for (int j = 0; j < blockDim.x; j++) {
        result += __ldg(&local_buffer[j]);
    }

    // Copy result from local memory to global memory
    output[i] = result;
}

