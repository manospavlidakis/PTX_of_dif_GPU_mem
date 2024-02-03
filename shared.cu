#include <stdio.h>

__global__ void reduce(float* input, float* output, int size) {
  extern __shared__ float shared[];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid < size) {
    shared[tid] = input[gid];
    printf("ptr: %f\n",shared[257]);
  } else {
    shared[tid] = 0.0f;
  }

  __syncthreads();

  for (int i = blockDim.x / 2; i > 0; i /= 2) {
    if (tid < i) {
      shared[tid] += shared[tid + i];
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[blockIdx.x] = shared[0];
  }
}

int main() {
  const int size = 64;
  const int threads_per_block = 32;
  const int num_blocks = (size + threads_per_block - 1) / threads_per_block;

  float input[size];
  float output[num_blocks];

  for (int i = 0; i < size; i++) {
    input[i] = 1.0f;
  }

  float* device_input;
  float* device_output;

  cudaMalloc(&device_input, size * sizeof(float));
  cudaMalloc(&device_output, num_blocks * sizeof(float));

  cudaMemcpy(device_input, input, size * sizeof(float), cudaMemcpyHostToDevice);

  reduce<<<num_blocks, threads_per_block, threads_per_block * sizeof(float)>>>(device_input, device_output, size);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
	  printf("Error launching kernel: %s\n", cudaGetErrorString(err));
	  return 1;
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
	  printf("Error Device Synchronize: %s\n", cudaGetErrorString(err));
	  return 1;
  }

  cudaMemcpy(output, device_output, num_blocks * sizeof(float), cudaMemcpyDeviceToHost);

  float result = 0.0f;

  for (int i = 0; i < num_blocks; i++) {
    result += output[i];
  }

  printf("Result: %f\n", result);

  cudaFree(device_input);
  cudaFree(device_output);

  return 0;
}

