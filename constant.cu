#include <stdio.h>

__constant__ int const_array[4];

__global__ void kernel(int* array) {
  int tid = threadIdx.x;
  //array[tid] += const_array[tid];
  printf("%d, %d, %d\n ",tid,const_array[tid],array[tid]);
}

int main() {
  int array[4] = {1, 2, 3, 4};
  int size = 4 * sizeof(int);
  int* device_array;

  cudaMalloc(&device_array, size);
  cudaMemcpy(device_array, array, size, cudaMemcpyHostToDevice);
  //cudaMemcpy(const_array, array, size, cudaMemcpyHostToDevice);

  cudaMemcpyToSymbol(const_array, array, size);

  kernel<<<1, 4>>>(device_array);

  cudaFree(device_array);
  return 0;
}

