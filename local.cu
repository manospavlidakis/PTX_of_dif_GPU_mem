#include <iostream>
#include <cuda_runtime.h>

using namespace std;
extern "C"
__global__ void vector_add(float * A, const float * B, float * C, const int size, const int local_memory_size)
{
   int item = (blockIdx.x * blockDim.x) + threadIdx.x;
   float local_memory[2];
   
   if ( item < size )
   {
      printf("%d, %d\n", item,size);
      local_memory[0] = 1024;
      local_memory[1] = B[item];
      local_memory[3] = local_memory[0] + local_memory[1];
      C[item] = local_memory[64];
      if (item == 1 )
	      A = &local_memory[0];
      if (item == 2)
	      printf("A[0]: %f\n", A[0]);
   }
}

int main() {
    const int size = 1000;
    const int local_memory_size = 2;
    const int num_blocks = (size + 255) / 256;
    const int num_threads_per_block = 256;
    std::cerr<<"threads: "<<num_threads_per_block<<" blocks: "<<num_blocks<<std::endl;

    // Allocate device memory
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, size * sizeof(float));
    cudaMalloc((void**)&d_B, size * sizeof(float));
    cudaMalloc((void**)&d_C, size * sizeof(float));

    // Allocate host memory and initialize inputs
    float* h_A = new float[size];
    float* h_B = new float[size];
    float* h_C = new float[size];
    for (int i = 0; i < size; i++) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    vector_add<<<num_blocks, num_threads_per_block>>>(d_A, d_B, d_C, size, local_memory_size);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
	    printf("Error Device Synchronize: %s\n", cudaGetErrorString(err));
	    return 1;
    }

    // Copy output from device to host
    cudaMemcpy(h_C, d_C, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print output
    /*for (int i = 0; i < size; i++) {
        cout << h_C[i] << " ";
    }
    cout << endl;
*/
    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
