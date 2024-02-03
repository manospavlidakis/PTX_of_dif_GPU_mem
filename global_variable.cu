
__device__ int my_global_var; // Declare a global variable on the device

__global__ void my_kernel() {
    // Access and modify the global variable in the kernel
    my_global_var += threadIdx.x;
    printf("Thread %d: my_global_var = %d\n", threadIdx.x, my_global_var);
}


