__device__ int myGlobalVariable[10] = {0,1,2,3,4,5,6,7,8,9};

__global__ void myKernel() {
	    int tid = threadIdx.x;
	    myGlobalVariable[tid] = tid;
	    printf("Thread %d set myGlobalVariable to %d\n", tid, myGlobalVariable[tid]);
}

