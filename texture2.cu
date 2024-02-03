#include <stdio.h>
#include <stdint.h>

typedef uint8_t mt;  // use an integer type

__global__ void kernel(cudaTextureObject_t tex, int *outputData)
{
    int x = threadIdx.x;
    //int y = 98; //NO error!!!!
    int y = threadIdx.y;
    mt val = tex2D<mt>(tex, x, y);
    outputData[0] = val;
    

    //mt val = tex2D<mt>(tex, x, y);
//    printf("x=%d, y=%d, val=%d \n",x,y, val);
}

int main(int argc, char **argv)
{
    int *dData = NULL;
    cudaMalloc((void **) &dData, 5);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("texturePitchAlignment: %lu\n", prop.texturePitchAlignment);
    cudaTextureObject_t tex;
    const int num_rows = 4;
    const int num_cols = prop.texturePitchAlignment*1; // should be able to use a different multiplier here
    const int ts = num_cols*num_rows;
    const int ds = ts*sizeof(mt);
    mt dataIn[ds];
    for (int i = 0; i < ts; i++) dataIn[i] = i;
    mt* dataDev = 0;
    size_t pitch;
    cudaMallocPitch((void**)&dataDev, &pitch,  num_cols*sizeof(mt), num_rows);
    cudaMemcpy2D(dataDev, pitch, dataIn, num_cols*sizeof(mt), num_cols*sizeof(mt), num_rows, cudaMemcpyHostToDevice);
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = dataDev;
    resDesc.res.pitch2D.width = num_cols;
    resDesc.res.pitch2D.height = num_rows;
    resDesc.res.pitch2D.desc = cudaCreateChannelDesc<mt>();
    resDesc.res.pitch2D.pitchInBytes = pitch;
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);
    dim3 threads(4, 4);
    kernel<<<1, threads>>>(tex, dData);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
          printf("Error launching kernel: %s\n", cudaGetErrorString(err));
          return 1;
    }
    printf("\n");
    return 0;
}
