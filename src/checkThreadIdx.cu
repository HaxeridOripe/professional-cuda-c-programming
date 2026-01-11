#include "stdio.h"
#include "cuda_runtime.h"

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

void initialInt(int *ip, int size){
    for(int i=0; i<size; i++){
        ip[i] = i;
    }
}

void printMatrix(int *C, int nx, int ny){
    int *ic = C;
    printf("\nMatrix: (%d,%d)\n", nx, ny);
    for(int i=0; i<ny; i++){
        for(int j=0; j<nx; j++){
            printf("%3d ", ic[j]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}

__global__ void printThreadIdx(int *A, const int nx, const int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) coordinate:(%d,%d) Global idx:%d A[idx]=%d\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z,
           ix, iy,
           idx, A[idx]);
}

void main(){
    int dev = 0;
    CHECK(cudaSetDevice(dev));
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    
    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    int *h_A = (int *)malloc(nBytes);
    initialInt(h_A, nxy);
    printMatrix(h_A, nx, ny);
    int *d_A;
    cudaMalloc((int**)&d_A, nBytes);
    cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
    dim3 block(4, 2);
    dim3 grid((nx + block.x -1)/block.x, (ny + block.y -1)/block.y);
    printf("grid:(%d,%d,%d) block:(%d,%d,%d)\n", 
           grid.x, grid.y, grid.z,
           block.x, block.y, block.z);
    printThreadIdx<<<grid, block>>>(d_A, nx, ny);
    cudaDeviceSynchronize();

    cudaFree(d_A);
    free(h_A);
}