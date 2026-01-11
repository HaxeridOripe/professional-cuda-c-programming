#include "stdio.h"
#include "cuda_runtime.h"
#include "windows.h"

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

void sumMatrixOnHost(int *A, int *B, int *C, const int nx, const int ny){
    for(int iy=0; iy<ny; iy++){
        for(int ix=0; ix<nx; ix++){
            unsigned int idx = iy * nx + ix;
            C[idx] = A[idx] + B[idx];
        }
    }
}

__global__ void sumMatrixOnGPU(int *A, int *B, int *C, const int nx, const int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
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
    printf("Matrix size: nx=%d ny=%d\n", nx, ny);

    size_t size = nxy * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *hostRef = (float *)malloc(size);
    float *gpuRef = (float *)malloc(size);

    initialInt((int *)h_A, nxy);
    initialInt((int *)h_B, nxy);
    memset(hostRef, 0, size);
    memset(gpuRef, 0, size);
    LARGE_INTEGER frequency;        // 计时器频率
    LARGE_INTEGER t1, t2;           // 计数器值

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&t1);
    sumMatrixOnHost((int *)h_A, (int *)h_B, (int *)hostRef, nx, ny);
    QueryPerformanceCounter(&t2);
    double elapsedTime = (double)(t2.QuadPart - t1.QuadPart) / frequency.QuadPart;
    printf("CPU Execution Time: %f sec\n", elapsedTime);

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((nx + block.x -1)/block.x, (ny + block.y -1)/block.y);
    printf("grid:(%d,%d,%d) block:(%d,%d,%d)\n", 
           grid.x, grid.y, grid.z,
           block.x, block.y, block.z);
    sumMatrixOnGPU<<<grid, block>>>( (int *)d_A, (int *)d_B, (int *)d_C, nx, ny);
    cudaMemcpy(gpuRef, d_C, size, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();

    cudaFree(d_A);
    free(h_A);
}