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

__global__ void sumMatrixOnGPU22D(int *A, int *B, int *C, const int nx, const int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if(ix < nx && iy < ny){
        C[idx] = A[idx] + B[idx];
    }
    return;
}

__global__ void sumMatrixOnGPU11D(int *A, int *B, int *C, const int nx, const int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if(ix < nx){
        for(int iy=0; iy < ny; iy++){
            int idx =iy * nx +ix;
            C[idx] = A[idx] + B[idx];
        }
    }
    return;
}

__global__ void sumMatrixOnGPU21D(int *A, int *B, int *C, const int nx, const int ny){
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = blockDim.y;
    int idx = iy * nx + ix;
    if(ix < nx && iy < ny){
        C[idx] = A[idx] + B[idx];
    }
    return;
}

void checkResult(float *host, float *device, int N) {
    for (int i = 0; i < N; i++) {
        if (fabs(host[i] - device[i]) > 1e-5) {
            printf("Results do not match at index %d: host %f, device %f\n", i, host[i], device[i]);
            return;
        }
    }
    printf("Results match!\n");
}

// __global__ void printThreadIdx(int *A, const int nx, const int ny){
//     int ix = threadIdx.x + blockIdx.x * blockDim.x;
//     int iy = threadIdx.y + blockIdx.y * blockDim.y;
//     unsigned int idx = iy * nx + ix;
//     printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) coordinate:(%d,%d) Global idx:%d A[idx]=%d\n",
//            threadIdx.x, threadIdx.y, threadIdx.z,
//            blockIdx.x, blockIdx.y, blockIdx.z,
//            ix, iy,
//            idx, A[idx]);
// }

int main(){
    int dev = 0;
    CHECK(cudaSetDevice(dev));
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    
    int nx = 1<<14;
    int ny = 1<<14;
    int nxy = nx * ny;
    printf("Matrix size: nx=%d ny=%d\n", nx, ny);

    size_t size = nxy * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *hostRef = (float *)malloc(size);
    float *gpuRef = (float *)malloc(size);

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
    cudaMalloc((void **)&d_C, size);
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, size);

    printf("---------------------------------------\n");
    printf("Testing 2D block and 2D grid configurations:\n");
    dim3 block(32, 32);
    dim3 grid((nx + block.x -1)/block.x, (ny + block.y -1)/block.y);
    printf("grid:(%d,%d,%d) block:(%d,%d,%d)\n", 
           grid.x, grid.y, grid.z,
           block.x, block.y, block.z);
    
    QueryPerformanceCounter(&t1);
    sumMatrixOnGPU22D<<<grid, block>>>( (int *)d_A, (int *)d_B, (int *)d_C, nx, ny);
    cudaDeviceSynchronize();
    QueryPerformanceCounter(&t2);
    elapsedTime = (double)(t2.QuadPart - t1.QuadPart) / frequency.QuadPart;
    printf("GPU Execution Time on block(%d,%d) and grid(%d,%d): %f sec\n", block.x, block.y, grid.x, grid.y, elapsedTime);
    
    cudaMemcpy(gpuRef, d_C, size, cudaMemcpyDeviceToHost);
    printf("Verifying results of block(%d,%d) and grid(%d,%d)...\n", block.x, block.y, grid.x, grid.y);
    checkResult((float *)hostRef, (float *)gpuRef, nxy);

    dim3 block1(32, 16);
    dim3 grid1((nx + block1.x -1)/block1.x, (ny + block1.y -1)/block1.y);
    printf("grid:(%d,%d,%d) block:(%d,%d,%d)\n", 
           grid1.x, grid1.y, grid1.z,
           block1.x, block1.y, block1.z);
    
    QueryPerformanceCounter(&t1);
    sumMatrixOnGPU22D<<<grid1, block1>>>( (int *)d_A, (int *)d_B, (int *)d_C, nx, ny);
    cudaDeviceSynchronize();
    QueryPerformanceCounter(&t2);
    elapsedTime = (double)(t2.QuadPart - t1.QuadPart) / frequency.QuadPart;
    printf("GPU Execution Time on block(%d,%d) and grid(%d,%d): %f sec\n", block1.x, block1.y, grid1.x, grid1.y, elapsedTime);
    
    cudaMemcpy(gpuRef, d_C, size, cudaMemcpyDeviceToHost);
    printf("Verifying results of block(%d,%d) and grid(%d,%d)...\n", block1.x, block1.y, grid1.x, grid1.y);
    checkResult((float *)hostRef, (float *)gpuRef, nxy);

    dim3 block2(16, 16);
    dim3 grid2((nx + block2.x -1)/block2.x, (ny + block2.y -1)/block2.y);
    printf("grid:(%d,%d,%d) block:(%d,%d,%d)\n", 
           grid2.x, grid2.y, grid2.z,
           block2.x, block2.y, block2.z);
    
    QueryPerformanceCounter(&t1);
    sumMatrixOnGPU22D<<<grid2, block2>>>( (int *)d_A, (int *)d_B, (int *)d_C, nx, ny);
    cudaDeviceSynchronize();
    QueryPerformanceCounter(&t2);
    elapsedTime = (double)(t2.QuadPart - t1.QuadPart) / frequency.QuadPart;
    printf("GPU Execution Time on block(%d,%d) and grid(%d,%d): %f sec\n", block2.x, block2.y, grid2.x, grid2.y, elapsedTime);
    
    cudaMemcpy(gpuRef, d_C, size, cudaMemcpyDeviceToHost);
    printf("Verifying results of block(%d,%d) and grid(%d,%d)...\n", block2.x, block2.y, grid2.x, grid2.y);
    checkResult((float *)hostRef, (float *)gpuRef, nxy);

    printf("---------------------------------------\n");
    printf("Testing 1D block and 1D grid configurations:\n");
    dim3 block_1D(256);
    dim3 grid_1D((nx + block_1D.x -1)/block_1D.x);
    printf("grid:(%d,%d,%d) block:(%d,%d,%d)\n", 
           grid_1D.x, grid_1D.y, grid_1D.z,
           block_1D.x, block_1D.y, block_1D.z);
    QueryPerformanceCounter(&t1);
    sumMatrixOnGPU11D<<<grid_1D, block_1D>>>( (int *)d_A, (int *)d_B, (int *)d_C, nx, ny);
    cudaDeviceSynchronize();
    QueryPerformanceCounter(&t2);
    elapsedTime = (double)(t2.QuadPart - t1.QuadPart) / frequency.QuadPart;
    printf("GPU Execution Time on block(%d) and grid(%d): %f sec\n", block_1D.x, grid_1D.x, elapsedTime);
    cudaMemcpy(gpuRef, d_C, size, cudaMemcpyDeviceToHost);
    printf("Verifying results of block(%d) and grid(%d)...\n", block_1D.x, grid_1D.x);
    checkResult((float *)hostRef, (float *)gpuRef, nxy);

    printf("---------------------------------------\n");
    printf("Testing 1D block and 2D grid configurations:\n");
    dim3 block_1D_2D(256);
    dim3 grid_1D_2D((nx + block_1D_2D.x -1)/block_1D_2D.x, ny);
    printf("grid:(%d,%d,%d) block:(%d,%d,%d)\n", 
           grid_1D_2D.x, grid_1D_2D.y, grid_1D_2D.z,
           block_1D_2D.x, block_1D_2D.y, block_1D_2D.z);
    QueryPerformanceCounter(&t1);
    sumMatrixOnGPU21D<<<grid_1D_2D, block_1D_2D>>>( (int *)d_A, (int *)d_B, (int *)d_C, nx, ny);
    cudaDeviceSynchronize();
    QueryPerformanceCounter(&t2);
    elapsedTime = (double)(t2.QuadPart - t1.QuadPart) / frequency.QuadPart;
    printf("GPU Execution Time on block(%d) and grid(%d,%d): %f sec\n", block_1D_2D.x, grid_1D_2D.x, grid_1D_2D.y, elapsedTime);
    cudaMemcpy(gpuRef, d_C, size, cudaMemcpyDeviceToHost);
    printf("Verifying results of block(%d) and grid(%d,%d)...\n", block_1D_2D.x, grid_1D_2D.x, grid_1D_2D.y);
    checkResult((float *)hostRef, (float *)gpuRef, nxy);

    printf("---------------------------------------\n");
    printf("All configs tested!\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(hostRef);
    free(gpuRef);
    free(h_A);
    free(h_B);

    cudaDeviceReset();
    
    return 0;
}