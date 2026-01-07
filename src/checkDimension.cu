#include <cuda_runtime.h>
#include <stdio.h>  
__global__ void checkIndex(void){
    printf("ThreadIdx:(%d %d %d) BlockIdx:(%d %d %d) BlockDim:(%d %d %d) GridDim:(%d %d %d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockIdx.x, blockIdx.y, blockIdx.z,
           blockDim.x, blockDim.y, blockDim.z,
           gridDim.x, gridDim.y, gridDim.z
    );
}
int main(int argc, char** argv){
    int nElem = 6;
    dim3 block(3);
    dim3 grid((nElem + block.x - 1) / block.x);
    printf("Grid:(%d %d %d) Block:(%d %d %d)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
    checkIndex<<<grid, block>>>();
    cudaDeviceSynchronize();
    return 0;
}   