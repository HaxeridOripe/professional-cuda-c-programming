#include <cuda_runtime.h>
#include <stdio.h>

int main(int argc, char **argv){
    //define total data elements
    int nElem = 1024;
    dim3 block(1024);
    dim3 grid((nElem + block.x - 1) / block.x);
    printf("grid.x: %u  block.x: %u\n", grid.x, block.x);

    block.x = 512;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("grid.x: %u  block.x: %u\n", grid.x, block.x);

    block.x = 256;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("grid.x: %u  block.x: %u\n", grid.x, block.x);

    block.x = 128;
    grid.x = (nElem + block.x - 1) / block.x;
    printf("grid.x: %u  block.x: %u\n", grid.x, block.x);
    cudaDeviceReset();
    return 0;
}