#include "stdio.h"

__global__ void hello(void){
    printf("hello!\n");
}
void main(){
    hello<<<1,10>>>();
    cudaDeviceSynchronize();
}