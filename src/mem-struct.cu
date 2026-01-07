#include "stdio.h"
#include "string.h"
#include "time.h"
#include "stdlib.h"

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

void arraySumOnHost(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

void assignRandomValues(float *array, int N) {
    for (int i = 0; i < N; i++) {
        array[i] = (float) (rand() & 0xFF) / 10.0f;
    }
}

void experimentOnHost(){
    int N = 1024;
    size_t size = N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    assignRandomValues(h_A, N);
    assignRandomValues(h_B, N);

    arraySumOnHost(h_A, h_B, h_C, N);

    printf("First 10 elements of the result array C:\n");
    printf("%12s%12s%12s\n", "A[i]", "B[i]", "C[i]");
    for (int i = 0; i < 10; i++) {
        printf("%12.2f%12.2f%12.2f\n", h_A[i], h_B[i], h_C[i]);
    }

    free(h_A);
    free(h_B);
    free(h_C);
}

__global__ void addArraysOnDevice(float *A, float *B, float *C, const int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
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

void exprimentOnDevice(){
    int N = 1024;
    size_t size = N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *hostRef = (float *)malloc(size);
    float *gpuRef = (float *)malloc(size);
    
    assignRandomValues(h_A, N);
    assignRandomValues(h_B, N);

    arraySumOnHost(h_A, h_B, hostRef, N);

    float *d_A, *d_B, *d_C;

    cudaMalloc((float **) &d_A, size);
    cudaMalloc((float **) &d_B, size);
    cudaMalloc((float **) &d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int nElem = 32;
    dim3 block(nElem);
    dim3 grid((N + nElem -1) / nElem);

    addArraysOnDevice<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaMemcpy(gpuRef, d_C, size, cudaMemcpyDeviceToHost);
    printf("First 10 elements of the result array C from device:\n");
    printf("%12s%12s%12s\n", "A[i]", "B[i]", "gpuRef[i]");
    for (int i = 0; i < 10; i++) {
        printf("%12.2f%12.2f%12.2f\n", h_A[i], h_B[i], gpuRef[i]);
    }

    checkResult(hostRef, gpuRef, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
}

int main(){
    time_t t;
    srand((unsigned int)time(&t));

    int dev = 0;
    cudaSetDevice(dev);

    experimentOnHost();

    exprimentOnDevice();

    return 0;
}