#include <cuda_runtime.h>
#include <stdio.h>
#include <windows.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

double cpuSecond() {
    return (double)(1.0 * clock() / CLOCKS_PER_SEC);
}

void assignRandomValues(float *array, int N) {
    for (int i = 0; i < N; i++) {
        array[i] = (float) (rand() & 0xFF) / 100000.0f;
    }
}

void sumArrayOnHost(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}
__global__ void sumArraysOnDevice(float *A, float *B, float *C, const int N) {
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

int main(int argc, char **argv){
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    const int N = 1<<28;
    printf("Vector size: %d\n", N);
    size_t size = N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *hostRef = (float *)malloc(size);
    float *gpuRef = (float *)malloc(size);

    assignRandomValues(h_A, N);
    assignRandomValues(h_B, N);
    memset(hostRef, 0, size);
    memset(gpuRef, 0, size);

    LARGE_INTEGER frequency;        // 计时器频率
    LARGE_INTEGER t1, t2;           // 计数器值

    QueryPerformanceFrequency(&frequency);

    QueryPerformanceCounter(&t1);
    sumArrayOnHost(h_A, h_B, hostRef, N);
    QueryPerformanceCounter(&t2);
    double elapsedTime = (double)(t2.QuadPart - t1.QuadPart) / frequency.QuadPart;

    printf("Sum array on host elapsed %lf sec\n", elapsedTime);

    float *d_A, *d_B, *d_C;

    cudaMalloc((float **) &d_A, size);
    cudaMalloc((float **) &d_B, size);
    cudaMalloc((float **) &d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int nElem = 1024;
    dim3 block(nElem);
    dim3 grid((N + nElem -1) / nElem);
    QueryPerformanceCounter(&t1);
    sumArraysOnDevice<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    QueryPerformanceCounter(&t2);
    elapsedTime = (double)(t2.QuadPart - t1.QuadPart) / frequency.QuadPart;
    printf("Sum array on device elapsed %lf sec\n", elapsedTime);
    cudaMemcpy(gpuRef, d_C, size, cudaMemcpyDeviceToHost);
    checkResult(hostRef, gpuRef, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return 0;
}