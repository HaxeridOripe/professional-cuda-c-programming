#include "cuda_runtime.h"
#include "stdio.h"

int main(){
    printf("Starting cudaDeviceProp example...\n");
    int devCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&devCount);
    if (error_id != cudaSuccess){
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }
    if(devCount == 0){
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", devCount);
    }
    int dev = 0, driverVersion = 0, runtimeVersion = 0;
    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("Device %d: \"%s\"\n", dev, deviceProp.name);
    printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
           driverVersion / 1000, (driverVersion % 100) / 10,
           runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
           static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f), (unsigned long long) deviceProp.totalGlobalMem);
    printf("  Memory Bus Width:                              %d-bit\n",
              deviceProp.memoryBusWidth);
    if (deviceProp.l2CacheSize){
        printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
    }
    printf("  Max Texture Dimension Size (x,y,z)            1D=(%d) 2D=(%d, %d) 3D=(%d, %d, %d)\n",
           deviceProp.maxTexture1D,
           deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
           deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
    printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, 2D=(%d, %d) x %d\n",
              deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
              deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);
    printf("  Total amount of constant memory:               %lu bytes\n",
           (unsigned long) deviceProp.totalConstMem);
    printf("  Total amount of shared memory per block:       %lu bytes\n",
           (unsigned long) deviceProp.sharedMemPerBlock);
    printf("  Total number of registers available per block: %d\n",
           deviceProp.regsPerBlock);
    printf("  Warp size:                                     %d\n",
           deviceProp.warpSize);
    printf("  Maximum number of threads per multiprocessor:  %d\n",
           deviceProp.maxThreadsPerMultiProcessor);
    printf("  Maximum number of threads per block:           %d\n",
           deviceProp.maxThreadsPerBlock);
    printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
           deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("  Maximum memory pitch:                          %lu bytes\n",
           (unsigned long) deviceProp.memPitch);
    printf("  Texture alignment:                             %lu bytes\n",
           (unsigned long) deviceProp.textureAlignment);
    printf("  Integrated GPU sharing Host Memory:            %s\n",
           (deviceProp.integrated ? "Yes" : "No"));
    printf("  Support host page-locked memory mapping:       %s\n",
           (deviceProp.canMapHostMemory ? "Yes" : "No"));
    printf("  Alignment requirement for Surfaces:            %s\n",
           (deviceProp.surfaceAlignment ? "Yes" : "No"));
    printf("  Device has ECC support:                        %s\n",
           (deviceProp.ECCEnabled ? "Yes" : "No"));
    printf("  Device supports Unified Addressing (UVA):      %s\n",
           (deviceProp.unifiedAddressing ? "Yes" : "No"));
    printf("  Device supports Compute Preemption:            %s\n",
           (deviceProp.computePreemptionSupported ? "Yes" : "No"));
    printf("  Supports Cooperative Kernel Launch:            %s\n",
              (deviceProp.cooperativeLaunch ? "Yes" : "No"));
    exit(EXIT_SUCCESS);

    return 0;

}