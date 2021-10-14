#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "/usr/local/cuda/include/cuda_runtime.h"

#define CUDA_CHECK_RETURN( value ) {                            \
    cudaError_t err = value;                                    \
    if(err != cudaSuccess) {                                    \
        fprintf(stderr, "Error %s at line %d in file %s \n",    \
            cudaGetErrorString(err), __LINE__, __FILE__);       \
        exit(1);                                                \
    }                                                           \
}

using namespace std;

int main(){
    int devCount;

    CUDA_CHECK_RETURN(cudaGetDeviceCount(&devCount));

    cout << "Available " << devCount << " devices" << endl;

    cudaDeviceProp properties;

    for(int i = 0; i < devCount; i++) {
        CUDA_CHECK_RETURN(cudaGetDeviceProperties(&properties, i));
        cout << "Device " << i + 1 << " name: " << properties.name << endl;
        cout << "Compute compatibility: " << properties.major << "." << properties.minor << endl; 
        cout << "Grid size: " << properties.maxGridSize[0]
                << ", " << properties.maxGridSize[1]
                << ", " << properties.maxGridSize[2];
    }
 
}