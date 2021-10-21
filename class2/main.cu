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

#define VECT_SIZE (1024u)
#define BLOC_SIZE (128u)

__global__ void sumVector(int *data1, int *data2, int *data3) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < VECT_SIZE) {
        data3[i] = data1[i] + data2[i];
    }

}
__global__ void fillVector(int *data) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < VECT_SIZE) {
        data[i] = i+1;
    }

}

int main(){
    int *h_data = (int*) malloc(VECT_SIZE * sizeof(int));
    int *d_data1 = NULL;
    int *d_data2 = NULL;
    int *d_data3 = NULL;

    CUDA_CHECK_RETURN(cudaMalloc(&d_data1, VECT_SIZE * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_data2, VECT_SIZE * sizeof(int)));
    CUDA_CHECK_RETURN(cudaMalloc(&d_data3, VECT_SIZE * sizeof(int)));
    //kernel config

    int blockSize = BLOC_SIZE;
    int gridSize = (VECT_SIZE + BLOC_SIZE - 1) / BLOC_SIZE;

    
    //kernel execution
    fillVector<<<gridSize, blockSize>>>(d_data1);
    //await for kernel computation
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());//kernel execution
    fillVector<<<gridSize, blockSize>>>(d_data2);
    //await for kernel computation
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    //kernel execution
    sumVector<<<gridSize, blockSize>>>(d_data1, d_data2, d_data3);
    //await for kernel computation
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // copy data (works both ways)
    CUDA_CHECK_RETURN(cudaMemcpy(h_data, d_data3, VECT_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    for(int i = 0; i < VECT_SIZE; i++) {
        cout << h_data[i];
        if( i < VECT_SIZE) {
            cout << ", ";
        }
    }

    free(h_data);
    CUDA_CHECK_RETURN(cudaFree(d_data1));
    CUDA_CHECK_RETURN(cudaFree(d_data2));
    CUDA_CHECK_RETURN(cudaFree(d_data3));
}