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

#define VECT_SIZE (3u)
#define BLOCK_SIZE (128u)

__global__ void multiplyMatrixes(int *data1, int *data2, int *data3) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if(x < VECT_SIZE && y < VECT_SIZE) {
        int r = 0;
        for(int i = 0; i < VECT_SIZE; i++) {
            r+= data1[x] * data2[y]; 
        }
        data3[x * VECT_SIZE + y] = r;
    }

}

int main(){
    const int MatrixSize = VECT_SIZE * VECT_SIZE * sizeof(int);

    int *h_data1 = (int*) malloc(MatrixSize);
    int *h_data2 = (int*) malloc(MatrixSize);
    int *h_data3 = (int*) malloc(MatrixSize);
    int *d_data1 = NULL;
    int *d_data2 = NULL;
    int *d_data3 = NULL;

    CUDA_CHECK_RETURN(cudaMalloc(&d_data1, MatrixSize));
    CUDA_CHECK_RETURN(cudaMalloc(&d_data2, MatrixSize));
    CUDA_CHECK_RETURN(cudaMalloc(&d_data3, MatrixSize));
    //kernel config
    for (int i = 0; i < VECT_SIZE * VECT_SIZE; i++) {
        h_data1[i] = i+1;
        h_data2[i] = i+1;
    }

    CUDA_CHECK_RETURN( cudaMemcpy( d_data1, h_data1, MatrixSize, cudaMemcpyHostToDevice ) );
    CUDA_CHECK_RETURN( cudaMemcpy( d_data2, h_data2, MatrixSize, cudaMemcpyHostToDevice ) );

    dim3 blockSize( BLOCK_SIZE, BLOCK_SIZE );
	dim3 gridSize( (VECT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE,
				   (VECT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE );

    
    //kernel execution
    multiplyMatrixes<<<gridSize, blockSize>>>(d_data1, d_data2, d_data3);
    //await for kernel computation
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());
    // copy data (works both ways)
    CUDA_CHECK_RETURN(cudaMemcpy(h_data3, d_data3, MatrixSize, cudaMemcpyDeviceToHost));

    for(int i = 0; i < VECT_SIZE * VECT_SIZE; i++) {
        cout << h_data1[i];
        if( i < VECT_SIZE * VECT_SIZE) {
            cout << ", ";
        }
    }

    free(h_data1);
    free(h_data2);
    free(h_data3);
    CUDA_CHECK_RETURN(cudaFree(d_data1));
    CUDA_CHECK_RETURN(cudaFree(d_data2));
    CUDA_CHECK_RETURN(cudaFree(d_data3));
}