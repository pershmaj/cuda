#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
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

#define VECT_SIZE (1048576u)
#define BLOCK_SIZE (16u)

// 32u -> 25 763ns
// 512u -> 278 967ns
// 1048576u -> 5369ns
// 134217728u -> 83ns
// 1073741824u -> 140ns


__global__ void multiplyMatrixes(int *A, int *B, int *C) {
    // int row = threadIdx.x + blockIdx.x * blockDim.x;
    // int col = threadIdx.y + blockIdx.y * blockDim.y;
    // if(row < VECT_SIZE && col < VECT_SIZE) {
    //     int r = 0;
    //     for(int i = 0; i < VECT_SIZE; i++) {
    //         r+= data1[row * VECT_SIZE + i] * data2[col + VECT_SIZE * i]; 
    //     }
    //     data3[row * VECT_SIZE + col] = r;
    // }

    int blockRow = blockIdx.x;
    int blockCol = blockIdx.y;
    int row = threadIdx.y;
    int col = threadIdx.x;
    int Cvalue = 0;


    int* Csub = &C[VECT_SIZE * BLOCK_SIZE * blockRow + BLOCK_SIZE * blockCol];

    for(int m = 0; m < VECT_SIZE / BLOCK_SIZE; m++) {
        const int* Asub = &A[VECT_SIZE * BLOCK_SIZE * blockRow + BLOCK_SIZE * m];
        const int* Bsub = &B[VECT_SIZE * BLOCK_SIZE * m + BLOCK_SIZE * blockCol];

        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col] = Asub[row * VECT_SIZE + col];
        Bs[row][col] = Bsub[row * VECT_SIZE + col];

        __syncthreads();

        for(int e = 0; e < BLOCK_SIZE; e++) {
            Cvalue += As[row][e] * Bs[e][row];
        }

        __syncthreads();
    }

    Csub[row * VECT_SIZE + col] = Cvalue;
}

int main(){
    const long MatrixSize = VECT_SIZE * VECT_SIZE * sizeof(int);

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
        h_data1[i] = random();//i+1;
        h_data2[i] = random();//i+1;
    }

    CUDA_CHECK_RETURN( cudaMemcpy( d_data1, h_data1, MatrixSize, cudaMemcpyHostToDevice ) );
    CUDA_CHECK_RETURN( cudaMemcpy( d_data2, h_data2, MatrixSize, cudaMemcpyHostToDevice ) );

    dim3 gridSize(ceilf(VECT_SIZE/(float)BLOCK_SIZE), ceilf(VECT_SIZE/(float)BLOCK_SIZE), 1);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    
    //kernel execution
    multiplyMatrixes<<<gridSize, blockSize>>>(d_data1, d_data2, d_data3);
    //await for kernel computation
    CUDA_CHECK_RETURN(cudaDeviceSynchronize());

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
    std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << "[ns]" << std::endl;
    // copy data (works both ways)
    CUDA_CHECK_RETURN(cudaMemcpy(h_data3, d_data3, MatrixSize, cudaMemcpyDeviceToHost));

    // for(int i = 0; i < VECT_SIZE * VECT_SIZE; i++) {
        // cout << h_data3[i];
        // cout << "\t";
        // if(i % VECT_SIZE - 1 == 0) cout << endl;
    // }

    free(h_data1);
    free(h_data2);
    free(h_data3);
    CUDA_CHECK_RETURN(cudaFree(d_data1));
    CUDA_CHECK_RETURN(cudaFree(d_data2));
    CUDA_CHECK_RETURN(cudaFree(d_data3));
}