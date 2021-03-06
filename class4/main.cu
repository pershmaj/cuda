#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "utils/pngio.h"

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

#define BLOCK_SIZE (16u)

#define FILTER_SIZE (5u)

#define TILE_SIZE (12u) //BLOCK_SIZE  - (FILTER_SIZE / 2) * 2


__global__ void imageProcessing(unsigned char *out, unsigned char *in, unsigned int pitch, unsigned int width, unsigned int height ) {
    int x_o = TILE_SIZE * blockDim.x + threadIdx.x;
    int y_o = TILE_SIZE * blockDim.y + threadIdx.y;

    int x_i = x_o - 2;
    int y_i = y_o - 2;

    __shared__ unsigned char sBuffer[BLOC_SIZE][BLOC_SIZE];

    if(x_i  >= 0 && x_i < width && y_i >= 0 && y_i < height) {
        sBuffer[threadIdx.y][threadIdx.x] = in[y_i * pitch + x_i];
    } else {
        sBuffer[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();
    int sum = 0;
   if(threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE) {
       for(int r = 0; r < FILTER_SIZE; r++) {
           for(int c = 0; c < FILTER_SIZE; c++) {
               sum += sBuffer[threadIdx.y + r][threadIdx.x + c];
           }
       }
       sum /= FILTER_SIZE * FILTER_SIZE;
       if(x_o < width && y_o  < height) {
           out[y_o * width + x_o] = sum;    

       }
   } 
}

int main(){

    png::image<png::rgb_pixel> img("lena.png");
    unsigned int width = img.get_width();
    unsigned int height = img.get_height();

    int size = width*height*sizeof(unsigned char);

    unsigned char *h_r = new unsigned char [size];
    unsigned char *h_g = new unsigned char [size];
    unsigned char *h_b = new unsigned char [size];

    unsigned char *h_r_n = new unsigned char [size];
    unsigned char *h_g_n = new unsigned char [size];
    unsigned char *h_b_n = new unsigned char [size];

    pvg::pngToRgb3(h_r, h_g, h_b, img);

    unsigned char *d_r = NULL;
    unsigned char *d_g = NULL;
    unsigned char *d_b = NULL;

    //bursting image size
    size_t pitch_r = 0;
    size_t pitch_g = 0;
    size_t pitch_b = 0;

    unsigned char *d_r_n = NULL;
    unsigned char *d_g_n = NULL;
    unsigned char *d_b_n = NULL;

    CUDA_CHECK_RETURN(cudaMallocPitch(&d_r, &pitch_r, width, height));
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_g, &pitch_g, width, height));
    CUDA_CHECK_RETURN(cudaMallocPitch(&d_b, &pitch_b, width, height));

    CUDA_CHECK_RETURN(cudaMalloc(&d_r_n, size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_g_n, size));
    CUDA_CHECK_RETURN(cudaMalloc(&d_b_n, size));

    CUDA_CHECK_RETURN(cudaMemcpy2D(d_r, pitch_r, h_r, width, width, height, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy2D(d_g, pitch_g, h_g, width, width, height, cudaMemcpyHostToDevice));
    CUDA_CHECK_RETURN(cudaMemcpy2D(d_b, pitch_b, h_b, width, width, height, cudaMemcpyHostToDevice));

    dim3 gridSize((width + TILE_SIZE - 1)/TILE_SIZE, (height + TILE_SIZE - 1)/TILE_SIZE);

    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);

    imageProcessing<<<gridSize, blockSize>>>(d_r_n, d_r, pitch_r, width, height);
    imageProcessing<<<gridSize, blockSize>>>(d_g_n, d_g, pitch_g, width, height);
    imageProcessing<<<gridSize, blockSize>>>(d_b_n, d_b, pitch_b, width, height);

    CUDA_CHECK_RETURN(cudaMemcpy(h_r_n, d_r_n, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(h_g_n, d_g_n, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK_RETURN(cudaMemcpy(h_b_n, d_b_n, size, cudaMemcpyDeviceToHost));


    pvg::rgb3ToPng(img, h_r_n, h_g_n, h_b_n);
    img.write("../lenaBlured.png");

    delete [] h_r;
    delete [] h_g;
    delete [] h_b;

    delete [] h_r_n;
    delete [] h_b_n;
    delete [] h_g_n;


}