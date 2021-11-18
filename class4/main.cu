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

#define BLOC_SIZE (16u)

#define FILTER_SIZE (5u)

#define TILE_SIZE (12u) //BLOCK_SIZE  - (FILTER_SIZE / 2) * 2


__global__ void sumVector(int *data1, int *data2, int *data3) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < VECT_SIZE) {
        data3[i] = data1[i] + data2[i];
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
}