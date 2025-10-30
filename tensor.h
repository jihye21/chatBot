#pragma once
#include <cuda_runtime.h>

struct GPUTensor {
    float* data;
    int rows, cols;
    GPUTensor(int r,int c): rows(r), cols(c){
        cudaMalloc(&data,sizeof(float)*r*c);
    }
    ~GPUTensor(){ cudaFree(data); }
};