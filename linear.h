#pragma once
#include "tensor.h"
#include <vector>
#include <cstdlib>
#include <cuda_runtime.h>

// matmul + bias forward
__global__ void matmul_add_bias(float* A,float* B,float* C,float* b,int M,int K,int N){
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<M && col<N){
        float sum=0;
        for(int k=0;k<K;k++) sum+=A[row*K+k]*B[k*N+col];
        C[row*N+col]=sum+b[col];
    }
}

// Gradient kernels
__global__ void matmul_grad_w(float* x,float* grad_out,float* dW,int M,int K,int N){
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<K && col<N){
        float sum=0;
        for(int i=0;i<M;i++) sum+=x[i*K+row]*grad_out[i*N+col];
        dW[row*N+col]=sum;
    }
}
__global__ void matmul_grad_x(float* grad_out,float* W,float* dx,int M,int K,int N){
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<M && col<K){
        float sum=0;
        for(int i=0;i<N;i++) sum+=grad_out[row*N+i]*W[col*N+i];
        dx[row*K+col]=sum;
    }
}

struct GPULinear {
    GPUTensor W,b;
    int in_dim,out_dim;
    GPUTensor x_cache;

    GPULinear(int in_dim,int out_dim): W(in_dim,out_dim), b(1,out_dim), in_dim(in_dim), out_dim(out_dim){
        std::vector<float> hW(in_dim*out_dim), hb(out_dim,0.0f);
        for(auto &v:hW) v=((float)rand()/RAND_MAX-0.5f)*0.02f;
        cudaMemcpy(W.data,hW.data(),sizeof(float)*hW.size(),cudaMemcpyHostToDevice);
        cudaMemcpy(b.data,hb.data(),sizeof(float)*hb.size(),cudaMemcpyHostToDevice);
    }

    GPUTensor forward(const GPUTensor& x){
        x_cache = x;
        GPUTensor y(x.rows,out_dim);
        dim3 threads(16,16);
        dim3 blocks((out_dim+15)/16,(x.rows+15)/16);
        matmul_add_bias<<<blocks,threads>>>(x.data,W.data,y.data,b.data,x.rows,in_dim,out_dim);
        return y;
    }

    GPUTensor backward(GPUTensor& grad_out, float lr){
        GPUTensor dX(x_cache.rows, x_cache.cols);
        GPUTensor dW(W.rows,W.cols);
        dim3 threads(16,16);
        dim3 blocksW((W.cols+15)/16,(W.rows+15)/16);
        matmul_grad_w<<<blocksW,threads>>>(x_cache.data,grad_out.data,dW.data,x_cache.rows,x_cache.cols,W.cols);
        dim3 blocksX((x_cache.cols+15)/16,(x_cache.rows+15)/16);
        matmul_grad_x<<<blocksX,threads>>>(grad_out.data,W.data,dX.data,x_cache.rows,x_cache.cols,W.cols);
        return dX;
    }
};
