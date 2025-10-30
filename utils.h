#pragma once
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "transformer.h"

__global__ void cross_entropy_backward_kernel(float* logits,int* targets,float* grad,int batch_size,int vocab_size){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<batch_size*vocab_size){
        int token_idx = idx % vocab_size;
        int batch_idx = idx / vocab_size;
        float z = logits[idx];
        float exp_sum=0;
        for(int j=0;j<vocab_size;j++){
            exp_sum += exp(logits[batch_idx*vocab_size+j]);
        }
        float prob = exp(z)/exp_sum;
        grad[idx] = prob - (token_idx==targets[batch_idx]?1.0f:0.0f);
    }
}

void cross_entropy_backward(float* logits,int* targets,float* grad,int batch_size,int vocab_size){
    dim3 threads(256);
    dim3 blocks((batch_size*vocab_size+255)/256);
    cross_entropy_backward_kernel<<<blocks,threads>>>(logits,targets,grad,batch_size,vocab_size);
}

int sample_from_logits(const std::vector<float>& logits){
    std::vector<float> probs(logits.size());
    float maxv=-1e9;
    for(float v:logits) if(v>maxv) maxv=v;
    float sum=0;
    for(int i=0;i<logits.size();i++){
        probs[i]=exp(logits[i]-maxv);
        sum+=probs[i];
    }
    for(float &v:probs) v/=sum;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0,1.0);
    float r=dist(gen);
    float accum=0;
    for(int i=0;i<probs.size();i++){
        accum+=probs[i];
        if(r<=accum) return i;
    }
    return probs.size()-1;
}

std::vector<int> generate_text_gpu(GPUMiniGPT& model,const std::vector<int>& seed,int max_len){
    std::vector<int> output=seed;
    for(int t=0;t<max_len;t++){
        GPUTensor x(seed.size(),model.Wout.W.rows);
        GPUTensor logits = model.forward(x);
        std::vector<float> host_logits(model.Wout.out_dim);
        cudaMemcpy(host_logits.data(),logits.data+(logits.rows-1)*model.Wout.out_dim,
                   sizeof(float)*model.Wout.out_dim,cudaMemcpyDeviceToHost);
        int next_idx = sample_from_logits(host_logits);
        output.push_back(next_idx);
    }
    return output;
}
