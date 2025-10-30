#pragma once
#include "linear.h"
#include "tensor.h"
#include <vector>

struct GPUAttention {
    GPULinear Wq,Wk,Wv,Wo;
    GPUAttention(int d_model): Wq(d_model,d_model), Wk(d_model,d_model),
                               Wv(d_model,d_model), Wo(d_model,d_model){}
    GPUTensor forward(const GPUTensor& x){ return Wo.forward(Wv.forward(x)); }
};

struct GPUTransformerBlock {
    GPUAttention attn;
    GPULinear W1,W2;
    GPUTransformerBlock(int d_model,int d_ff): attn(d_model), W1(d_model,d_ff), W2(d_ff,d_model){}
    GPUTensor forward(const GPUTensor& x){ return W2.forward(W1.forward(attn.forward(x))); }
};

struct GPUMiniGPT {
    std::vector<GPUTransformerBlock> blocks;
    GPULinear Wout;
    GPUMiniGPT(int d_model,int d_ff,int n_layers,int vocab_size): Wout(d_model,vocab_size){
        for(int i=0;i<n_layers;i++) blocks.emplace_back(GPUTransformerBlock(d_model,d_ff));
    }
    GPUTensor forward(const GPUTensor& x){
        GPUTensor out = x;
        for(auto &blk: blocks) out = blk.forward(out);
        return Wout.forward(out);
    }
};
