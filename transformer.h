#pragma once
#include "linear.h"
#include "tensor.h"
#include <vector>

struct CPUAttention {
    CPULinear Wq,Wk,Wv,Wo;
    CPUAttention(int d_model): Wq(d_model,d_model), Wk(d_model,d_model),
                               Wv(d_model,d_model), Wo(d_model,d_model){}
    CPUTensor forward(const CPUTensor& x){ return Wo.forward(Wv.forward(x)); }
};

struct CPUTransformerBlock {
    CPUAttention attn;
    CPULinear W1,W2;
    CPUTransformerBlock(int d_model,int d_ff): attn(d_model), W1(d_model,d_ff), W2(d_ff,d_model){}
    CPUTensor forward(const CPUTensor& x){ return W2.forward(W1.forward(attn.forward(x))); }
};

struct CPUMiniGPT {
    std::vector<CPUTransformerBlock> blocks;
    CPULinear Wout;
    CPUMiniGPT(int d_model,int d_ff,int n_layers,int vocab_size): Wout(d_model,vocab_size){
        for(int i=0;i<n_layers;i++) blocks.emplace_back(CPUTransformerBlock(d_model,d_ff));
    }
    CPUTensor forward(const CPUTensor& x){
        CPUTensor out = x;
        for(auto &blk: blocks) out = blk.forward(out);
        return Wout.forward(out);
    }
};
