#pragma once
#include "linear.h"
#include "tensor.h"
#include <vector>
#include <fstream>

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

    void backward(const CPUTensor& grad, float lr){
        CPUTensor g2 = W2.backward(grad, lr);
        CPUTensor g1 = W1.backward(g2, lr);
        attn.Wo.backward(g1, lr);
        attn.Wv.backward(g1, lr);
        attn.Wq.backward(g1, lr);
        attn.Wk.backward(g1, lr);
    }
};

struct CPUMiniGPT {
    std::vector<CPULinear> layers;

    std::vector<CPUTransformerBlock> blocks;
    CPULinear Wout;
    
    CPUMiniGPT(int n_layers, int d_model, int vocab_size){
    for(int i=0;i<n_layers;i++)
        blocks.push_back(CPUTransformerBlock(d_model, d_model));
        Wout = CPULinear(d_model, vocab_size);
    }

    CPUTensor forward(const CPUTensor& x){
        CPUTensor out = x;
        for(auto &blk: blocks) out = blk.forward(out);
        return Wout.forward(out);
    }

    void backward(const CPUTensor& grad, float lr){
        CPUTensor grad_out = grad;
        Wout.backward(grad_out, lr);

        for(int i = blocks.size()-1; i >=0 ; i--){
            blocks[i].backward(grad_out, lr);
        }
    }

    void save(const std::string &filename){
        std::ofstream fout(filename,std::ios::binary);
        for(auto &layer: layers) layer.save(fout);
        fout.close();
    }

    void load(const std::string &filename){
        std::ifstream fin(filename,std::ios::binary);
        for(auto &layer: layers) layer.load(fin);
        fin.close();
    }
};
