#include "dataset.h"
#include "transformer.h"
#include "linear.h"
#include "utils.h"
#include <iostream>

int main(){
    build_vocab(corpus);
    std::vector<std::vector<int>> tokenized;
    for(auto &s: corpus) tokenized.push_back(tokenize(s));

    GPUMiniGPT model(16,32,2,idx2word.size());
    float lr=0.01f;

    for(int epoch=0; epoch<100; epoch++){
        for(auto &tokens: tokenized){
            GPUTensor x(tokens.size(),16);
            GPUTensor logits = model.forward(x);

            int* d_targets;
            cudaMalloc(&d_targets,sizeof(int)*tokens.size());
            cudaMemcpy(d_targets,tokens.data(),sizeof(int)*tokens.size(),cudaMemcpyHostToDevice);

            GPUTensor grad_logits(logits.rows, logits.cols);
            cross_entropy_backward(logits.data,d_targets,grad_logits.data,logits.rows,logits.cols);

            model.Wout.backward(grad_logits,lr);
            for(auto &blk: model.blocks){
                blk.W1.backward(grad_logits,lr);
                blk.W2.backward(grad_logits,lr);
            }

            cudaFree(d_targets);
        }
        if(epoch%10==0) std::cout<<"Epoch "<<epoch<<" done"<<std::endl;
    }

    std::vector<int> seed = tokenize("hello");
    std::vector<int> gen = generate_text_gpu(model,seed,5);
    for(int idx: gen) std::cout<<idx2word[idx]<<" ";
    std::cout<<std::endl;
}
