#include "dataset.h"
#include "transformer.h"
#include "linear.h"
#include "utils.h"
#include <iostream>

int main(){
    build_vocab(corpus);
    std::vector<std::vector<int>> tokenized;
    for(auto &s: corpus) tokenized.push_back(tokenize(s));

    CPUMiniGPT model(16,32,2,idx2word.size());
    float lr=0.01f;

    for(int epoch=0; epoch<10; epoch++){
        for(auto &tokens: tokenized){
            CPUTensor x(tokens.size(),16);
            CPUTensor logits = model.forward(x);

            std::vector<float> grad_logits;
            cross_entropy_backward_cpu(logits.data,tokens,grad_logits,logits.rows,logits.cols);

            CPUTensor grad_tensor(logits.rows, logits.cols);
            grad_tensor.data=grad_logits;
            model.Wout.backward(grad_tensor, lr);
        }
        std::cout<<"Epoch "<<epoch<<" done"<<std::endl;
    }

    std::vector<int> seed = tokenize("hello");
    std::vector<int> gen = generate_text_cpu(model, seed,5);
    for(int idx: gen) std::cout<<idx2word[idx]<<" ";
    std::cout<<std::endl;
}
