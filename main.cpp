#include "dataset.h"
#include "tensor.h"
#include "linear.h"
#include "transformer.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <ctime>

std::vector<int> generate_response(const std::vector<int>& seed, CPUMiniGPT &model, Embedding &embed, int max_len=10) {
    std::vector<int> output_tokens = seed;

    for(int t=0; t<max_len; t++){
        CPUTensor x = embed.forward(output_tokens);
        if(x.data.empty()) break;

        CPUTensor logits = model.forward(x);
        if(logits.data.empty()) break;

        int last_idx = (int)(logits.rows - 1);
        int vocab_size = logits.cols;
        std::vector<float> last_logits(logits.data.begin() + last_idx * vocab_size,
                                       logits.data.begin() + (last_idx + 1) * vocab_size);

        for(int w : output_tokens) last_logits[w] -= 1e6;

        int next_token = sample_topk_temperature(last_logits, 10, 0.8f); 
        if(next_token < 0 || next_token >= (int)idx2word.size()) next_token = 0;

        output_tokens.push_back(next_token);
    }

    return std::vector<int>(output_tokens.begin() + seed.size(), output_tokens.end());
}

int main(){
    srand((unsigned int)time(0));

    load_all_corpus_from_folder("../corpus");

    // Build vocab & tokenize corpus
    build_vocab(corpus);
    int vocab_size = (int)idx2word.size();
    std::cout << "Vocabulary size: " << vocab_size << std::endl;

    CPUMiniGPT model(4,6,vocab_size);
    Embedding embed(vocab_size, 64);

    std::ifstream fchk("embed.bin", std::ios::binary);
    if(fchk.good()){
        fchk.close();
        embed.load("embed.bin");
        model.load("weights.bin");
        std::cout << "가중치 로드 완료\n";
    }

    std::vector<std::vector<int>> tokenized;
    for(auto &s: corpus) tokenized.push_back(tokenize(s));

    float lr=0.005f;
    int epochs=50;

    // Training loop
    for(int epoch=0; epoch<epochs; epoch++){

        int sample_idx = 0;

        for(auto &tokens: tokenized){

            CPUTensor x = embed.forward(tokens);
            CPUTensor logits = model.forward(x);

            std::vector<float> grad_logits;
            cross_entropy_backward_cpu(logits.data, tokens, grad_logits, logits.rows, logits.cols);

            CPUTensor grad_tensor(logits.rows, logits.cols);
            grad_tensor.data = grad_logits;
            model.backward(grad_tensor, lr);

            sample_idx++;
        }
        std::cout << "Epoch " << epoch << " done\n";
    }

    embed.save("embed.bin");
    model.save("weights.bin");
    std::ofstream fout("../corpus.txt");
    for(auto &line: corpus) fout << line << "\n";
    fout.close();
    std::cout << "가중치 저장 완료\n";

    // 상호작용
    std::string input;
    std::cout << "\n--- ChatBot ready! Type 'exit' to quit ---\n";
    while(true){
        std::cout << "You: ";
        std::getline(std::cin,input);
        if(input=="exit") break;

        std::vector<int> seed = tokenize(input);
        if(seed.empty()){ std::cout << "Bot: ...\n"; continue; }
        
        CPUTensor x = embed.forward(seed);
        if(x.data.empty()) {
            std::cout << "Bot: <UNK> x\n";
            continue;
        }
        
        CPUTensor logits = model.forward(x);
        
        if(logits.data.empty()) {
            std::cout << "Bot: <UNK> logits\n";
            continue;
        }
        
        std::vector<int> output_tokens;
        for(int t=0;t<10;t++){
            int idx = sample_topk_temperature(logits.data, 10, 1.1f);
            if(idx < 0 || idx >= (int)idx2word.size()) idx = 0;
                output_tokens.push_back(idx);
        }

        std::vector<int> response_tokens = generate_response(seed, model, embed, 10);

        std::cout << "Bot: ";
        for(int idx: output_tokens){
            if(idx >= 0 && idx < (int)idx2word.size())
                std::cout << idx2word[idx] << " ";
            else
                std::cout << "<UNK> ";
        }
        std::cout << std::endl;
    }

    return 0;
}
