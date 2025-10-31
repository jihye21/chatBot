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

std::vector<int> generate_response(const std::vector<int>& seed, CPUMiniGPT &model, Embedding &embed, int max_len=20) {
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

    std::cout << "Total corpus size: " << corpus.size() << std::endl;

    // Build vocab & tokenize corpus
    build_vocab(corpus);
    int pad_token_id = word2idx["<PAD> "];

    int vocab_size = (int)idx2word.size();
    std::cout << "Vocabulary size: " << vocab_size << std::endl;

    std::vector<std::vector<int>> tokenized;
    for(auto &s: corpus) tokenized.push_back(tokenize(s));

    int min_len = 9999, max_len = 0;
    for (auto &t : tokenized) {
        int len = t.size();
        if (len < min_len) min_len = len;
        if (len > max_len) max_len = len;
    }
    std::cout << "Token length range: " << min_len << " ~ " << max_len << std::endl;

    CPUMiniGPT model(4,6,vocab_size);
    Embedding embed(vocab_size, 64);

    std::ifstream fchk("embed.bin", std::ios::binary);
    if(fchk.good()){
        fchk.close();
        embed.load("embed.bin");
        model.load("weights.bin");
        std::cout << "가중치 로드 완료\n";
    }

    float lr=0.005f;
    int epochs=50;
    int seq_len = 3;

    int steps = 0;

    // Training loop
    for(int epoch=0; epoch<epochs; epoch++){
        steps = 0;

        for(size_t sample_idx=0; sample_idx<tokenized.size(); sample_idx++){
            auto &tokens = tokenized[sample_idx];

            if (tokens.empty()) continue;

            for(size_t start=0; start + seq_len <= tokens.size(); start++){
                std::vector<int> input_seq(tokens.begin() + start, tokens.begin() + start + seq_len);
                std::vector<int> target_seq(tokens.begin() + start + 1, tokens.begin() + start + seq_len + 1);
                
                while (input_seq.size() < seq_len) input_seq.push_back(pad_token_id);
                while (target_seq.size() < seq_len) target_seq.push_back(pad_token_id);

                bool valid = true;
                for(auto idx: input_seq){
                    if(idx < 0 || idx >= vocab_size){
                        valid = false;
                        break;
                    }
                }
                if(!valid) continue;
                
                CPUTensor x = embed.forward(input_seq);
                if (x.data.empty()) continue;

                CPUTensor logits = model.forward(x);
                if (logits.data.empty()) continue;

                std::vector<float> grad_logits;
                cross_entropy_backward_cpu(logits.data, target_seq, grad_logits, logits.rows, logits.cols);

                CPUTensor grad_tensor(logits.rows, logits.cols);
                grad_tensor.data = grad_logits;
                model.backward(grad_tensor, lr);

                steps++;
            }
        }
        std::cout << "Epoch " << epoch << " done, steps: " << steps << std::endl;
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
        
        std::vector<int> response_tokens = generate_response(seed, model, embed, 20);

        std::cout << "Bot: ";
        for(int idx: response_tokens){
            if (idx == pad_token_id) continue;
            if(idx >= 0 && idx < (int)idx2word.size())
                std::cout << idx2word[idx] << " ";
            else
                std::cout << "<UNK> ";
        }
        std::cout << std::endl;
    }

    return 0;
}
