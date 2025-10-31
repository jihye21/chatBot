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
#include <cmath>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dis(0.0f, 1.0f);

float compute_step_loss(const std::vector<float>& logits, const std::vector<int>& targets,
                        int rows, int cols, int pad_token_id){
    float loss = 0.0f;
    int count = 0;
    for(int i=0;i<rows;i++){
        if(targets[i]==pad_token_id) continue;
        float max_logit = -1e20f;
        for(int j=0;j<cols;j++) if(logits[i*cols+j] > max_logit) max_logit = logits[i*cols+j];

        float sum_exp = 0.0f;
        for(int j=0;j<cols;j++) sum_exp += std::exp(logits[i*cols+j] - max_logit);

        float log_p = logits[i*cols + targets[i]] - max_logit - std::log(sum_exp + 1e-8f);
        loss -= log_p;
        count++;
    }
    if(count>0) loss /= count;
    return loss;
}

void xavier_init(std::vector<float>& weights, int in_dim, int out_dim){
    float bound = std::sqrt(6.0f / (in_dim + out_dim));
    for(auto &v: weights) v = ((float)rand()/RAND_MAX*2 - 1) * bound;
}

int sample_from_top_p(const std::vector<float>& logits, float temperature, float top_p) {
    std::vector<float> probs(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp((logits[i] - max_logit) / temperature);
        sum_exp += probs[i];
    }
    for (auto& p : probs) p /= sum_exp;

    std::vector<int> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&](int a, int b){ return probs[a] > probs[b]; });

    float cumulative = 0.0f;
    std::vector<int> candidates;
    for (int idx : indices) {
        cumulative += probs[idx];
        candidates.push_back(idx);
        if (cumulative >= top_p) break;
    }

    float subset_sum = 0.0f;
    for (int idx : candidates) subset_sum += probs[idx];
    for (int idx : candidates) probs[idx] /= subset_sum;

    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(gen);
    float accum = 0.0f;
    for (int idx : candidates) {
        accum += probs[idx];
        if (r <= accum) return idx;
    }
    return candidates.back();
}

int sample_topk_top_p_temperature(std::vector<float>& logits, int top_k, float top_p, float temperature) {
    int vocab_size = (int)logits.size();

    for(int i=0;i<vocab_size;i++) logits[i] /= temperature;

    std::vector<int> top_indices(vocab_size);
    std::iota(top_indices.begin(), top_indices.end(), 0);
    std::sort(top_indices.begin(), top_indices.end(), [&logits](int a, int b){ return logits[a] > logits[b]; });

    if(top_k > 0 && top_k < vocab_size) top_indices.resize(top_k);

    std::vector<float> exp_logits;
    float max_logit = -1e20f;
    for(int idx: top_indices) if(logits[idx] > max_logit) max_logit = logits[idx];

    for(int idx: top_indices) exp_logits.push_back(std::exp(logits[idx]-max_logit));

    float sum_exp = std::accumulate(exp_logits.begin(), exp_logits.end(), 0.0f);

    if(top_p < 1.0f) {
        std::vector<std::pair<int,float>> prob_pairs;
        for(size_t i=0;i<top_indices.size();i++) prob_pairs.emplace_back(top_indices[i], exp_logits[i]/sum_exp);
        std::sort(prob_pairs.begin(), prob_pairs.end(), [](auto &a, auto &b){ return a.second > b.second; });

        float cum_prob = 0.0f;
        std::vector<int> filtered_indices;
        std::vector<float> filtered_probs;
        for(auto &p : prob_pairs){
            cum_prob += p.second;
            filtered_indices.push_back(p.first);
            filtered_probs.push_back(p.second);
            if(cum_prob >= top_p) break;
        }
        top_indices = filtered_indices;
        exp_logits = filtered_probs;
        sum_exp = std::accumulate(exp_logits.begin(), exp_logits.end(), 0.0f);
    }

    for(auto &v : exp_logits) v /= sum_exp;

    float r = dis(gen);
    float accum = 0.0f;
    for(size_t i=0;i<exp_logits.size();i++){
        accum += exp_logits[i];
        if(r <= accum) return top_indices[i];
    }

    return top_indices.back();
}

std::vector<int> generate_response_context(
    const std::vector<int>& seed,
    CPUMiniGPT &model,
    Embedding &embed,
    int max_len = 60,
    int top_k = 15,
    float temperature = 0.74f,
    float top_p = 0.92f,
    float repetition_penalty = 1.4f,
    int pad_token_id = -1,
    int eos_token_id = -1,
    bool debug = false
) {
    std::vector<int> output_tokens = seed;
    std::vector<int> recent_tokens = seed;

    for (int t = 0; t < max_len; t++) {
        CPUTensor x = embed.forward(output_tokens);
        if (x.data.empty()) break;

        CPUTensor logits = model.forward(x);
        if (logits.data.empty()) break;

        int vocab_size = logits.cols;
        std::vector<float> last_logits(logits.data.begin(), logits.data.begin() + vocab_size);

        if (pad_token_id >= 0 && pad_token_id < vocab_size)
            last_logits[pad_token_id] = -1e6f;

        for (int token : recent_tokens) {
            if (token >= 0 && token < vocab_size)
                last_logits[token] /= repetition_penalty;
        }

        int next_token = sample_topk_top_p_temperature(last_logits, top_k, top_p, temperature);

        if (next_token == eos_token_id) break;

        output_tokens.push_back(next_token);
        recent_tokens.push_back(next_token);
        if (recent_tokens.size() > 60) recent_tokens.erase(recent_tokens.begin());

        if (debug) {
            std::cout << "Step " << t << ": next_token=" << next_token
                      << " (" << idx2word[next_token] << ")\n";
        }
    }

    return std::vector<int>(output_tokens.begin() + seed.size(), output_tokens.end());
}

std::vector<int> generate_response(const std::vector<int>& seed,
                                   CPUMiniGPT &model, Embedding &embed,
                                   int max_len=60, int top_k=10, float temperature=0.8f,
                                   float top_p=0.95f, float repetition_penalty=1.25f,
                                   int pad_token_id=-1, int eos_token_id=-1) {
    std::vector<int> output_tokens = seed;
    std::vector<int> recent_tokens;

    for (int t = 0; t < max_len; t++) {
        CPUTensor x = embed.forward(output_tokens);
        CPUTensor logits = model.forward(x);
        if (logits.data.empty()) break;

        int vocab_size = logits.cols;
        std::vector<float> last_logits(logits.data.begin(), logits.data.begin() + vocab_size);

        if (pad_token_id >= 0 && pad_token_id < vocab_size)
            last_logits[pad_token_id] = -1e6f;

        for (int token : recent_tokens)
            if (token >= 0 && token < vocab_size)
                last_logits[token] /= repetition_penalty;

        int next_token = sample_topk_top_p_temperature(last_logits, top_k, top_p, temperature);

        if(next_token < 0 || next_token >= vocab_size)
            next_token = 0;

        output_tokens.push_back(next_token);
        recent_tokens.push_back(next_token);

        if (recent_tokens.size() > 60) recent_tokens.erase(recent_tokens.begin());

        if (next_token == eos_token_id) break;
    }

    return std::vector<int>(output_tokens.begin() + seed.size(), output_tokens.end());
}

int main(){
    srand((unsigned int)time(0));

    load_all_corpus_from_folder("../corpus");
    std::cout << "Total corpus size: " << corpus.size() << std::endl;

    // vocab & tokenization
    corpus.push_back("<PAD>");
    build_vocab(corpus);
    int pad_token_id = word2idx["<PAD>"];
    int vocab_size = (int)idx2word.size();
    std::cout << "Vocabulary size: " << vocab_size << std::endl;

    std::vector<std::vector<int>> tokenized;
    for(auto &s: corpus) tokenized.push_back(tokenize(s));

    int min_len = 9999, max_len = 0;
    for(auto &t : tokenized){
        int len = t.size();
        if(len < min_len) min_len = len;
        if(len > max_len) max_len = len;
    }
    std::cout << "Token length range: " << min_len << " ~ " << max_len << std::endl;

    CPUMiniGPT model(4, 6, vocab_size);
    Embedding embed(vocab_size, 64);
    embed.W.resize(vocab_size * 64);
    xavier_init(embed.W, vocab_size, 64);

    for(auto &layer: model.layers){
        layer.W.data.resize(layer.W.rows * layer.W.cols);
        layer.b.data.resize(layer.b.cols, 0.0f);
        xavier_init(layer.W.data, layer.W.rows, layer.W.cols);
    }

    std::ifstream fchk("embed.bin", std::ios::binary);
    if(fchk.good()){
        fchk.close();
        embed.load("embed.bin");
        model.load("weights.bin");
        std::cout << "가중치 로드 완료\n";
    }

    float lr = 0.0005f;
    int epochs = 50;
    int seq_len = 16;

    std::cout << "--- Debug: Embedding & Linear Init ---\n";
    std::cout << "Embedding sample (W[0~4]): ";
    for(int i=0;i<5;i++) std::cout << embed.W[i] << " ";
    std::cout << "\n";

    for(size_t l=0;l<model.layers.size();l++){
        std::cout << "Layer " << l << " W sample (0~4): ";
        for(int i=0;i<5;i++) std::cout << model.layers[l].W.data[i] << " ";
        std::cout << "\n";
        std::cout << "Layer " << l << " b sample (0~4): ";
        for(int i=0;i<5 && i<(int)model.layers[l].b.data.size();i++) std::cout << model.layers[l].b.data[i] << " ";
        std::cout << "\n";
    }

    // 학습
    for(int epoch=0; epoch<epochs; epoch++){
        int steps = 0;
        float epoch_loss = 0.0f;

        for(auto &tokens: tokenized){
            if(tokens.empty()) continue;

            int cur_seq_len = std::min(seq_len, (int)tokens.size());
            size_t max_start = (tokens.size() > seq_len) ? tokens.size() - seq_len : 1;

            for(size_t start=0; start<max_start; start++){
                size_t end_idx = std::min(start + seq_len, tokens.size());
                std::vector<int> input_seq(tokens.begin() + start, tokens.begin() + end_idx);
                std::vector<int> target_seq(tokens.begin() + start + 1,
                                            tokens.begin() + std::min(start + seq_len + 1, tokens.size()));

                while(input_seq.size() < seq_len) input_seq.push_back(pad_token_id);
                while(target_seq.size() < seq_len) target_seq.push_back(pad_token_id);

                bool valid = true;
                for(auto idx: input_seq){
                    if(idx < 0 || idx >= vocab_size){ valid = false; break; }
                }
                if(!valid) continue;

                CPUTensor x = embed.forward(input_seq);
                if(x.data.empty()) continue;

                if(x.data.empty()){
                    std::cout << "Embedding forward output is empty!\n";
                } else {
                    std::cout << "Embedding forward sample (0~4): ";
                    for(int i=0;i<5;i++) std::cout << x.data[i] << " ";
                    std::cout << "\n";
                }

                CPUTensor logits = model.forward(x);
                if(logits.data.empty()){
                    std::cout << "Model forward output is empty!\n";
                } else {
                    std::cout << "Logits sample (0~4): ";
                    for(int i=0;i<5;i++) std::cout << logits.data[i] << " ";
                    std::cout << "\n";
                }

                if(logits.data.empty()) continue;

                std::vector<float> grad_logits;
                cross_entropy_backward_cpu(logits.data, target_seq, grad_logits, logits.rows, logits.cols, pad_token_id);

                float step_loss = compute_step_loss(logits.data, target_seq, logits.rows, logits.cols, pad_token_id);
                epoch_loss += step_loss;

                // 디버깅
                if(steps < 5){
                    std::cout << "Step " << steps << " loss: " << step_loss
                              << ", grad_logits[0]: " << grad_logits[0] << "\n";
                }

                CPUTensor grad_tensor(logits.rows, logits.cols);
                grad_tensor.data = grad_logits;
                model.backward(grad_tensor, lr);

                steps++;
            }
        }

        std::cout << "Epoch " << epoch << " done, steps: " << steps
                  << ", avg loss: " << (steps > 0 ? epoch_loss / steps : 0.0f) << "\n";
    }

    embed.save("embed.bin");
    model.save("weights.bin");
    std::ofstream fout("../corpus.txt");
    for(auto &line: corpus) fout << line << "\n";
    fout.close();
    std::cout << "가중치 저장 완료\n";

    std::string input;
    std::cout << "\n--- ChatBot ready! Type 'exit' to quit ---\n";
    while(true){
        std::cout << "You: ";
        std::getline(std::cin, input);
        if(input == "exit") break;

        std::vector<int> seed = tokenize(input);
        if(seed.empty()){ std::cout << "Bot: ...\n"; continue; }

        std::vector<int> response_tokens = generate_response_context(
            seed, model, embed,
            50,      // max_len
            15,      // top_k
            0.74,   // temperature
            0.92f,    // top_p
            1.4f,    // repetition_penalty
            pad_token_id,
            -1,      // eos_token_id
            false
        );

        std::cout << "Bot: ";
        for(int idx: response_tokens){
            if(idx == pad_token_id) continue;
            if(idx >= 0 && idx < (int)idx2word.size()) std::cout << idx2word[idx] << " ";
            else std::cout << "<UNK> ";
        }
        std::cout << "\n";
    }

    return 0;
}
