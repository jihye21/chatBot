#pragma once
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <algorithm>

int sample_topk_temperature(const std::vector<float>& logits, int k, float temperature=1.0f){
    int vocab_size = logits.size();
    std::vector<std::pair<float,int>> logit_idx;
    for(int i=0;i<vocab_size;i++) logit_idx.push_back({logits[i],i});
    std::sort(logit_idx.begin(), logit_idx.end(), [](auto &a, auto &b){return a.first>b.first;});

    std::vector<float> topk_probs;
    std::vector<int> topk_idx;
    float sum=0;
    for(int i=0;i<k && i<logit_idx.size();i++){
        float p = std::exp(logit_idx[i].first/temperature);
        topk_probs.push_back(p);
        topk_idx.push_back(logit_idx[i].second);
        sum+=p;
    }
    for(auto &p: topk_probs) p/=sum;

    float r = (float) rand()/RAND_MAX;
    float cum=0;
    for(int i=0;i<topk_probs.size();i++){
        cum+=topk_probs[i];
        if(r<=cum) return topk_idx[i];
    }
    return topk_idx.back();
}

void cross_entropy_backward_cpu(const std::vector<float>& logits,
                                const std::vector<int>& targets,
                                std::vector<float>& grad,
                                int rows, int cols,
                                int pad_token_id = -1) {
    grad.resize(logits.size(), 0.0f);

    for (int i = 0; i < rows; i++) {
        if (targets[i] == pad_token_id) continue;

        float max_logit = -1e20f;
        for (int j = 0; j < cols; j++)
            if (logits[i*cols + j] > max_logit) max_logit = logits[i*cols + j];

        float sum_exp = 0.0f;
        for (int j = 0; j < cols; j++)
            sum_exp += std::exp(logits[i*cols + j] - max_logit);

        for (int j = 0; j < cols; j++) {
            float p = std::exp(logits[i*cols + j] - max_logit) / (sum_exp + 1e-8f);
            grad[i*cols + j] = p;
        }

        grad[i*cols + targets[i]] -= 1.0f;
    }

    int valid_count = 0;
    for (int i = 0; i < rows; i++)
        if (targets[i] != pad_token_id) valid_count++;
    if (valid_count > 0)
        for (auto &g : grad) g /= valid_count;
}

int sample_from_logits(const std::vector<float>& logits, float temperature=1.0f){
    std::vector<float> probs(logits.size());
    float sum=0.0f;
    for(size_t i=0;i<logits.size();i++){
        probs[i] = std::exp(logits[i]/temperature);
        sum += probs[i];
    }
    for(size_t i=0;i<probs.size();i++) probs[i] /= sum;

    float r = ((float) rand()/RAND_MAX);
    float cum=0.0f;
    for(size_t i=0;i<probs.size();i++){
        cum += probs[i];
        if(r <= cum) return i;
    }
    return logits.size()-1;
}

std::vector<int> generate_text_cpu(CPUMiniGPT& model,const std::vector<int>& seed,int max_len){
    std::vector<int> output;
    int vocab_size = model.Wout.W.cols;
    for(int t=0; t<max_len; t++){
        CPUTensor x(1, model.Wout.W.rows); 
        CPUTensor logits = model.forward(x);
        int next_idx = sample_from_logits(logits.data);
        output.push_back(next_idx);
    }
    return output;
}
