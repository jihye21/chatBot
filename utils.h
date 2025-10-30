#pragma once
#include <vector>
#include <cmath>
#include <random>

void cross_entropy_backward_cpu(const std::vector<float>& logits,const std::vector<int>& targets,std::vector<float>& grad,int batch_size,int vocab_size){
    grad.resize(batch_size*vocab_size);
    for(int i=0;i<batch_size;i++){
        float maxv=-1e9;
        for(int j=0;j<vocab_size;j++) if(logits[i*vocab_size+j]>maxv) maxv=logits[i*vocab_size+j];
        float sum=0;
        for(int j=0;j<vocab_size;j++) sum+=exp(logits[i*vocab_size+j]-maxv);
        for(int j=0;j<vocab_size;j++){
            float prob=exp(logits[i*vocab_size+j]-maxv)/sum;
            grad[i*vocab_size+j]=prob - (j==targets[i]?1.0f:0.0f);
        }
    }
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

std::vector<int> generate_text_cpu(CPUMiniGPT& model,const std::vector<int>& seed,int max_len){
    std::vector<int> output=seed;
    for(int t=0;t<max_len;t++){
        CPUTensor x(seed.size(),model.Wout.W.cols);
        CPUTensor logits = model.forward(x);
        int next_idx = sample_from_logits(logits.data);
        output.push_back(next_idx);
    }
    return output;
}
