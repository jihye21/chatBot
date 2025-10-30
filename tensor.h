#pragma once
#include <vector>
#include <random>
#include <fstream>
#include <iostream>

struct CPUTensor {
    int rows, cols;
    std::vector<float> data;

    CPUTensor(): rows(0), cols(0) {}
    CPUTensor(int r, int c): rows(r), cols(c), data((size_t)r*c, 0.0f) {}
};

struct Embedding {
    int vocab_size;
    int d_model;
    std::vector<float> W;

    Embedding() : vocab_size(0), d_model(0) {}
    Embedding(int vocab, int dim): vocab_size(vocab), d_model(dim), W(vocab*dim){
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-0.1f,0.1f);
        for(auto &v: W) v = dist(gen);
    }

    CPUTensor forward(const std::vector<int>& tokens){
        if(tokens.empty()) return CPUTensor(0,0);

        CPUTensor out(tokens.size(), d_model);
        int vocab_size = (int)(W.size() / d_model);

        if(vocab_size == 0){
        std::cerr << "[ERROR] forward: W.size() is zero\n";
        return out;
        }

        for(int i=0;i<tokens.size();i++){
            int idx = tokens[i];
            for(int j=0;j<d_model;j++)
                out.data[i*d_model+j] = W[idx*d_model+j];
        }

        return out;
    }

    void save(const std::string &filename){
        std::ofstream fout(filename,std::ios::binary);
        fout.write((char*)W.data(), W.size()*sizeof(float));
        fout.close();
    }

    void load(const std::string &filename){
        std::ifstream fin(filename,std::ios::binary);
        fin.read((char*)W.data(), W.size()*sizeof(float));
        fin.close();
    }
};