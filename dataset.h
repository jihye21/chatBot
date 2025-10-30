#pragma once
#include <vector>
#include <string>
#include <unordered_map>

std::vector<std::string> corpus = {
    "hello world",
    "this is a test",
    "mini gpt is fun",
    "we are learning c plus plus"
};

std::unordered_map<std::string,int> word2idx;
std::vector<std::string> idx2word;

void build_vocab(const std::vector<std::string>& texts){
    int idx=0;
    for(auto &sentence: texts){
        std::string word;
        for(char c:sentence){
            if(c==' '){
                if(!word.empty() && word2idx.find(word)==word2idx.end()){
                    word2idx[word]=idx++;
                    idx2word.push_back(word);
                }
                word.clear();
            } else word+=c;
        }
        if(!word.empty() && word2idx.find(word)==word2idx.end()){
            word2idx[word]=idx++;
            idx2word.push_back(word);
        }
    }
}

std::vector<int> tokenize(const std::string& sentence){
    std::vector<int> tokens;
    std::string word;
    for(char c:sentence){
        if(c==' '){
            if(!word.empty()) tokens.push_back(word2idx[word]);
            word.clear();
        } else word+=c;
    }
    if(!word.empty()) tokens.push_back(word2idx[word]);
    return tokens;
}
