#include "dataset.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

std::vector<std::string> corpus;
std::unordered_map<std::string,int> word2idx;
std::vector<std::string> idx2word;

void load_corpus(const std::string &filename){
    corpus.clear();
    std::ifstream fin(filename);
    if(!fin){
        std::cerr << "Error: Cannot open " << filename << std::endl;
        return;
    }
    std::string line;
    while(std::getline(fin, line)){
        if(!line.empty())
            corpus.push_back(line);
    }
    fin.close();
}

void load_corpus_files(const std::vector<std::string> &files) {
    for (auto &filename : files) {
        std::ifstream fin(filename);
        if(!fin) {
            std::cerr << "Warning: Cannot open " << filename << std::endl;
            continue;
        }
        std::string line;
        while(std::getline(fin, line)) {
            if(!line.empty()) corpus.push_back(line);
        }
        fin.close();
    }
}

void load_all_corpus_from_folder(const std::string &folder_path) {
    corpus.clear();
    for (const auto &entry : fs::directory_iterator(folder_path)) {
        if (entry.path().extension() == ".txt") {
            std::ifstream fin(entry.path());
            if (!fin) {
                std::cerr << "Warning: Cannot open " << entry.path() << std::endl;
                continue;
            }
            std::string line;
            while (std::getline(fin, line)) {
                if (!line.empty())
                    corpus.push_back(line);
            }
            fin.close();
            std::cout << "Loaded " << entry.path().filename() << std::endl;
        }
    }
    std::cout << "Total corpus size: " << corpus.size() << std::endl;
}

void build_vocab(const std::vector<std::string> &corpus){
    int idx = 0;
    word2idx.clear();
    idx2word.clear();
    
    word2idx["<UNK>"] = idx++;
    idx2word.push_back("<UNK>");

    word2idx["<PAD>"] = idx++;
    idx2word.push_back("<PAD>");
    
    for(auto &sentence: corpus){
        std::string clean;
        for(char c: sentence){
            if(c != '.' && c != ',' && c != '!' && c != '?') clean += c;
        }

        size_t start=0, end=0;
        while((end = clean.find(' ', start)) != std::string::npos){
            std::string word = clean.substr(start, end-start);
            if(!word.empty() && word2idx.find(word) == word2idx.end()){
                word2idx[word] = idx++;
                idx2word.push_back(word);
            }
            start = end + 1;
        }

        std::string word = clean.substr(start);
        if(!word.empty() && word2idx.find(word) == word2idx.end()){
            word2idx[word] = idx++;
            idx2word.push_back(word);
        }
    }
}

std::vector<int> tokenize(const std::string& sentence){
    std::vector<int> tokens;
    if(word2idx.empty()) return tokens;

    std::string clean;
    for(char c: sentence){
        if(c != '.' && c != ',' && c != '!' && c != '?')
            clean += c;
    }

    size_t start = 0, end = 0;
    while((end = clean.find(' ', start)) != std::string::npos){
        std::string word = clean.substr(start, end - start);
        if(!word.empty()){
            auto it = word2idx.find(word);
            tokens.push_back(it != word2idx.end() ? it->second : word2idx["<UNK>"]);
        }
        start = end + 1;
    }

    std::string word = clean.substr(start);
    if(!word.empty()){
        auto it = word2idx.find(word);
        tokens.push_back(it != word2idx.end() ? it->second : word2idx["<UNK>"]);
    }
    return tokens;
}

inline std::string safe_idx2word(int idx){
    if(idx < 0 || idx >= static_cast<int>(idx2word.size())) return "<UNK>";
    return idx2word[idx];
}