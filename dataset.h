#pragma once
#include <vector>
#include <string>
#include <unordered_map>

extern std::vector<std::string> corpus;
extern std::unordered_map<std::string,int> word2idx;
extern std::vector<std::string> idx2word;

void build_vocab(const std::vector<std::string> &corpus);
std::vector<int> tokenize(const std::string& sentence);
std::string safe_idx2word(int idx);
void load_corpus(const std::string &filename);
void load_corpus_files(const std::vector<std::string> &files);
void load_all_corpus_from_folder(const std::string &folder_path);