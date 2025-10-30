#pragma once
#include <vector>

struct CPUTensor {
    std::vector<float> data;
    int rows, cols;

    // 기본 생성자
    CPUTensor(): rows(0), cols(0), data() {}

    // 기존 생성자
    CPUTensor(int r,int c): rows(r), cols(c), data(r*c,0.0f) {}
};
