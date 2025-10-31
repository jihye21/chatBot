#pragma once
#include "tensor.h"
#include <vector>
#include <cstdlib>
#include <fstream>
#include <cmath>

struct CPULinear {
    CPUTensor W,b;
    int in_dim,out_dim;
    CPUTensor x_cache;

    CPULinear(): W(0,0), b(0,0), in_dim(0), out_dim(0) {}

    CPULinear(int in_dim,int out_dim): W(in_dim,out_dim), b(1,out_dim), in_dim(in_dim), out_dim(out_dim){
        float scale = 0.02f;
        for(auto &v: W.data) v = ((float)rand()/RAND_MAX - 0.5f) * scale;
        for(auto &v: b.data) v = 0.0f;
    }

    CPUTensor forward(const CPUTensor& x){
        x_cache = x;
        CPUTensor y(x.rows,out_dim);
        for(int i=0;i<x.rows;i++){
            for(int j=0;j<out_dim;j++){
                float sum=0;
                for(int k=0;k<in_dim;k++)
                    sum += x.data[i*in_dim+k]*W.data[k*out_dim+j];
                y.data[i*out_dim+j] = sum + b.data[j];
                if(std::isnan(y.data[i*out_dim+j])) {
                    std::cout << "NaN detected in Linear forward!\n";
                    y.data[i*out_dim+j] = 0.0f;
                }
            }
        }
        return y;
    }

    CPUTensor backward(const CPUTensor& grad_out, float lr){
        CPUTensor dX(x_cache.rows, x_cache.cols);
        for(int n=0;n<x_cache.rows;n++){
            for(int i=0;i<in_dim;i++){
                float sum = 0.0f;
                for(int j=0;j<out_dim;j++){
                    sum += grad_out.data[n*out_dim+j]*W.data[i*out_dim+j];
                }
                dX.data[n*in_dim+i] = sum;
            }
        }

        for(int i=0;i<in_dim;i++){
            for(int j=0;j<out_dim;j++){
                float grad=0.0f;
                for(int n=0;n<x_cache.rows;n++)
                    grad += x_cache.data[n*in_dim+i]*grad_out.data[n*out_dim+j];
                W.data[i*out_dim+j] -= lr*grad;
            }
        }

        for(int j=0;j<out_dim;j++){
            float grad=0.0f;
            for(int n=0;n<x_cache.rows;n++)
                grad += grad_out.data[n*out_dim+j];
            b.data[j] -= lr*grad;
        }
        
        return dX;
    }

    void save(std::ofstream &fout){
        fout.write((char*)W.data.data(), W.data.size()*sizeof(float));
        fout.write((char*)b.data.data(), b.data.size()*sizeof(float));
    }

    void load(std::ifstream &fin){
        fin.read((char*)W.data.data(), W.data.size()*sizeof(float));
        fin.read((char*)b.data.data(), b.data.size()*sizeof(float));
    }
};
