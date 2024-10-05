#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "utils.hpp"

bool verifyMatrix(float *A1, float *A2, int M, int N){
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            if( abs(A1[i*N+j] - A2[i*N+j]) > 1e-5){
                return false;
            }
        }
    }
    return true;
}

void printMatrix(float* A, int M, int N){
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            std::cout << A[i*N+j] << " ";
        }
        std::cout << "\n";
    }
}

void initializeMatrix(float* A, int M, int N, std::mt19937& gen){
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            A[i*N+j] = dis(gen);
        }
    }
}