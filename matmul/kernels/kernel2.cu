#pragma once

#include <cuda_runtime.h>

__global__ void sgemm2(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta){
    
    int row = (threadIdx.x / 32) + blockIdx.x * 32;
    int col = (threadIdx.x % 32) + blockIdx.y * 32;
    if(row<M && col<N){
        float value = 0;
        for(int i=0;i<K;i++){
            value += A[row*K+i] * B[i*N+col];
        }
        C[row*N+col] = alpha * value + beta * C[row*N+col];
    }
}