#pragma once

#include <cuda_runtime.h>

__global__ void sgemm1(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta){
    int row = threadIdx.x + blockIdx.x * blockDim.x;
    int col = threadIdx.y + blockIdx.y * blockDim.y;
    if(row<M && col<N){
        float value = 0;
        for(int i=0;i<K;i++){
            value += A[row*K+i] * B[i*N+col];
        }
        C[row*N+col] = alpha * value + beta * C[row*N+col];
    }
}