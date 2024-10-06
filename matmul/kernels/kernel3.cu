#pragma once

#include <cuda_runtime.h>

template <const int BLOCK_SIZE>
__global__ void sgemm3(const float* A, const float* B, float* C, int M, int N, int K, float alpha, float beta){
    const int di = threadIdx.y;
    const int dj = threadIdx.x;
    const int row = di + blockIdx.y * BLOCK_SIZE;
    const int col = dj + blockIdx.x * BLOCK_SIZE;
    float value = 0;
    A += row*K;
    B += col;
    if(row<M && col<N){
        __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];
        
        for(int k=0;k<K;k+=BLOCK_SIZE){
            // const int maxdk = min(BLOCK_SIZE, K-k); 25 instead of 21
            sA[di][dj] = A[dj];
            sB[di][dj] = B[di*N];

            __syncthreads();
            for(int dk=0;dk<BLOCK_SIZE;dk++){
                value += sA[di][dk] * sB[dk][dj];
            }
            A += BLOCK_SIZE;
            B += BLOCK_SIZE * N;
            __syncthreads();
        }
        C[row*N+col] = alpha * value + beta * C[row*N+col];
    }
}