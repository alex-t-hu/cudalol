#include <iostream>
#include <iomanip>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "utils.hpp"
#include "kernels/kernels.cu"

float* runSGEMMAndGetResult(float *A, float *B, float *C, int M, int N, int K, float alpha, float beta, cublasHandle_t handle){
    
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M*K*sizeof(float));
    cudaMalloc((void**)&d_B, K*N*sizeof(float));
    cudaMalloc((void**)&d_C, M*N*sizeof(float));
    
    cudaMemcpy(d_A, A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, M*N*sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float* C_ref = new float[M*N];

    for(int i=0;i<10;i++){
        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float time = 0;
        cudaEventElapsedTime(&time, start, stop);
        std::cout << time << " ";
        if(i==0){
            cudaMemcpy(C_ref, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
        }
    }
    std::cout << "\n";
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return C_ref;
}


int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    int M = 5;
    int N = 4;
    int K = 3;
    float alpha = 1.0;
    float beta = 0.1;

    float* A = (float*)malloc(M*K*sizeof(float));
    float* B = (float*)malloc(K*N*sizeof(float));
    float* C = (float*)malloc(M*N*sizeof(float));

    const int seed = 42;
    std::mt19937 gen(seed);
    initializeMatrix(A, M, K, gen);
    initializeMatrix(B, K, N, gen);
    initializeMatrix(C, M, N, gen);

    dim3 blockDim(16, 16);
    dim3 gridDim((M+blockDim.x-1)/blockDim.x, (N+blockDim.y-1)/blockDim.y);

    std::cout << std::setprecision(4) << std::fixed;
    float* C_1 = runKernelAndGetResult(sgemm1, A, B, C, M, N, K, alpha, beta, gridDim, blockDim);
    printMatrix(C_1, M, N);

    float* C_ref = runSGEMMAndGetResult(A, B, C, M, N, K, alpha, beta, handle);
    printMatrix(C_ref, M, N);
    std::cout << verifyMatrix(C_1, C_ref, M, N) << "\n";

    cublasDestroy(handle);
    free(A);
    free(B);
    free(C);

    return 0;
}
