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

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaMemcpy(C_ref, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaDeviceSynchronize();

    for(int i=0;i<5;i++){
        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float time = 0;
        cudaEventElapsedTime(&time, start, stop);
        std::cout << time << " ";
    }
    std::cout << "\n";
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return C_ref;
}


int main(int argc, char* argv[]) {
    printCUDAInfo();
    cublasHandle_t handle;
    cublasCreate(&handle);

    int M = 4096;
    int N = 4096;
    int K = 4096;
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


    std::cout << std::setprecision(4) << std::fixed;

    float*C_1;
    if(argc == 1 || atoi(argv[1])==1){
        dim3 blockDim(16, 32);
        dim3 gridDim((M+blockDim.x-1)/blockDim.x, (N+blockDim.y-1)/blockDim.y);
        C_1 = runKernelAndGetResult(sgemm1, A, B, C, M, N, K, alpha, beta, gridDim, blockDim);
    }else if(atoi(argv[1])==2){
        dim3 blockDim(32*32);
        dim3 gridDim((M+31)/32, (N+31)/32);
        C_1 = runKernelAndGetResult(sgemm2, A, B, C, M, N, K, alpha, beta, gridDim, blockDim);
    }else if(atoi(argv[1])==3){
        #define BLOCK_SIZE 16 // 16 better than 32 in this case
        dim3 blockDim(BLOCK_SIZE ,BLOCK_SIZE);
        dim3 gridDim((M+BLOCK_SIZE-1)/BLOCK_SIZE, (N+BLOCK_SIZE-1)/BLOCK_SIZE);
        C_1 = runKernelAndGetResult(sgemm3<BLOCK_SIZE>, A, B, C, M, N, K, alpha, beta, gridDim, blockDim);
    }

    float* C_ref = runSGEMMAndGetResult(A, B, C, M, N, K, alpha, beta, handle);
    std::cout << verifyMatrix(C_1, C_ref, M, N, 0.01) << "\n";

    cublasDestroy(handle);
    free(A);
    free(B);
    free(C);

    return 0;
}
