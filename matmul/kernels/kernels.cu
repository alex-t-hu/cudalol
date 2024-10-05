#include <iostream>
#include "kernel1.cu"

template <typename KernelFunc>
float* runKernelAndGetResult(
    KernelFunc kernel,
    float *A, float *B, float *C, int M, int N, int K, float alpha, float beta, dim3 gridDim, dim3 blockDim){
    
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

    kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    cudaMemcpy(C_ref, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
    cudaDeviceSynchronize();

    for(int i=0;i<10;i++){
        cudaEventRecord(start);
        kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
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