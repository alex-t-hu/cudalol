#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "utils.hpp"

bool verifyMatrix(float *A1, float *A2, int M, int N, float thresh){
    bool isSame = true;
    for(int i=0;i<M;i++){
        for(int j=0;j<N;j++){
            if( fabs(A1[i*N+j] - A2[i*N+j]) > thresh){ // 1e-5 produces errors
                isSame=false;
                std::cout << "A1: " << A1[i*N+j] << " A2: " << A2[i*N+j] << std::endl;
            }
        }
    }
    return isSame;
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

void printCUDAInfo(){
    cudaDeviceProp prop;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&prop, device);
    std::cout << "CUDA Device Info:" << std::endl;
    std::cout << "  Name: " << prop.name << std::endl;
    std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "  Shared Memory Per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "  Shared Memory per MP: " << prop.sharedMemPerMultiprocessor / 1024 << " KB" << std::endl;
    std::cout << "  Total Registers Per Block: " << prop.regsPerBlock << std::endl;
    std::cout << "  Max registers per MP: " << prop.regsPerMultiprocessor << std::endl;
    std::cout << "  Warp Size: " << prop.warpSize << std::endl;
    std::cout << "  Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "  Max Threads Per MP: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "  Max Threads Dimension (x, y, z): (" << prop.maxThreadsDim[0] << ", "
              << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << ")" << std::endl;
    std::cout << "  Max Grid Size (x, y, z): (" << prop.maxGridSize[0] << ", "
              << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << ")" << std::endl;
}