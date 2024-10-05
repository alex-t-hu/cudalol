#pragma once

#define CHECK_CUDA(call)                                                     \
    {                                                                        \
        const cudaError_t error = call;                                      \
        if (error != cudaSuccess)                                            \
        {                                                                    \
            std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", ";   \
            std::cerr << "code: " << error << ", reason: " << cudaGetErrorString(error) << std::endl; \
            exit(1);                                                         \
        }                                                                    \
    }

bool verifyMatrix(float *A1, float *A2, int M, int N);

void printMatrix(float* A, int M, int N);

void initializeMatrix(float* A, int M, int N, std::mt19937& gen);