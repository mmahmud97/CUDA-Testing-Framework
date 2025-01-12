/***************************************************************************
 * File: vector_add.cu
 * Description: Demonstrates basic CUDA vector addition.
 * 
 * Compile Command:
 *   nvcc -o vector_add vector_add.cu
 ***************************************************************************/

#include <iostream>
#include <cuda_runtime.h>

//------------------------------------------------------------------------------
// Constant definitions
//------------------------------------------------------------------------------
const int THREADS_PER_BLOCK = 256;
const int N = 1024;  // Vector size

//------------------------------------------------------------------------------
// CUDA error-checking macro
//------------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t error = call;                                       \
        if (error != cudaSuccess) {                                     \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__\
                      << " - " << cudaGetErrorString(error) << std::endl;\
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

//------------------------------------------------------------------------------
// Vector addition kernel
//------------------------------------------------------------------------------
__global__ void vectorAdd(const float* A, const float* B, float* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

//------------------------------------------------------------------------------
// Main function
//------------------------------------------------------------------------------
int main() {
    // Host vectors
    float *h_A, *h_B, *h_C;

    // Device vectors
    float *d_A, *d_B, *d_C;

    size_t size = N * sizeof(float);

    // Allocate host memory
    h_A = new float[N];
    h_B = new float[N];
    h_C = new float[N];

    // Initialize host vectors with sample data
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Calculate grid size
    int gridSize = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel
    vectorAdd<<<gridSize, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Print the first 10 results
    for (int i = 0; i < 10; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free host memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}

