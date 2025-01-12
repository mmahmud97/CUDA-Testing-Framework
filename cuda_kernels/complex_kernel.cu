/***************************************************************************
 * File: complex_kernel.cu
 * Description: Demonstrates element-wise transformations and parallel reduction.
 * 
 * Compile Command:
 *   nvcc -o complex_kernel complex_kernel.cu
 ***************************************************************************/

#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

const int ARRAY_SIZE = 512;

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
// Kernel #1: Apply sin/cos transformations
//------------------------------------------------------------------------------
__global__ void transformKernel(const float* in, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        out[idx] = sinf(val) + cosf(val);
    }
}

//------------------------------------------------------------------------------
// Kernel #2: Parallel reduction to compute the sum of an array
//------------------------------------------------------------------------------
__global__ void reduceKernel(const float* in, float* out, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (i < n) {
        sdata[tid] = in[i];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        out[blockIdx.x] = sdata[0];
    }
}

//------------------------------------------------------------------------------
// Main function
//------------------------------------------------------------------------------
int main() {
    // Host memory
    float h_in[ARRAY_SIZE], h_transform[ARRAY_SIZE];
    float h_intermediate[32];
    float h_result = 0.0f;

    // Device memory
    float *d_in, *d_transform, *d_intermediate;

    // Initialize input array
    for (int i = 0; i < ARRAY_SIZE; i++) {
        h_in[i] = static_cast<float>(i) * 0.1f;
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_in, ARRAY_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_transform, ARRAY_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_intermediate, 32 * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, ARRAY_SIZE * sizeof(float), cudaMemcpyHostToDevice));

    // Launch transformKernel
    int blockSize = 256;
    int gridSize = (ARRAY_SIZE + blockSize - 1) / blockSize;
    transformKernel<<<gridSize, blockSize>>>(d_in, d_transform, ARRAY_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Launch reduceKernel
    reduceKernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_transform, d_intermediate, ARRAY_SIZE);
    CUDA_CHECK(cudaGetLastError());

    // Copy partial sums back to host
    CUDA_CHECK(cudaMemcpy(h_intermediate, d_intermediate, gridSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Finish reduction on CPU
    for (int i = 0; i < gridSize; i++) {
        h_result += h_intermediate[i];
    }

    // Print final reduced result
    std::cout << h_result << std::endl;

    // Free device memory
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_transform));
    CUDA_CHECK(cudaFree(d_intermediate));

    return 0;
}

