/***************************************************************************
 * File: matrix_mul.cu
 * Description: Demonstrates basic matrix multiplication using CUDA.
 * 
 * Compile Command:
 *   nvcc -o matrix_mul matrix_mul.cu
 ***************************************************************************/

#include <iostream>
#include <cuda_runtime.h>

const int M = 4;  // Rows of matrix A
const int N = 4;  // Columns of matrix A / Rows of matrix B
const int P = 4;  // Columns of matrix B

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
// Matrix multiplication kernel
//------------------------------------------------------------------------------
__global__ void matrixMultiply(const float* A, const float* B, float* C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * p + col];
        }
        C[row * p + col] = sum;
    }
}

//------------------------------------------------------------------------------
// Main function
//------------------------------------------------------------------------------
int main() {
    // Host matrices
    float h_A[M * N], h_B[N * P], h_C[M * P];

    // Device matrices
    float *d_A, *d_B, *d_C;

    size_t size_A = M * N * sizeof(float);
    size_t size_B = N * P * sizeof(float);
    size_t size_C = M * P * sizeof(float);

    // Initialize matrices A and B with sample data
    for (int i = 0; i < M * N; i++) h_A[i] = static_cast<float>(i + 1);
    for (int i = 0; i < N * P; i++) h_B[i] = static_cast<float>(i + 1);

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_A, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size_B));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size_C));

    // Copy matrices from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    // Define block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((P + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    matrixMultiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, P);

    // Check for kernel errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));

    // Print the result matrix
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < P; j++) {
            std::cout << h_C[i * P + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}

