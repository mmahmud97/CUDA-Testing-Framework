# CUDA Code Testing Framework with GPU Monitoring

## Overview
This project provides a lightweight **CUDA Code Testing Framework** to automate:

- **Compilation** of CUDA programs using `nvcc`.
- **Execution** on an NVIDIA GPU.
- **Output verification** by comparing program results to expected outputs.
- **GPU performance monitoring**, including:
  - **Memory usage** (in MB).
  - **Temperature changes** (in °C).

The framework ensures CUDA programs run correctly and efficiently while tracking their impact on GPU resources.
MIT LICENSE

## Project Structure
The project is organized into three main directories:

- **cuda_kernels/**: Contains the CUDA programs to be tested.
- **expected_output/**: Stores the expected outputs for each CUDA program to verify correctness.
- **test_runner.py**: The main Python script that compiles, runs, and tests the CUDA programs.

## Features
This framework provides the following features to streamline the testing of CUDA programs:

- ✅ **Automatic Compilation**: Uses the `nvcc` compiler to compile CUDA programs.
- ✅ **Output Verification**: Compares the actual output of the CUDA programs to expected results for correctness.
- ✅ **GPU Performance Monitoring**:
  - Tracks **GPU memory usage** before and after program execution.
  - Monitors **GPU temperature changes** during execution.
- ✅ **Support for Multiple CUDA Programs**: Easily add new CUDA kernels and expected outputs for automated testing.

## Prerequisites
Ensure you have the following tools and libraries installed before running the framework:

1. **CUDA Toolkit** (e.g., CUDA 11.x or higher)  
   - Download and install from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit).
2. **Python 3.x**  
   - Download from [Python Official Website](https://www.python.org/downloads/).
3. **pynvml** (Python NVIDIA Management Library)  
   - Used for monitoring GPU memory and temperature.

### 📦 Installing `pynvml`

Run the following command to install `pynvml` via `pip`:
**pip install pynvml**


## How to Run
Follow these steps to run the CUDA Code Testing Framework:

1. **Clone or copy the repository** to your local machine.

2. **Navigate to the project directory**:

   **cd CUDA-Testing-Framework**
   **python test_runner.py**

## CUDA Programs
The framework includes three sample CUDA programs for testing. Below are descriptions of three program and their expected outputs.

### 1️⃣ **vector_add.cu**
- **Description**: Performs element-wise addition of two vectors.
- **Expected Output**:
  0 3 6 9 12 15 18 21 24 27

### 2️⃣ **matrix_mul.cu**
- **Description**: Multiplies two matrices to produce a result matrix.
- **Expected Output**:
   90 100 110 120
  202 228 254 280
  314 356 398 440
  426 484 542 600

### 3️⃣ **complex_kernel.cu**
- **Description**: Applies mathematical transformations (sin(x) + cos(x)) to an array and performs a parallel reduction to compute the sum of transformed values.
- **Expected Output**:
  130.118

## Performance Metrics Collected
The framework monitors GPU performance during the execution of each CUDA program. The following metrics are collected:

| Metric                | Description                                              |
|-----------------------|----------------------------------------------------------|
| **GPU Memory Usage**   | Tracks the difference in GPU memory usage (in MB) before and after execution of the CUDA program. |
| **GPU Temperature**    | Measures the GPU temperature (in °C) before and after execution, and reports the temperature increase caused by the CUDA program. |

These metrics provide insight into the **resource utilization** of each CUDA program and can help identify **potential performance bottlenecks**.

## Sample Output
Below is a sample output from running the `test_runner.py` script:

```plaintext
[INFO] NVML Initialized: GPU Detected - NVIDIA GeForce RTX 3090

=== Running Test: VectorAdd ===
[INFO] Compiling: nvcc -o vector_add cuda_kernels/vector_add.cu
[INFO] Running: ./vector_add
[PASS] VectorAdd: Output matches expected results.
[INFO] GPU Memory Used: 4.56 MB
[INFO] GPU Temperature Increase: 1 °C

=== Running Test: MatrixMul ===
[INFO] Compiling: nvcc -o matrix_mul cuda_kernels/matrix_mul.cu
[INFO] Running: ./matrix_mul
[PASS] MatrixMul: Output matches expected results.
[INFO] GPU Memory Used: 5.12 MB
[INFO] GPU Temperature Increase: 2 °C

=== Running Test: ComplexKernel ===
[INFO] Compiling: nvcc -o complex_kernel cuda_kernels/complex_kernel.cu
[INFO] Running: ./complex_kernel
[PASS] ComplexKernel: Output matches expected results.
[INFO] GPU Memory Used: 6.89 MB
[INFO] GPU Temperature Increase: 3 °C

