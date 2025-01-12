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

---

## Project Structure
```plaintext
cuda_testing_framework/
├── cuda_kernels/
│   ├── vector_add.cu
│   ├── matrix_mul.cu
│   └── complex_kernel.cu
├── expected_output/
│   ├── vector_add_output.txt
│   ├── matrix_mul_output.txt
│   └── complex_kernel_output.txt
└── test_runner.py
'''

Features
Automatic Compilation of CUDA programs using nvcc.
Output Verification by comparing actual outputs to expected results.
GPU Performance Monitoring:
Memory Usage before and after execution.
Temperature Changes during execution.
