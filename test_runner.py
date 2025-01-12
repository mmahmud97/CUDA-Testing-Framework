#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_runner.py
A Python script to automate the testing of CUDA programs, and monitor GPU memory usage and temperature.
"""

import os
import subprocess
import time
import pynvml  # NVIDIA Management Library for GPU monitoring

# Configuration
CUDA_KERNELS_DIR = "cuda_kernels"
EXPECTED_OUTPUT_DIR = "expected_output"
EXECUTABLE_EXTENSION = ".exe" if os.name == "nt" else ""

# Test configuration
TEST_CONFIGS = [
    {
        "name": "VectorAdd",
        "source_file": "vector_add.cu",
        "executable_name": "vector_add" + EXECUTABLE_EXTENSION,
        "expected_output_file": "vector_add_output.txt",
    },
    {
        "name": "MatrixMul",
        "source_file": "matrix_mul.cu",
        "executable_name": "matrix_mul" + EXECUTABLE_EXTENSION,
        "expected_output_file": "matrix_mul_output.txt",
    },
    {
        "name": "ComplexKernel",
        "source_file": "complex_kernel.cu",
        "executable_name": "complex_kernel" + EXECUTABLE_EXTENSION,
        "expected_output_file": "complex_kernel_output.txt",
    }
]

# Initialize NVML for GPU monitoring
def initialize_nvml():
    try:
        pynvml.nvmlInit()
        print(f"[INFO] NVML Initialized: GPU Detected - {pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))}")
    except pynvml.NVMLError as error:
        print(f"[ERROR] NVML Initialization Failed: {error}")
        exit(1)

# Get GPU memory usage in MB
def get_gpu_memory_usage():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming GPU 0
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.used / (1024 * 1024)  # Convert bytes to MB

# Get GPU temperature in Celsius
def get_gpu_temperature():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # Assuming GPU 0
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    return temp

# Compile the CUDA program
def compile_cuda_program(source_file, output_binary):
    cmd = ["nvcc", "-o", output_binary, source_file]
    print(f"[INFO] Compiling: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print("[ERROR] Compilation failed!")
        print(e.output)
        return False

# Run the compiled executable
def run_executable(exec_path):
    print(f"[INFO] Running: {exec_path}")
    try:
        result = subprocess.run(exec_path, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print("[ERROR] Execution failed!")
        return None

# Compare output to expected results
def compare_output(actual_output, expected_output_file):
    with open(expected_output_file, 'r') as f:
        expected_output = f.read().strip()
    return actual_output == expected_output

# Main function to run all tests
def main():
    initialize_nvml()  # Initialize NVML for GPU monitoring

    for test_config in TEST_CONFIGS:
        print(f"\n=== Running Test: {test_config['name']} ===\n")

        # Paths
        source_file = os.path.join(CUDA_KERNELS_DIR, test_config["source_file"])
        executable_path = os.path.join(".", test_config["executable_name"])
        expected_output_path = os.path.join(EXPECTED_OUTPUT_DIR, test_config["expected_output_file"])

        # Compile the CUDA program
        if not compile_cuda_program(source_file, executable_path):
            print(f"[FAIL] {test_config['name']}: Compilation Error")
            continue

        # Measure GPU memory usage and temperature before running
        pre_gpu_memory = get_gpu_memory_usage()
        pre_gpu_temp = get_gpu_temperature()

        # Run the compiled executable
        actual_output = run_executable(executable_path)
        if actual_output is None:
            print(f"[FAIL] {test_config['name']}: Runtime Error")
            continue

        # Measure GPU memory usage and temperature after running
        post_gpu_memory = get_gpu_memory_usage()
        post_gpu_temp = get_gpu_temperature()

        # Calculate memory and temperature differences
        gpu_memory_used = post_gpu_memory - pre_gpu_memory
        gpu_temp_increase = post_gpu_temp - pre_gpu_temp

        # Compare the output to the expected output
        if compare_output(actual_output, expected_output_path):
            print(f"[PASS] {test_config['name']}: Output matches expected results.")
        else:
            print(f"[FAIL] {test_config['name']}: Output does not match expected results.")

        # Report GPU memory usage and temperature
        print(f"[INFO] GPU Memory Used: {gpu_memory_used:.2f} MB")
        print(f"[INFO] GPU Temperature Increase: {gpu_temp_increase} °C")

if __name__ == "__main__":
    main()
