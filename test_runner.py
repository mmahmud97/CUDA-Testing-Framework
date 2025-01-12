#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_runner.py
A Python script to automate the testing of CUDA programs.
"""

import os
import subprocess
import time

# Configuration
CUDA_KERNELS_DIR = "cuda_kernels"
EXPECTED_OUTPUT_DIR = "expected_output"
EXECUTABLE_EXTENSION = ".exe" if os.name == "nt" else ""  # Windows or Linux/Mac

# Test configuration for the vector_add program
TEST_CONFIG = {
    "name": "VectorAdd",
    "source_file": "vector_add.cu",
    "executable_name": "vector_add" + EXECUTABLE_EXTENSION,
    "expected_output_file": "vector_add_output.txt",
}

def compile_cuda_program(source_file, output_binary):
    """Compiles a CUDA program using nvcc."""
    cmd = ["nvcc", "-o", output_binary, source_file]
    print(f"[INFO] Compiling: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print("[ERROR] Compilation failed!")
        print(e.output)
        return False

def run_executable(exec_path):
    """Runs a compiled executable and captures its output."""
    print(f"[INFO] Running: {exec_path}")
    try:
        result = subprocess.run(exec_path, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print("[ERROR] Execution failed!")
        return None

def compare_output(actual_output, expected_output_file):
    """Compares the actual output to the expected output."""
    with open(expected_output_file, 'r') as f:
        expected_output = f.read().strip()

    return actual_output == expected_output

def main():
    """Main function to run the test."""
    print("\n=== Running Test: VectorAdd ===\n")
    
    # Paths
    source_file = os.path.join(CUDA_KERNELS_DIR, TEST_CONFIG["source_file"])
    executable_path = os.path.join(".", TEST_CONFIG["executable_name"])
    expected_output_path = os.path.join(EXPECTED_OUTPUT_DIR, TEST_CONFIG["expected_output_file"])

    # 1. Compile the CUDA program
    if not compile_cuda_program(source_file, executable_path):
        print("[FAIL] Compilation Error")
        return

    # 2. Run the compiled executable
    actual_output = run_executable(executable_path)
    if actual_output is None:
        print("[FAIL] Runtime Error")
        return

    # 3. Compare the output to the expected output
    if compare_output(actual_output, expected_output_path):
        print("[PASS] Output matches expected results.")
    else:
        print("[FAIL] Output does not match expected results.")

if __name__ == "__main__":
    main()

