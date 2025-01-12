# CUDA-Testing-Framework

CUDA Code Testing Framework with GPU Monitoring
Overview
This project is a CUDA Code Testing Framework designed to:

Compile CUDA programs automatically using nvcc.
Run the compiled binaries on a GPU.
Verify correctness by comparing outputs to expected results.
Measure performance metrics, including:
GPU memory usage during execution.
GPU temperature changes caused by running the CUDA programs.
Generate a summary report with pass/fail results, GPU memory usage, and temperature changes.
This framework is suitable for testing CUDA applications in a professional production environment and is designed with NVIDIA-level stakeholders in mind.

Project Structure
Copy code
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
Directory Breakdown:
Directory/File	Description
cuda_kernels/	Contains CUDA programs (.cu files) to be tested.
expected_output/	Stores expected outputs for each CUDA program to verify correctness.
test_runner.py	The main Python script that compiles, runs, and verifies the CUDA programs.
CUDA Programs in the Framework
1️⃣ vector_add.cu
A simple CUDA program that performs vector addition:

Adds two input vectors A and B and stores the result in vector C.
Outputs the first 10 elements of the result vector to the console.
Expected Output:

Copy code
0 3 6 9 12 15 18 21 24 27
2️⃣ matrix_mul.cu
A CUDA program that performs matrix multiplication:

Multiplies two matrices A and B to produce a result matrix C.
Outputs the entire result matrix to the console.
Expected Output:

Copy code
90 100 110 120
202 228 254 280
314 356 398 440
426 484 542 600
3️⃣ complex_kernel.cu
A more advanced CUDA program that demonstrates:

Element-wise transformations using sin(x) + cos(x).
Parallel reduction to compute the sum of transformed values.
Expected Output:

Copy code
130.118
How to Use the Framework
✅ Prerequisites
CUDA Toolkit must be installed on your system.
NVIDIA GPU with the necessary CUDA-compatible drivers.
Python 3.x installed with the following packages:
pynvml for GPU monitoring.
subprocess (built-in) for running shell commands.
📦 Installing Required Python Packages
Run the following command to install the required Python package:

bash
Copy code
pip install pynvml
🛠️ Running the Test Framework
Clone or copy the repository to your local machine.

Navigate to the project directory:

bash
Copy code
cd cuda_testing_framework
Run the Python test runner:
bash
Copy code
python test_runner.py
📋 Sample Output:
yaml
Copy code
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
Features of the Framework
Feature	Description
Automatic Compilation	Uses nvcc to compile CUDA programs before running them.
Output Verification	Compares the actual output of the CUDA programs to expected results.
GPU Memory Monitoring	Tracks the GPU memory usage before and after running the CUDA programs.
GPU Temperature Monitoring	Tracks the GPU temperature changes caused by each CUDA program.
Performance Metrics Collected
Metric	Description
GPU Memory Used	The difference in GPU memory usage (in MB) before and after running the executable.
GPU Temperature Increase	The increase in GPU temperature (in °C) during the execution of the CUDA program.
Adding New Tests
You can easily add new CUDA programs to the framework by:

Placing the new .cu file in the cuda_kernels/ directory.
Adding the expected output to the expected_output/ directory.
Updating the TEST_CONFIGS list in test_runner.py with the new test case.
Example:

python
Copy code
TEST_CONFIGS = [
    {
        "name": "NewTest",
        "source_file": "new_test.cu",
        "executable_name": "new_test",
        "expected_output_file": "new_test_output.txt",
    }
]
How the Framework Works
Compile the CUDA Program
The framework uses nvcc to compile each .cu file into an executable.

Run the Compiled Executable
The compiled binary is executed on the GPU, and its output is captured.

Compare Output
The output from the CUDA program is compared to the expected output file.

Collect Performance Metrics
The framework monitors:

GPU Memory Usage before and after execution.
GPU Temperature before and after execution.
Generate a Report
The framework prints a summary report to the console showing the pass/fail status, memory usage, and temperature changes.

Example Use Cases
Automated Testing for CUDA Developers
Quickly verify that CUDA kernels produce correct outputs and perform efficiently on GPUs.

Performance Tracking
Identify potential performance bottlenecks by tracking GPU memory usage and temperature changes.

Regression Testing
Ensure that CUDA programs continue to produce correct outputs as code changes over time.

Future Enhancements (Suggestions)
Here are some ideas for further improving the framework:

Log Test Results to a File
Output the test results in JSON, CSV, or HTML format for easy reporting and analysis.

Set Thresholds for Memory and Temperature
Automatically flag tests if memory usage or temperature exceeds certain thresholds.

Integrate with CI/CD Pipelines
Automate the framework in a continuous integration/continuous deployment (CI/CD) pipeline using GitHub Actions, Jenkins, etc.

Conclusion
This CUDA Code Testing Framework is a powerful tool for ensuring the correctness and performance of CUDA applications. It provides:

Automated compilation and testing.
Output verification.
GPU memory usage tracking.
GPU temperature monitoring.
With this framework, you can ensure the stability and efficiency of your CUDA programs while monitoring their impact on GPU resources.

Author: Your Name
License: MIT License

