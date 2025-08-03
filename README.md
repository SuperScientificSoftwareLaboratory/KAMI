# KAMI: Communication-Avoiding General Matrix Multiplication within a Single GPU

This repository contains the artifact for the SC '25 paper submission "KAMI: Communication-Avoiding General Matrix Multiplication within a Single GPU."

## Artifact Setup

### Hardware

KAMI is evaluated on four GPUs: NVIDIA GH200, NVIDIA RTX 5090, AMD 7900 XTX and Intel Max 1100.

### Software

- The NVIDIA GH200 is installed with Ubuntu 22.04 using GCC v11.4 and NVCC v12.8.

- The NVIDIA RTX 5090 is installed with Ubuntu 24.04 using GCC v11.4 and NVCC v12.8.

- The AMD 7900 XTX is installed with Ubuntu 24.04 using GCC v11.4 and ROCm 6.10.

- The Intel Max 1100 is using intel® Tiber™ AI Cloud.

## Steps to Reproduce

1. Clone the repository:

   ```bash
   git clone https://github.com/ForADAE/SC25-pap926.git
   ```

2. Navigate into the repository:

   ```bash
   cd SC25-pap926
    ```

3. Run evaluation scripts (Optional):

    Navigate to the `scripts/` directory:

    ```bash
    cd scripts
    ```

    Then, depending on your hardware:

   - For **NVIDIA GH200**:

     ```bash
     bash all_GH200.sh
     ```

   - For **NVIDIA RTX 5090**:

     ```bash
     bash all_5090.sh
     ```

   - For **AMD 7900 XTX**:

     ```bash
     bash all_AMD.sh
     ```

   - For **Intel Max 1100**:

     ```bash
     bash all_intel.sh
     ```

     The output logs will be stored in the `logs/` directory. Each full benchmark run typically takes 100 minutes.

4. Reproduce paper plots:

   Navigate to the `plots/` directory:

    ```bash
    cd plots
    ```
  
   Install necessary Python packages (if not already installed):

    ```bash
    pip3 install numpy pandas matplotlib seaborn
    ```

   Then run:

    ```bash
    bash plots_all.sh
    ```

## Expected Results

The artifact includes performance results for KAMI in both double-precision (FP64) and half-precision (FP16) floating-point formats, along with baseline implementations including cuBLAS, cuBLASDx, CUTLASS, MAGMA, and SYCL-Bench.

Upon successful execution:

- **Performance**: In all tested configurations (matrix sizes, precisions, and hardware platforms), KAMI is expected to outperform cuBLAS, cuBLASDx, CUTLASS, MAGMA, and SYCL-Bench in terms of achieved TFLOPS.
- **Output Files**: The logs containing timing results will be saved under the `logs/` directory, with one file per benchmark run.
- **Plots**: The `plots/` directory will contain regenerated figures identical to those in the SC '25 paper submission.
