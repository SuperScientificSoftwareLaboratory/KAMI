#!/bin/bash

# Arrays of M_BLOCK, N_BLOCK, and K_BLOCK values
M_BLOCKS=(64)
N_BLOCKS=(32)
K_BLOCKS=(64 96 128 160 192 224 256 320 352 384 448 512)

# CSV output file
output_file="../logs/usage_reg/usage_reg_results"

echo "" >${output_file}.csv                          # Clear CSV file
echo "m,n,k,dimension,reg_count" >${output_file}.csv # CSV header
echo "" >${output_file}.log                          # Clear log file

# Function to run tests
run_test() {
    local m_block=$1
    local n_block=$2
    local k_block=$3
    local program=$4
    local suffix=$5

    nvcc_output=$(nvcc -arch=sm_120 -O3 -Xptxas=-O3 --ptxas-options=-v -lineinfo \
        -DM_BLOCK=${m_block} -DN_BLOCK=${n_block} -DK_BLOCK=${k_block} \
        -o $program $program.cu 2>>${output_file}.log)

    echo "$nvcc_output" >>${output_file}.log

    reg_count=$(grep "ptxas info    : Used .* registers" ${output_file}.log | tail -n 1 | awk '{print $5}')

    echo "${m_block},${n_block},${k_block},${suffix},${reg_count}" >>${output_file}.csv
}

# Iterate through all combinations of M_BLOCK, N_BLOCK, and K_BLOCK
for m_block in "${M_BLOCKS[@]}"; do
    for n_block in "${N_BLOCKS[@]}"; do
        for k_block in "${K_BLOCKS[@]}"; do
            run_test "$m_block" "$n_block" "$k_block" "../src/usage_reg/block_gemm_1d_half_mma_usage_reg" "1d"
            wait
            run_test "$m_block" "$n_block" "$k_block" "../src/usage_reg/block_gemm_2d_half_mma_usage_reg" "2d"
            wait
            run_test "$m_block" "$n_block" "$k_block" "../src/usage_reg/block_gemm_3d_half_mma_usage_reg" "3d"
            wait
        done
    done
done

echo "Test completed."
