#!/bin/bash

programs=("../src/usage_cycle/block_gemm_1d_half_mma_usage_cycle" "../src/usage_cycle/block_gemm_2d_half_mma_usage_cycle" "../src/usage_cycle/block_gemm_3d_half_mma_usage_cycle")

log_file="../logs/usage_cycle/usage_cycle_5090"

echo "" >${log_file}.log

for program in "${programs[@]}"; do
    nvcc -arch=sm_120 -O3 -Xptxas -O3 --ptxas-options=-v -lineinfo -o ${program} ${program}.cu >>${log_file}.log 2>&1
    echo "Running ${program}..." >>${log_file}.log
    ./${program} >>${log_file}.log 2>&1
done

echo "dimension,warp_id,communication_time,computation_time" >${log_file}.csv

grep 'warp_id' ${log_file}.log | sed 's/warp_id: //' >>${log_file}.csv
