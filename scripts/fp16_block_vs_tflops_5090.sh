#!/bin/bash

run_block_test() {
    local mode=$1
    local src_dir=$2
    local log_prefix=$3

    declare -a NUM_RANK_BLOCK_LIST
    declare -a NUM_ALLOC_RANK_BLOCK_LIST

    case "$mode" in
        1d)
            exe=block_gemm_1d_half_mma
            NUM_RANK_BLOCK_LIST=(1 2 2 4 4 4)
            NUM_ALLOC_RANK_BLOCK_LIST=(1 2 3 4 8 16)
            ;;
        2d)
            exe=block_gemm_2d_half_mma
            NUM_RANK_BLOCK_LIST=(1 1 1 2 2 4)
            NUM_ALLOC_RANK_BLOCK_LIST=(1 2 3 4 8 16)
            ;;
        3d)
            exe=block_gemm_3d_half_mma
            NUM_RANK_BLOCK_LIST=(1 1 1 1 2 2)
            NUM_ALLOC_RANK_BLOCK_LIST=(1 2 3 4 8 16)
            ;;
        *)
            echo "Unknown mode: $mode"
            exit 1
            ;;
    esac

    cd "$src_dir" || { echo "Directory not found: $src_dir"; exit 1; }
    log_file="${log_prefix}.log"
    csv_file="${log_prefix}.csv"
    echo "" > "$log_file"

    for i in "${!NUM_RANK_BLOCK_LIST[@]}"; do
        num_rank=${NUM_RANK_BLOCK_LIST[$i]}
        num_alloc_rank=${NUM_ALLOC_RANK_BLOCK_LIST[$i]}

        echo "Testing M=N=K=64, NUM_RANK_BLOCK=${num_rank}, NUM_ALLOC_RANK_BLOCK=${num_alloc_rank}..."

        nvcc -arch=sm_90a -O3 -Xptxas -O3 --ptxas-options=-v -lineinfo \
            -DM_BLOCK=64 -DN_BLOCK=64 -DK_BLOCK=64 \
            -DNUM_RANK_BLOCK=${num_rank} -DNUM_ALLOC_RANK_BLOCK=${num_alloc_rank} \
            -o ${exe} ${exe}.cu >> "$log_file" 2>&1

        if [ $? -eq 0 ]; then
            echo "Compilation successful, running..."
            ./${exe} >> "$log_file" 2>&1
        else
            echo "Compilation failed for NUM_RANK_BLOCK=${num_rank}, skipping..."
        fi
    done

    grep '\[hemeng_log\]' "$log_file" | sed 's/\[hemeng_log\],//' > "$csv_file"
    cd - > /dev/null
}

run_block_test 1d "../src/block_gemm/KAMI/1d/fp16" "../../../../../logs/tflops_vs_blocksize/remap_5090_fp16_1d"
run_block_test 2d "../src/block_gemm/KAMI/2d/fp16" "../../../../../logs/tflops_vs_blocksize/remap_5090_fp16_2d"
run_block_test 3d "../src/block_gemm/KAMI/3d/fp16" "../../../../../logs/tflops_vs_blocksize/remap_5090_fp16_3d"