#!/bin/bash

run_block_test() {
    local dir=$1
    local log_path=$2
    local mode=$3

    if [ ! -d "$dir" ]; then
        echo "Directory does not exist: $dir"
        exit 1
    fi
    local curr_dir=$(pwd)
    cd "$dir"
    echo "" >"${log_path}.log"

    declare -A NUM_RANK_BLOCK
    declare -A NUM_ALLOC_RANK_BLOCK
    declare -a M_BLOCKS

    case "$mode" in
    1d)
        exe=block_gemm_1d_half_wmma
        src=${exe}.cu
        NUM_RANK_BLOCK=([16]=1 [32]=1 [64]=1 [128]=8 [192]=12)
        NUM_ALLOC_RANK_BLOCK=([16]=1 [32]=1 [64]=1 [128]=8 [192]=16)
        M_BLOCKS=(16 32 64 128 192)
        ;;
    2d)
        exe=block_gemm_2d_half_wmma
        src=${exe}.cu
        NUM_RANK_BLOCK=([16]=1 [32]=1 [64]=2 [128]=2 [192]=3)
        NUM_ALLOC_RANK_BLOCK=([16]=1 [32]=1 [64]=4 [128]=9 [192]=9)
        M_BLOCKS=(16 32 64 128 192)
        ;;
    3d)
        exe=block_gemm_3d_half_wmma
        src=${exe}.cu
        NUM_RANK_BLOCK=([16]=1 [32]=1 [64]=2 [128]=2)
        NUM_ALLOC_RANK_BLOCK=([16]=1 [32]=1 [64]=8 [128]=8)
        M_BLOCKS=(16 32 64 128)
        ;;
    *)
        echo "Unknown mode: $mode"
        exit 1
        ;;
    esac

    for m_block in "${M_BLOCKS[@]}"; do
        num_rank=${NUM_RANK_BLOCK[$m_block]}
        num_alloc=${NUM_ALLOC_RANK_BLOCK[$m_block]}

        echo "Testing M_BLOCK=${m_block}, NUM_RANK_BLOCK=${num_rank}, NUM_ALLOC_RANK_BLOCK=${num_alloc}"
        rm -f ${exe}
        hipcc --offload-arch=gfx1100 -O3 \
            -DM_BLOCK=${m_block} -DN_BLOCK=${m_block} -DK_BLOCK=${m_block} \
            -DNUM_RANK_BLOCK=${num_rank} -DNUM_ALLOC_RANK_BLOCK=${num_alloc} \
            -o ${exe} ${src}

        if [ $? -eq 0 ]; then
            echo "Compilation successful, running ${exe}"
            ./${exe} >>"${log_path}.log" 2>&1
        else
            echo "Compilation failed, skipping M_BLOCK=${m_block}"
        fi
    done

    grep '\[hemeng_log\]' "${log_path}.log" | sed 's/\[hemeng_log\],//' >"${log_path}.csv"
    cd "$curr_dir"
}

run_block_test "../src/block_gemm/amd/1d" "../../../../logs/block_gemm/square/amd/fp16_block_square_1d_amd" 1d
run_block_test "../src/block_gemm/amd/2d" "../../../../logs/block_gemm/square/amd/fp16_block_square_2d_amd" 2d
run_block_test "../src/block_gemm/amd/3d" "../../../../logs/block_gemm/square/amd/fp16_block_square_3d_amd" 3d
