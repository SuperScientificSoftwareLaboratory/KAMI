#!/bin/bash

run_block_test() {
    local mode=$1
    local src_dir=$2
    local log_prefix=$3

    declare -A M_TO_NUM_RANK
    declare -A M_TO_NUM_ALLOC_RANK
    case "$mode" in
    1d)
        exe=block_gemm_1d_half_mma
        M_TO_NUM_RANK=([16]=1 [32]=1 [64]=1 [128]=4 [192]=4)
        M_TO_NUM_ALLOC_RANK=([16]=8 [32]=1 [64]=1 [128]=4 [192]=4)
        ;;
    2d)
        exe=block_gemm_2d_half_mma
        M_TO_NUM_RANK=([16]=1 [32]=1 [64]=1 [128]=2 [192]=2)
        M_TO_NUM_ALLOC_RANK=([16]=3 [32]=3 [64]=1 [128]=4 [192]=4)
        ;;
    3d)
        exe=block_gemm_3d_half_mma
        M_TO_NUM_RANK=([16]=1 [32]=1 [64]=2 [128]=2 [192]=2)
        M_TO_NUM_ALLOC_RANK=([16]=3 [32]=1 [64]=8 [128]=8 [192]=8)
        ;;
    *)
        echo "Unknown mode: $mode"
        exit 1
        ;;
    esac

    cd "$src_dir" || {
        echo "Directory not found: $src_dir"
        exit 1
    }
    log_file="${log_prefix}.log"
    csv_file="${log_prefix}.csv"
    echo "" >"$log_file"

    for m_block in $(echo "${!M_TO_NUM_RANK[@]}" | tr ' ' '\n' | sort -n); do
        num_rank=${M_TO_NUM_RANK[$m_block]}
        num_alloc_rank=${M_TO_NUM_ALLOC_RANK[$m_block]}

        echo "Testing M_BLOCK=${m_block}, NUM_RANK_BLOCK=${num_rank}, NUM_ALLOC_RANK_BLOCK=${num_alloc_rank}..."
        real_exe=$exe
        if [ "$mode" = "1d" ] && [ "$m_block" -eq 128 ]; then
            real_exe=block_gemm_1d_half_mma_128
        fi
        if [ "$mode" = "1d" ] && [ "$m_block" -eq 192 ]; then
            pipe_block=6
        else
            pipe_block=1
        fi
        nvcc -arch=sm_90a -O3 -Xptxas -O3 --ptxas-options=-v -lineinfo \
            -DM_BLOCK=${m_block} -DN_BLOCK=${m_block} -DK_BLOCK=${m_block} \
            -DNUM_RANK_BLOCK=${num_rank} -DNUM_ALLOC_RANK_BLOCK=${num_alloc_rank} \
            -DNUM_PIPE_BLOCK=${pipe_block} \
            -o ${exe} ${real_exe}.cu >>"$log_file" 2>&1

        if [ $? -eq 0 ]; then
            echo "Compilation successful, running..."
            ./${exe} >>"$log_file" 2>&1
        else
            echo "Compilation failed for M_BLOCK=${m_block}, skipping..."
        fi
    done

    grep '\[hemeng_log\]' "$log_file" | sed 's/\[hemeng_log\],//' >"$csv_file"
    cd - >/dev/null
}

run_cublasdx_test() {
    local src_dir=$1
    local log_prefix=$2

    cd "$src_dir" || {
        echo "Directory not found: $src_dir"
        exit 1
    }
    log_file="${log_prefix}.log"
    csv_file="${log_prefix}.csv"
    echo "" >"$log_file"

    for value in 16 32 64 128 192; do
        echo "Running cuBLASDx test with INPUT_M=$value and INPUT_K=$value..." | tee -a $log_file

        rm -rf fp16_single_gemm_performance

        nvcc -std=c++17 -arch=sm_90a --expt-relaxed-constexpr -lcublas -O3 --ptxas-options=-v -lineinfo \
            -DCUBLASDX_EXAMPLE_ENABLE_SM_90 -DINPUT_M=$value -DINPUT_K=$value \
            -I./24.08/include/cublasdx/include/ \
            -I./24.08/external/cutlass/include/ -I./24.08/include \
            -o fp16_single_gemm_performance fp16_single_gemm_performance.cu

        if [ ! -f "./fp16_single_gemm_performance" ]; then
            echo "Compilation failed for cuBLASDx with INPUT_M=$value and INPUT_K=$value" | tee -a $log_file
            continue
        fi
        chmod +x ./fp16_single_gemm_performance
        ./fp16_single_gemm_performance >>$log_file 2>&1
    done

    grep '\[hemeng_log\]' "$log_file" | sed 's/\[hemeng_log\],//' >"$csv_file"
    cd - >/dev/null
}

run_cutlass_test() {
    local config=$1
    local src_dir=$2
    local log_prefix=$3

    cd "$src_dir" || {
        echo "Directory not found: $src_dir"
        exit 1
    }

    echo "Running CUTLASS with configuration: $config"

    rm -rf fp16_mma_multistage
    nvcc -std=c++17 -arch=sm_90a --expt-relaxed-constexpr -lcublas -O3 --ptxas-options=-v -lineinfo \
        -Iutil/include -Iinclude \
        -Igoogletest/include \
        googletest/libgtest.a \
        googletest/libgtest_main.a \
        -lpthread \
        $(for param in $config; do echo "-D$param"; done) -o fp16_mma_multistage fp16_mma_multistage.cu

    if [ $? -eq 0 ]; then
        echo "Compilation successful. Running the executable..."
        ./fp16_mma_multistage >>"${log_prefix}.log" 2>&1
        grep '\[hemeng_log\]' "${log_prefix}.log" | sed 's/\[hemeng_log\],//' >"${log_prefix}.csv"
    else
        echo "Compilation failed for configuration: $config"
    fi

    cd - >/dev/null
}

run_block_test 1d "../src/block_gemm/KAMI/1d/fp16" "../../../../../logs/block_gemm/square/H200/fp16_block_square_1d_H200"
run_block_test 2d "../src/block_gemm/KAMI/2d/fp16" "../../../../../logs/block_gemm/square/H200/fp16_block_square_2d_H200"
run_block_test 3d "../src/block_gemm/KAMI/3d/fp16" "../../../../../logs/block_gemm/square/H200/fp16_block_square_3d_H200"

run_cublasdx_test "../src/block_gemm/cuBLASDx" "../../../logs/block_gemm/square/H200/fp16_cublasdx"

CONFIGS=(
    "Problem_m=16 Problem_n=16 Problem_k=16 Block_m=16 Block_n=16 Block_k=16 Warp_m=16 Warp_n=16 Warp_k=16 Instruction_m=16 Instruction_n=8 Instruction_k=8 STAGES=5"
    "Problem_m=32 Problem_n=32 Problem_k=32 Block_m=32 Block_n=32 Block_k=32 Warp_m=32 Warp_n=32 Warp_k=32 Instruction_m=16 Instruction_n=8 Instruction_k=16 STAGES=5"
    "Problem_m=64 Problem_n=64 Problem_k=64 Block_m=64 Block_n=64 Block_k=64 Warp_m=32 Warp_n=32 Warp_k=64 Instruction_m=16 Instruction_n=8 Instruction_k=16 STAGES=4"
    "Problem_m=128 Problem_n=128 Problem_k=128 Block_m=128 Block_n=128 Block_k=32 Warp_m=32 Warp_n=64 Warp_k=32 Instruction_m=16 Instruction_n=8 Instruction_k=16 STAGES=2"
    "Problem_m=192 Problem_n=192 Problem_k=192 Block_m=192 Block_n=192 Block_k=32 Warp_m=192 Warp_n=64 Warp_k=32 Instruction_m=16 Instruction_n=8 Instruction_k=16 STAGES=1"
)

echo "" >"../logs/block_gemm/square/H200/fp16_CUTLASS.log"
echo "" >"../logs/block_gemm/square/H200/fp16_CUTLASS.csv"
cd ../src/block_gemm/CUTLASS/googletest
g++ -std=c++17 -O2 -I include -I . -c src/gtest-all.cc -o gtest-all.o
ar rcs libgtest.a gtest-all.o

g++ -std=c++17 -O2 -I include -I . -c src/gtest_main.cc -o gtest_main.o
ar rcs libgtest_main.a gtest_main.o
cd ../../../../scripts
for config in "${CONFIGS[@]}"; do
    run_cutlass_test "$config" "../src/block_gemm/CUTLASS" "../../../logs/block_gemm/square/H200/fp16_CUTLASS"
done
