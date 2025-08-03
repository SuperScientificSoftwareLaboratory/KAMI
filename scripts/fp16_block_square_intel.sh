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

    declare -A M_TO_NUM_RANK
    local source_file=""
    local executable=""
    case "$mode" in
    1d)
        M_TO_NUM_RANK=([16]=1 [32]=1 [64]=1 [128]=2 [192]=3)
        source_file="block_gemm_1d_half_sycl.cpp"
        executable="block_gemm_1d_half_sycl"
        ;;
    2d)
        M_TO_NUM_RANK=([16]=1 [32]=1 [64]=1 [128]=2 [192]=2)
        source_file="block_gemm_2d_half_sycl.cpp"
        executable="block_gemm_2d_half_sycl"
        ;;
    3d)
        M_TO_NUM_RANK=([16]=1 [32]=1 [64]=2 [128]=2 [192]=2)
        source_file="block_gemm_3d_half_sycl.cpp"
        executable="block_gemm_3d_half_sycl"
        ;;
    *)
        echo "Unknown mode: $mode"
        exit 1
        ;;
    esac

    for m_block in 16 32 64 128 192; do
        num_rank=${M_TO_NUM_RANK[$m_block]}
        if [ -z "$num_rank" ]; then
            echo "No NUM_RANK_BLOCK found for M_BLOCK=${m_block}"
            continue
        fi

        echo "Testing: M_BLOCK=${m_block}, NUM_RANK_BLOCK=${num_rank}"
        rm -f ${executable}  
        icpx -fsycl -O3 -ftarget-register-alloc-mode=pvc:large \
        -DM_BLOCK=${m_block} -DN_BLOCK=${m_block} -DK_BLOCK=${m_block} \
        -DNUM_RANK_BLOCK=${num_rank} \
        ${source_file} -o ${executable} >>"${log_path}.log" 2>&1
        ./${executable} >>"${log_path}.log" 2>&1
    done

    grep '\[hemeng_log\]' "${log_path}.log" | sed 's/\[hemeng_log\],//' >"${log_path}.csv"
    cd - >/dev/null
}

ONEAPI_VARS="/opt/intel/oneapi/2025.1/oneapi-vars.sh"

if [ -f "$ONEAPI_VARS" ]; then
    echo "[INFO] Loading oneAPI environment from $ONEAPI_VARS"
    source "$ONEAPI_VARS" --force
else
    echo "[ERROR] oneAPI 2025.1 environment not found at:"
    echo "$ONEAPI_VARS"
    echo "Please check if the version is installed or the path is correct."
    exit 1
fi

run_block_test "../src/block_gemm/intel/1d" "../../../../logs/block_gemm/square/intel/fp16_block_square_1d_intel" 1d
run_block_test "../src/block_gemm/intel/2d" "../../../../logs/block_gemm/square/intel/fp16_block_square_2d_intel" 2d
run_block_test "../src/block_gemm/intel/3d" "../../../../logs/block_gemm/square/intel/fp16_block_square_3d_intel" 3d

cd ../src/block_gemm/intel/basic || exit 1
log_file="../../../../logs/block_gemm/square/intel/fp16_basic_intel"
mkdir -p "$(dirname "${log_file}")"
echo "" >"${log_file}.log"

echo "Compiling basic_gemm.cpp..."
icpx -std=c++17 -O2 -fsycl basic_gemm.cpp -o basic_gemm

for m in 16 32 64 128 192; do
    echo "Running basic_gemm ${m}"
    ./basic_gemm "$m" >>"${log_file}.log" 2>&1
done

grep '\[hemeng_log\]' "${log_file}.log" | sed 's/\[hemeng_log\],//' >"${log_file}.csv"
