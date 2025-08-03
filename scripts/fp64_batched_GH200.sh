#!/bin/bash

base_dir=$(pwd)
batchnum_list=(1000 10000)
block_sizes=(16 32 48 64 96 128)

declare -A block_to_num_rank=(
    [16]=1
    [32]=2
    [48]=3
    [64]=4
    [96]=6
    [128]=8
)
magma_log_dir="${base_dir}/../logs/batched_gemm/H200"
mkdir -p "$magma_log_dir"
magma_log_prefix="${magma_log_dir}/MAGMA_batch_fp64"
magma_log_file="${magma_log_prefix}.log"
magma_csv_file="${magma_log_prefix}.csv"
echo "" >"$magma_log_file"

kami_log_prefix="${magma_log_dir}/KAMI_batch_fp64"
kami_log_file="${kami_log_prefix}.log"
kami_csv_file="${kami_log_prefix}.csv"
echo "" >"$kami_log_file"

if [ -z "$CUDADIR" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDADIR="/usr/local/cuda"
        echo "CUDADIR set to: $CUDADIR"
    else
        echo "WARNING: /usr/local/cuda not found and CUDADIR is not set."
    fi
else
    echo "CUDADIR already set to: $CUDADIR"
fi

if [ -n "$CUDADIR" ]; then
    CUDA_INCLUDE="$CUDADIR/include"
    CUDA_LIB="$CUDADIR/lib64"
    echo "CUDA_INCLUDE = $CUDA_INCLUDE"
    echo "CUDA_LIB = $CUDA_LIB"
else
    echo "CUDA_INCLUDE and CUDA_LIB not set because CUDADIR is missing."
fi

if [ -f "/opt/intel/oneapi/setvars.sh" ]; then
    . /opt/intel/oneapi/setvars.sh
    echo "oneAPI environment sourced successfully."
    echo "LD_LIBRARY_PATH is now: $LD_LIBRARY_PATH"
else
    echo "WARNING: /opt/intel/oneapi/setvars.sh not found. oneAPI environment not set."
fi

MAGMA_PARENT_DIR="${base_dir}/../src/batched_gemm/MAGMA"
MAGMA_ZIP="${MAGMA_PARENT_DIR}/magma-2.9.0.zip"
MAGMA_DIR="${MAGMA_PARENT_DIR}/magma-2.9.0"
TESTING_DIR="${MAGMA_DIR}/testing"
export GPU_TARGET=sm_90a
if [ ! -d "$MAGMA_DIR" ]; then
    if [ -f "$MAGMA_ZIP" ]; then
        echo "Unzipping MAGMA package..." | tee -a "$magma_log_file"
        unzip "$MAGMA_ZIP" -x "__MACOSX/*" -d "$MAGMA_PARENT_DIR" >>"$magma_log_file" 2>&1
    else
        echo "MAGMA zip file not found at $MAGMA_ZIP. Cannot proceed with compilation." | tee -a "$magma_log_file"
        exit 1
    fi
fi
chmod +x "${MAGMA_DIR}/tools/codegen.py"

cd "$TESTING_DIR" || {
    echo "Failed to enter directory: $TESTING_DIR" | tee -a "$magma_log_file"
    exit 1
}

echo "Compiling testing_dgemm_batched ..." | tee -a "$magma_log_file"
make -j"$(nproc)" >>"$magma_log_file" 2>&1

if [ ! -f ./testing_dgemm_batched ]; then
    echo "Compilation failed: testing_dgemm_batched was not generated." | tee -a "$magma_log_file"
    exit 1
else
    echo "Compilation successful! Running tests..." | tee -a "$magma_log_file"
fi

for batch in "${batchnum_list[@]}"; do
    for n in "${block_sizes[@]}"; do
        echo "Running: batch=$batch, n=$n" | tee -a "$magma_log_file"
        ./testing_dgemm_batched --batch "$batch" -n "$n" >>"$magma_log_file" 2>&1
        echo "----------------------------------------" >>"$magma_log_file"
    done
done

cd "$base_dir" || exit

for batchnum in "${batchnum_list[@]}"; do
    for block_size in "${block_sizes[@]}"; do
        num_rank_block=${block_to_num_rank[$block_size]}
        if [ "$block_size" -eq 16 ]; then
            if [ "$batchnum" -eq 1000 ]; then
                exe="batched_gemm_2d_double_mma"
                exe_dir="${base_dir}/../src/batched_gemm/KAMI/2d"
            else
                exe="batched_gemm_3d_double_mma"
                exe_dir="${base_dir}/../src/batched_gemm/KAMI/3d"
            fi
        else
            exe="batched_gemm_1d_double_mma"
            exe_dir="${base_dir}/../src/batched_gemm/KAMI/1d"
        fi
        echo "Running $exe with M/N/K=$block_size, Batch=$batchnum, NUM_RANK_BLOCK=$num_rank_block" | tee -a "$kami_log_file"
        cd "$exe_dir" || {
            echo "Directory not found: $exe_dir" | tee -a "$kami_log_file"
            exit 1
        }
        rm -f $exe
        if [ "$block_size" -eq 128 ]; then
            src_file="batched_gemm_1d_double_mma_128.cu"
            extra_define="-DK_ALL_BLOCK=$block_size"
        else
            src_file="${exe}.cu"
            extra_define="-DK_BLOCK=$block_size"
        fi
        nvcc -arch=sm_90a -O3 -Xptxas -O3 --ptxas-options=-v -lineinfo \
            -DM_BLOCK=$block_size -DN_BLOCK=$block_size $extra_define \
            -DNUM_BATCHES=$batchnum -DNUM_RANK_BLOCK=$num_rank_block \
            -o $exe $src_file >>"$kami_log_file" 2>&1

        if [ $? -eq 0 ]; then
            echo "Compilation successful. Running..." | tee -a "$kami_log_file"
            ./$exe >>"$kami_log_file" 2>&1
        else
            echo "Compilation failed for M/N/K=$block_size, skipping..." | tee -a "$kami_log_file"
        fi

        cd "$base_dir"
        sleep 2
    done
done
grep '\[hemeng_log\]' "$kami_log_file" | sed 's/\[hemeng_log\],//' >"$kami_csv_file"
grep '\[hemeng_log\]' "$magma_log_file" | sed 's/\[hemeng_log\],//' >"$magma_csv_file"
echo "All tests completed."
echo "MAGMA logs: $magma_log_file, CSV: $magma_csv_file"
echo "KAMI logs: $kami_log_file, CSV: $kami_csv_file"
