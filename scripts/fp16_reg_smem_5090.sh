log_file="../logs/reg_smem/reg_smem_5090"

reg_smem_test() {
    SOURCE="../src/reg_smem/block_gemm_1d_half_mma.cu"
    OUTPUT="../src/reg_smem/block_gemm_1d_half_mma"

    declare -A MBLOCK_CONFIG=(
        [32]="2"
        [64]="2"
        [96]="2"
        [128]="2"
    )

    for M_BLOCK in "${!MBLOCK_CONFIG[@]}"; do
        NUM_RANK_BLOCKS=(${MBLOCK_CONFIG[$M_BLOCK]})
        MAX_REG_K=$((M_BLOCK / 16))

        for NUM_RANK_BLOCK in "${NUM_RANK_BLOCKS[@]}"; do

            if [ "$M_BLOCK" -eq 128 ]; then
                reg_k_values=(2 4 6 8)
            else
                reg_k_values=()
                for ((i = 1; i <= MAX_REG_K; i++)); do
                    reg_k_values+=($i)
                done
            fi

            for REG_K_SPLIT_BLOCK in "${reg_k_values[@]}"; do
                echo -e "\n========================================"
                echo "Testing configuration:"
                echo "M_BLOCK=$M_BLOCK"
                echo "NUM_RANK_BLOCK=$NUM_RANK_BLOCK"
                echo "REG_K_SPLIT_BLOCK=$REG_K_SPLIT_BLOCK"

                rm -f $OUTPUT

                nvcc -arch=sm_120 -O3 -Xptxas -O3 \
                    --ptxas-options=-v --ptxas-options=-warn-lmem-usage \
                    --ptxas-options=-warn-spills -lineinfo \
                    -DM_BLOCK=$M_BLOCK \
                    -DN_BLOCK=$M_BLOCK \
                    -DK_BLOCK=$M_BLOCK \
                    -DNUM_RANK_BLOCK=$NUM_RANK_BLOCK \
                    -DREG_K_SPLIT_BLOCK=$REG_K_SPLIT_BLOCK \
                    -o "$OUTPUT" "$SOURCE"

                if [ -f "$OUTPUT" ]; then
                    echo "[SUCCESS] Compilation succeeded. Running program..."
                    ./$OUTPUT
                else
                    echo -e "[ERROR] Compilation failed for:"
                    echo "M_BLOCK=$M_BLOCK NUM_RANK_BLOCK=$NUM_RANK_BLOCK REG_K_SPLIT_BLOCK=$REG_K_SPLIT_BLOCK"
                fi

                sleep 0.1
            done
        done
    done

    echo -e "\nAll test combinations completed!"
}

echo "" >${log_file}.log

reg_smem_test >&${log_file}.log
echo "dimension,M_BLOCK,N_BLOCK,K_BLOCK,NUM_RANK_BLOCK,NUM_ALLOC_RANK_BLOCK,k_block,REG_K_SPLIT_BLOCK,SMEM_K_SPLIT_BLOCK,time,Tflops" >${log_file}.csv 
grep '\[hemeng_log\]' ${log_file}.log | sed 's/\[hemeng_log\],//' >> ${log_file}.csv
