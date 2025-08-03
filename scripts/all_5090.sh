start=$(date +%s)

bash fp16_block_square_5090.sh >> all_5090.log 2>&1
echo "fp16_block_square_5090.sh finished" >> all_5090.log 2>&1

bash fp16_block_vs_tflops_5090.sh >> all_5090.log 2>&1
echo "fp16_block_vs_tflops_5090.sh finished" >> all_5090.log 2>&1

bash fp16_cycle_use_5090.sh >> all_5090.log 2>&1
echo "fp16_cycle_use_5090.sh finished" >> all_5090.log 2>&1

bash fp16_reg_smem_5090.sh >> all_5090.log 2>&1
echo "fp16_reg_smem_5090.sh finished" >> all_5090.log 2>&1

bash fp16_reg_use.sh >> all_5090.log 2>&1
echo "fp16_reg_use.sh finished" >> all_5090.log 2>&1

end=$(date +%s)
duration=$((end - start))

minutes=$((duration / 60))
seconds=$((duration % 60))

echo "All scripts finished. Total time: ${minutes} minutes ${seconds} seconds" >> all_5090.log 2>&1