start=$(date +%s)

bash fp16_block_lowrank_GH200.sh > all_gh200.log 2>&1
echo "fp16_block_lowrank_GH200.sh finished" >> all_gh200.log 2>&1
bash fp16_block_square_GH200.sh >> all_gh200.log 2>&1
echo "fp16_block_square_GH200.sh finished" >> all_gh200.log 2>&1
bash fp16_cycle_use_GH200.sh >> all_gh200.log 2>&1
echo "fp16_cycle_use_GH200.sh finished" >> all_gh200.log 2>&1
bash fp16_sparse_GH200.sh >> all_gh200.log 2>&1
echo "fp16_sparse_GH200.sh finished" >> all_gh200.log 2>&1
bash fp64_block_square_GH200.sh >> all_gh200.log 2>&1
echo "fp64_block_square_GH200.sh finished" >> all_gh200.log 2>&1
bash fp64_batched_GH200.sh >> all_gh200.log 2>&1
echo "fp64_batched_GH200.sh finished" >> all_gh200.log 2>&1

end=$(date +%s)
duration=$((end - start))

minutes=$((duration / 60))
seconds=$((duration % 60))

echo "All scripts finished. Total time: ${minutes} minutes ${seconds} seconds" >> all_gh200.log 2>&1