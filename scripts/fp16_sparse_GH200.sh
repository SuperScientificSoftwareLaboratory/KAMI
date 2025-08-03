spgemm_1d() {
    M_BLOCKS=(16 32 64 128 192)

    NUM_RANK_BLOCKS=(1 2 3 4 6 8 9 16)

    for m_block in "${M_BLOCKS[@]}"; do
        for num_rank in "${NUM_RANK_BLOCKS[@]}"; do

            echo "Testing M_BLOCK=${m_block}, NUM_RANK_BLOCK=${num_rank}"

            nvcc -arch=sm_90a -O3 -Xptxas -O3 --ptxas-options=-v -lineinfo \
                -DM_BLOCK=${m_block} -DN_BLOCK=${m_block} -DK_BLOCK=${m_block} -DNUM_RANK_BLOCK=${num_rank} \
                -o ../src/sparse/spgemm/1d/block_spgemm_1d_half_mma ../src/sparse/spgemm/1d/block_spgemm_1d_half_mma.cu

            if [ $? -eq 0 ]; then
                echo "Compilation successful, running executable..."
                ../src/sparse/spgemm/1d/block_spgemm_1d_half_mma
            else
                echo "Compilation failed for M_BLOCK=${m_block}, NUM_RANK_BLOCK=${num_active_rank}, NUM_ALLOC_RANK_BLOCK=${num_rank}, skipping..."
            fi

        done
    done
}

spgemm_2d() {
    M_BLOCKS=(16 32 64 128 192)

    NUM_RANK_BLOCKS=(1 2 3 4)

    for m_block in "${M_BLOCKS[@]}"; do
        for num_rank in "${NUM_RANK_BLOCKS[@]}"; do

            echo "Testing M_BLOCK=${m_block}, NUM_RANK_BLOCK=${num_rank}"

            nvcc -arch=sm_90a -O3 -Xptxas -O3 --ptxas-options=-v -lineinfo \
                -DM_BLOCK=${m_block} -DN_BLOCK=${m_block} -DK_BLOCK=${m_block} -DNUM_RANK_BLOCK=${num_rank} \
                -o ../src/sparse/spgemm/2d/block_spgemm_2d_half_mma ../src/sparse/spgemm/2d/block_spgemm_2d_half_mma.cu

            if [ $? -eq 0 ]; then
                echo "Compilation successful, running executable..."
                ../src/sparse/spgemm/2d/block_spgemm_2d_half_mma
            else
                echo "Compilation failed for M_BLOCK=${m_block}, NUM_RANK_BLOCK=${num_active_rank}, NUM_ALLOC_RANK_BLOCK=${num_rank}, skipping..."
            fi

        done
    done
}

spgemm_3d() {
    M_BLOCKS=(16 32 64 128 192)

    NUM_RANK_BLOCKS=(1 2)

    for m_block in "${M_BLOCKS[@]}"; do
        for num_rank in "${NUM_RANK_BLOCKS[@]}"; do

            echo "Testing M_BLOCK=${m_block}, NUM_RANK_BLOCK=${num_rank}"

            nvcc -arch=sm_90a -O3 -Xptxas -O3 --ptxas-options=-v -lineinfo \
                -DM_BLOCK=${m_block} -DN_BLOCK=${m_block} -DK_BLOCK=${m_block} -DNUM_RANK_BLOCK=${num_rank} \
                -o ../src/sparse/spgemm/3d/block_spgemm_3d_half_mma ../src/sparse/spgemm/3d/block_spgemm_3d_half_mma.cu

            if [ $? -eq 0 ]; then
                echo "Compilation successful, running executable..."
                ../src/sparse/spgemm/3d/block_spgemm_3d_half_mma
            else
                echo "Compilation failed for M_BLOCK=${m_block}, NUM_RANK_BLOCK=${num_active_rank}, NUM_ALLOC_RANK_BLOCK=${num_rank}, skipping..."
            fi

        done
    done
}

spmm_1d() {
    M_BLOCKS=(16 32 64 128 192)

    NUM_RANK_BLOCKS=(1 2 3 4 6 8 9 16)

    for m_block in "${M_BLOCKS[@]}"; do
        for num_rank in "${NUM_RANK_BLOCKS[@]}"; do

            echo "Testing M_BLOCK=${m_block}, NUM_RANK_BLOCK=${num_rank}"

            nvcc -arch=sm_90a -O3 -Xptxas -O3 --ptxas-options=-v -lineinfo \
                -DM_BLOCK=${m_block} -DN_BLOCK=${m_block} -DK_BLOCK=${m_block} -DNUM_RANK_BLOCK=${num_rank} \
                -o ../src/sparse/spmm/1d/block_spmm_1d_half_mma ../src/sparse/spmm/1d/block_spmm_1d_half_mma.cu

            if [ $? -eq 0 ]; then
                echo "Compilation successful, running executable..."
                ../src/sparse/spmm/1d/block_spmm_1d_half_mma
            else
                echo "Compilation failed for M_BLOCK=${m_block}, NUM_RANK_BLOCK=${num_active_rank}, NUM_ALLOC_RANK_BLOCK=${num_rank}, skipping..."
            fi

        done
    done
}

spmm_2d() {
    M_BLOCKS=(16 32 64 128 192)

    NUM_RANK_BLOCKS=(1 2 3 4)

    for m_block in "${M_BLOCKS[@]}"; do
        for num_rank in "${NUM_RANK_BLOCKS[@]}"; do

            echo "Testing M_BLOCK=${m_block}, NUM_RANK_BLOCK=${num_rank}"

            nvcc -arch=sm_90a -O3 -Xptxas -O3 --ptxas-options=-v -lineinfo \
                -DM_BLOCK=${m_block} -DN_BLOCK=${m_block} -DK_BLOCK=${m_block} -DNUM_RANK_BLOCK=${num_rank} \
                -o ../src/sparse/spmm/2d/block_spmm_2d_half_mma ../src/sparse/spmm/2d/block_spmm_2d_half_mma.cu

            if [ $? -eq 0 ]; then
                echo "Compilation successful, running executable..."
                ../src/sparse/spmm/2d/block_spmm_2d_half_mma
            else
                echo "Compilation failed for M_BLOCK=${m_block}, NUM_RANK_BLOCK=${num_active_rank}, NUM_ALLOC_RANK_BLOCK=${num_rank}, skipping..."
            fi

        done
    done
}

spmm_3d() {
    M_BLOCKS=(16 32 64 128 192)

    NUM_RANK_BLOCKS=(1 2)

    for m_block in "${M_BLOCKS[@]}"; do
        for num_rank in "${NUM_RANK_BLOCKS[@]}"; do

            echo "Testing M_BLOCK=${m_block}, NUM_RANK_BLOCK=${num_rank}"

            nvcc -arch=sm_90a -O3 -Xptxas -O3 --ptxas-options=-v -lineinfo \
                -DM_BLOCK=${m_block} -DN_BLOCK=${m_block} -DK_BLOCK=${m_block} -DNUM_RANK_BLOCK=${num_rank} \
                -o ../src/sparse/spmm/3d/block_spmm_3d_half_mma ../src/sparse/spmm/3d/block_spmm_3d_half_mma.cu

            if [ $? -eq 0 ]; then
                echo "Compilation successful, running executable..."
                ../src/sparse/spmm/3d/block_spmm_3d_half_mma
            else
                echo "Compilation failed for M_BLOCK=${m_block}, NUM_RANK_BLOCK=${num_active_rank}, NUM_ALLOC_RANK_BLOCK=${num_rank}, skipping..."
            fi

        done
    done
}

spmm_1d >&../logs/sparse/gh200_spmm_fp16_1d.log
grep '\[hemeng_log\]' ../logs/sparse/gh200_spmm_fp16_1d.log | sed 's/\[hemeng_log\],//' >../logs/sparse/gh200_spmm_fp16_1d.csv
spmm_2d >&../logs/sparse/gh200_spmm_fp16_2d.log
grep '\[hemeng_log\]' ../logs/sparse/gh200_spmm_fp16_2d.log | sed 's/\[hemeng_log\],//' >../logs/sparse/gh200_spmm_fp16_2d.csv
spmm_3d >&../logs/sparse/gh200_spmm_fp16_3d.log
grep '\[hemeng_log\]' ../logs/sparse/gh200_spmm_fp16_3d.log | sed 's/\[hemeng_log\],//' >../logs/sparse/gh200_spmm_fp16_3d.csv

spgemm_1d >&../logs/sparse/gh200_spgemm_1d_fp16.log
grep '\[hemeng_log\]' ../logs/sparse/gh200_spgemm_1d_fp16.log | sed 's/\[hemeng_log\],//' >../logs/sparse/gh200_spgemm_1d_fp16.csv
spgemm_2d >&../logs/sparse/gh200_spgemm_2d_fp16.log
grep '\[hemeng_log\]' ../logs/sparse/gh200_spgemm_2d_fp16.log | sed 's/\[hemeng_log\],//' >../logs/sparse/gh200_spgemm_2d_fp16.csv
spgemm_3d >&../logs/sparse/gh200_spgemm_3d_fp16.log
grep '\[hemeng_log\]' ../logs/sparse/gh200_spgemm_3d_fp16.log | sed 's/\[hemeng_log\],//' >../logs/sparse/gh200_spgemm_3d_fp16.csv
