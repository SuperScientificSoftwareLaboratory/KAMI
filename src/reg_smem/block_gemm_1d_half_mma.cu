#include <mma.h>
#include <cuda_runtime.h>
#include <iostream>

#define WARP_SIZE 32

using namespace nvcuda;

#ifndef M_BLOCK
#define M_BLOCK 128
#define N_BLOCK 128
#define K_BLOCK 128
#endif

#ifndef NUM_RANK_BLOCK
#define NUM_RANK_BLOCK 2
#endif

#ifndef NUM_ALLOC_RANK_BLOCK
#define NUM_ALLOC_RANK_BLOCK NUM_RANK_BLOCK
#endif

#define WMMA_M_BLOCK 16
#define WMMA_N_BLOCK 8
#define WMMA_K_BLOCK 16

#define SUB_M_BLOCK (M_BLOCK / NUM_RANK_BLOCK)
#define SUB_N_BLOCK (N_BLOCK / NUM_RANK_BLOCK)

#define K_SPLIT_BLOCK (K_BLOCK / WMMA_K_BLOCK)

#ifndef REG_K_SPLIT_BLOCK
#define REG_K_SPLIT_BLOCK 6
#endif

#define SMEM_K_SPLIT_BLOCK (K_SPLIT_BLOCK - REG_K_SPLIT_BLOCK)

#define REG_K_BLOCK (K_BLOCK / K_SPLIT_BLOCK * REG_K_SPLIT_BLOCK)
#define SMEM_K_BLOCK (K_BLOCK / K_SPLIT_BLOCK * SMEM_K_SPLIT_BLOCK)

#define NUM_PIPE_K_BLOCK 1

const int shmem_size_communication_block = SUB_N_BLOCK * REG_K_BLOCK / NUM_PIPE_K_BLOCK;
const int shmem_size_cache_a_block = M_BLOCK * SMEM_K_BLOCK;
const int shmem_size_cache_b_block = N_BLOCK * SMEM_K_BLOCK;
const int shmem_size_block_gemm = (shmem_size_communication_block + shmem_size_cache_a_block + shmem_size_cache_b_block) * sizeof(half);

#define NUM_ITER 1000

#define THREADS_PER_BLOCK (NUM_RANK_BLOCK * WARP_SIZE)

#define BLOCKS_PER_GRID 16384

__device__ __forceinline__ void mma_m16n8k16_fp16(uint32_t *acc, uint32_t *frag_a, uint32_t *frag_b)
{
    uint32_t const *A = reinterpret_cast<uint32_t const *>(&frag_a[0]);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_b[0]);
    uint32_t *C = reinterpret_cast<uint32_t *>(&acc[0]);

    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16"
        " { %0, %1 }, "
        " { %2, %3, %4, %5 }, "
        " { %6, %7 }, "
        " { %8, %9 };"
        : "=r"(C[0]), "=r"(C[1])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]));
}

__global__ void block_gemm_1d_half_mma(const half *A, const half *B, half *C, const half alpha, const half beta)
{
    extern __shared__ half shmem_b[];

    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31;

    if (warp_id >= NUM_RANK_BLOCK)
    {
        return;
    }

    uint32_t(*shmem_a_cache)[SMEM_K_BLOCK / WMMA_K_BLOCK][WMMA_M_BLOCK * WMMA_K_BLOCK / 2] =
        reinterpret_cast<uint32_t(*)[SMEM_K_BLOCK / WMMA_K_BLOCK][WMMA_M_BLOCK * WMMA_K_BLOCK / 2]>(shmem_b + shmem_size_communication_block + warp_id * SUB_M_BLOCK * SMEM_K_BLOCK);
    uint32_t(*shmem_b_cache)[SMEM_K_BLOCK / WMMA_K_BLOCK][WMMA_N_BLOCK * WMMA_K_BLOCK / 2] =
        reinterpret_cast<uint32_t(*)[SMEM_K_BLOCK / WMMA_K_BLOCK][WMMA_N_BLOCK * WMMA_K_BLOCK / 2]>(shmem_b + shmem_size_communication_block + shmem_size_cache_a_block + warp_id * SUB_N_BLOCK * SMEM_K_BLOCK);

    uint32_t a_frags[SUB_M_BLOCK / WMMA_M_BLOCK][REG_K_BLOCK / WMMA_K_BLOCK][4];

    uint32_t b_frags[SUB_N_BLOCK / WMMA_N_BLOCK][REG_K_BLOCK / WMMA_K_BLOCK][2];
    uint32_t b_frags_compute[SUB_N_BLOCK / WMMA_N_BLOCK][REG_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK][2];

    uint32_t c_frags[SUB_M_BLOCK / WMMA_M_BLOCK][N_BLOCK / WMMA_N_BLOCK][2];

    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
    {
        for (int j = 0; j < REG_K_BLOCK / WMMA_K_BLOCK; ++j)
        {
            int group_id = lane_id >> 2;
            int thread_in_group = lane_id % 4;
            int row_upper = group_id;
            int row_lower = group_id + 8;
            int col_left = thread_in_group * 2;
            int col_right = thread_in_group * 2 + 8;

            int global_offset_upper_left = warp_id * (SUB_M_BLOCK * K_BLOCK) + (i * WMMA_M_BLOCK + row_upper) * K_BLOCK + j * WMMA_K_BLOCK + col_left;
            int global_offset_lower_left = warp_id * (SUB_M_BLOCK * K_BLOCK) + (i * WMMA_M_BLOCK + row_lower) * K_BLOCK + j * WMMA_K_BLOCK + col_left;

            int global_offset_upper_right = warp_id * (SUB_M_BLOCK * K_BLOCK) + (i * WMMA_M_BLOCK + row_upper) * K_BLOCK + j * WMMA_K_BLOCK + col_right;
            int global_offset_lower_right = warp_id * (SUB_M_BLOCK * K_BLOCK) + (i * WMMA_M_BLOCK + row_lower) * K_BLOCK + j * WMMA_K_BLOCK + col_right;

            a_frags[i][j][0] = *(uint32_t *)(&(A[global_offset_upper_left]));
            a_frags[i][j][1] = *(uint32_t *)(&(A[global_offset_lower_left]));
            a_frags[i][j][2] = *(uint32_t *)(&(A[global_offset_upper_right]));
            a_frags[i][j][3] = *(uint32_t *)(&(A[global_offset_lower_right]));
        }
    }

    for (int i = 0; i < SUB_N_BLOCK / WMMA_N_BLOCK; ++i)
    {
        for (int j = 0; j < REG_K_BLOCK / WMMA_K_BLOCK; ++j)
        {
            int row = lane_id >> 2;
            int col_left = (lane_id % 4) * 2;
            int col_right = (lane_id % 4) * 2 + 8;
            int global_offset_left = warp_id * (SUB_N_BLOCK * K_BLOCK) + (i * WMMA_N_BLOCK + row) * K_BLOCK + j * WMMA_K_BLOCK + col_left;
            int global_offset_right = warp_id * (SUB_N_BLOCK * K_BLOCK) + (i * WMMA_N_BLOCK + row) * K_BLOCK + j * WMMA_K_BLOCK + col_right;

            b_frags[i][j][0] = *(uint32_t *)(&(B[global_offset_left]));
            b_frags[i][j][1] = *(uint32_t *)(&(B[global_offset_right]));
        }
    }

    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
    {
        for (int j = 0; j < SMEM_K_BLOCK / WMMA_K_BLOCK; ++j)
        {
            int group_id = lane_id >> 2;
            int thread_in_group = lane_id % 4;
            int row_upper = group_id;
            int row_lower = group_id + 8;
            int col_left = thread_in_group * 2;
            int col_right = thread_in_group * 2 + 8;

            int global_offset_upper_left = warp_id * (SUB_M_BLOCK * K_BLOCK) + (i * WMMA_M_BLOCK + row_upper) * K_BLOCK + (j + REG_K_BLOCK / WMMA_K_BLOCK) * WMMA_K_BLOCK + col_left;
            int global_offset_lower_left = warp_id * (SUB_M_BLOCK * K_BLOCK) + (i * WMMA_M_BLOCK + row_lower) * K_BLOCK + (j + REG_K_BLOCK / WMMA_K_BLOCK) * WMMA_K_BLOCK + col_left;

            int global_offset_upper_right = warp_id * (SUB_M_BLOCK * K_BLOCK) + (i * WMMA_M_BLOCK + row_upper) * K_BLOCK + (j + REG_K_BLOCK / WMMA_K_BLOCK) * WMMA_K_BLOCK + col_right;
            int global_offset_lower_right = warp_id * (SUB_M_BLOCK * K_BLOCK) + (i * WMMA_M_BLOCK + row_lower) * K_BLOCK + (j + REG_K_BLOCK / WMMA_K_BLOCK) * WMMA_K_BLOCK + col_right;

            shmem_a_cache[i][j][lane_id * 4 + 0] = *(uint32_t *)(&(A[global_offset_upper_left]));
            shmem_a_cache[i][j][lane_id * 4 + 1] = *(uint32_t *)(&(A[global_offset_lower_left]));
            shmem_a_cache[i][j][lane_id * 4 + 2] = *(uint32_t *)(&(A[global_offset_upper_right]));
            shmem_a_cache[i][j][lane_id * 4 + 3] = *(uint32_t *)(&(A[global_offset_lower_right]));
        }
    }

    for (int i = 0; i < SUB_N_BLOCK / WMMA_N_BLOCK; ++i)
    {
        for (int j = 0; j < SMEM_K_BLOCK / WMMA_K_BLOCK; ++j)
        {
            int row = lane_id >> 2;
            int col_left = (lane_id % 4) * 2;
            int col_right = (lane_id % 4) * 2 + 8;
            int global_offset_left = warp_id * (SUB_N_BLOCK * K_BLOCK) + (i * WMMA_N_BLOCK + row) * K_BLOCK + (j + REG_K_BLOCK / WMMA_K_BLOCK) * WMMA_K_BLOCK + col_left;
            int global_offset_right = warp_id * (SUB_N_BLOCK * K_BLOCK) + (i * WMMA_N_BLOCK + row) * K_BLOCK + (j + REG_K_BLOCK / WMMA_K_BLOCK) * WMMA_K_BLOCK + col_right;

            shmem_b_cache[i][j][lane_id * 2 + 0] = *(uint32_t *)(&(B[global_offset_left]));
            shmem_b_cache[i][j][lane_id * 2 + 1] = *(uint32_t *)(&(B[global_offset_right]));
        }
    }

    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
    {
        for (int j = 0; j < N_BLOCK / WMMA_N_BLOCK; ++j)
        {
            c_frags[i][j][0] = 0.0;
            c_frags[i][j][1] = 0.0;
        }
    }

    __syncthreads();

    for (int idx_iter = 0; idx_iter < NUM_ITER; ++idx_iter)
    {
        // use reg data
        for (int idx_stage = 0; idx_stage < NUM_RANK_BLOCK; ++idx_stage)
        {
            for (int idx_pipe = 0; idx_pipe < NUM_PIPE_K_BLOCK; ++idx_pipe)
            {
                if (warp_id == idx_stage)
                {
                    for (int i = 0; i < SUB_N_BLOCK / WMMA_N_BLOCK; i++)
                    {
                        for (int j = 0; j < REG_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK; j++)
                        {
                            int real_j = j + idx_pipe * (REG_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK);

                            *(uint32_t *)(shmem_b + lane_id * 4 + (i * (REG_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK) + j) * WMMA_N_BLOCK * WMMA_K_BLOCK) = b_frags[i][real_j][0];
                            *(uint32_t *)(shmem_b + lane_id * 4 + (i * (REG_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK) + j) * WMMA_N_BLOCK * WMMA_K_BLOCK + 2) = b_frags[i][real_j][1];
                        }
                    }
                }

                __syncthreads();

                for (int i = 0; i < SUB_N_BLOCK / WMMA_N_BLOCK; i++)
                {
                    for (int j = 0; j < REG_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK; j++)
                    {
                        b_frags_compute[i][j][0] = *(uint32_t *)(shmem_b + lane_id * 4 + (i * (REG_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK) + j) * WMMA_N_BLOCK * WMMA_K_BLOCK);
                        b_frags_compute[i][j][1] = *(uint32_t *)(shmem_b + lane_id * 4 + (i * (REG_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK) + j) * WMMA_N_BLOCK * WMMA_K_BLOCK + 2);
                    }
                }

                __syncthreads();

                for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
                {
                    for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; ++j)
                    {
                        for (int k = 0; k < REG_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK; ++k)
                        {
                            int real_k = k + idx_pipe * (REG_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK);
                            mma_m16n8k16_fp16(c_frags[i][j + idx_stage * (SUB_N_BLOCK / WMMA_N_BLOCK)], a_frags[i][real_k], b_frags_compute[j][k]);
                        }
                    }
                }
            }
        }

        __syncthreads();

        for (int idx_s2r_stage = 0; idx_s2r_stage < SMEM_K_BLOCK / REG_K_BLOCK; idx_s2r_stage++)
        {
            int s2r_k_offset = idx_s2r_stage * (REG_K_BLOCK / WMMA_K_BLOCK);

            // move shared cached data to reg
            for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
            {
                for (int j = 0; j < REG_K_BLOCK / WMMA_K_BLOCK; ++j)
                {
                    a_frags[i][j][0] = shmem_a_cache[i][s2r_k_offset + j][0 + lane_id * 4];
                    a_frags[i][j][1] = shmem_a_cache[i][s2r_k_offset + j][1 + lane_id * 4];
                    a_frags[i][j][2] = shmem_a_cache[i][s2r_k_offset + j][2 + lane_id * 4];
                    a_frags[i][j][3] = shmem_a_cache[i][s2r_k_offset + j][3 + lane_id * 4];
                }
            }

            for (int i = 0; i < SUB_N_BLOCK / WMMA_N_BLOCK; i++)
            {
                for (int j = 0; j < REG_K_BLOCK / WMMA_K_BLOCK; j++)
                {
                    b_frags[i][j][0] = shmem_b_cache[i][s2r_k_offset + j][0 + lane_id * 2];
                    b_frags[i][j][1] = shmem_b_cache[i][s2r_k_offset + j][1 + lane_id * 2];
                }
            }

            __syncthreads();

            // use reg data
            for (int idx_stage = 0; idx_stage < NUM_RANK_BLOCK; ++idx_stage)
            {
                for (int idx_pipe = 0; idx_pipe < NUM_PIPE_K_BLOCK; ++idx_pipe)
                {
                    if (warp_id == idx_stage)
                    {
                        for (int i = 0; i < SUB_N_BLOCK / WMMA_N_BLOCK; i++)
                        {
                            for (int j = 0; j < REG_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK; j++)
                            {
                                int real_j = j + idx_pipe * (REG_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK);

                                *(uint32_t *)(shmem_b + lane_id * 4 + (i * (REG_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK) + j) * WMMA_N_BLOCK * WMMA_K_BLOCK) = b_frags[i][real_j][0];
                                *(uint32_t *)(shmem_b + lane_id * 4 + (i * (REG_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK) + j) * WMMA_N_BLOCK * WMMA_K_BLOCK + 2) = b_frags[i][real_j][1];
                            }
                        }
                    }

                    __syncthreads();

                    for (int i = 0; i < SUB_N_BLOCK / WMMA_N_BLOCK; i++)
                    {
                        for (int j = 0; j < REG_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK; j++)
                        {
                            b_frags_compute[i][j][0] = *(uint32_t *)(shmem_b + lane_id * 4 + (i * (REG_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK) + j) * WMMA_N_BLOCK * WMMA_K_BLOCK);
                            b_frags_compute[i][j][1] = *(uint32_t *)(shmem_b + lane_id * 4 + (i * (REG_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK) + j) * WMMA_N_BLOCK * WMMA_K_BLOCK + 2);
                        }
                    }

                    __syncthreads();

                    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
                    {
                        for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; ++j)
                        {
                            for (int k = 0; k < REG_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK; ++k)
                            {
                                int real_k = k + idx_pipe * (REG_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK);
                                mma_m16n8k16_fp16(c_frags[i][j + idx_stage * (SUB_N_BLOCK / WMMA_N_BLOCK)], a_frags[i][real_k], b_frags_compute[j][k]);
                            }
                        }
                    }
                }
            }
        }

        // tail process
        if (SMEM_K_BLOCK % REG_K_BLOCK != 0)
        {
            const int s2r_k_offset = (SMEM_K_BLOCK / REG_K_BLOCK) * (REG_K_BLOCK / WMMA_K_BLOCK);
            // move shared cached data to reg
            for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
            {
                for (int j = 0; j < (SMEM_K_BLOCK - SMEM_K_BLOCK / REG_K_BLOCK * REG_K_BLOCK) / WMMA_K_BLOCK; ++j)
                {
                    a_frags[i][j][0] = shmem_a_cache[i][s2r_k_offset + j][0 + lane_id * 4];
                    a_frags[i][j][1] = shmem_a_cache[i][s2r_k_offset + j][1 + lane_id * 4];
                    a_frags[i][j][2] = shmem_a_cache[i][s2r_k_offset + j][2 + lane_id * 4];
                    a_frags[i][j][3] = shmem_a_cache[i][s2r_k_offset + j][3 + lane_id * 4];
                }
            }

            for (int i = 0; i < SUB_N_BLOCK / WMMA_N_BLOCK; i++)
            {
                for (int j = 0; j < (SMEM_K_BLOCK - SMEM_K_BLOCK / REG_K_BLOCK * REG_K_BLOCK) / WMMA_K_BLOCK; j++)
                {
                    b_frags[i][j][0] = shmem_b_cache[i][s2r_k_offset + j][0 + lane_id * 2];
                    b_frags[i][j][1] = shmem_b_cache[i][s2r_k_offset + j][1 + lane_id * 2];
                }
            }

            __syncthreads();

            // use reg data
            for (int idx_stage = 0; idx_stage < NUM_RANK_BLOCK; ++idx_stage)
            {
                for (int idx_pipe = 0; idx_pipe < NUM_PIPE_K_BLOCK; ++idx_pipe)
                {
                    if (warp_id == idx_stage)
                    {
                        for (int i = 0; i < SUB_N_BLOCK / WMMA_N_BLOCK; i++)
                        {
                            for (int j = 0; j < (SMEM_K_BLOCK - SMEM_K_BLOCK / REG_K_BLOCK * REG_K_BLOCK) / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK; j++)
                            {
                                int real_j = j + idx_pipe * ((SMEM_K_BLOCK - SMEM_K_BLOCK / REG_K_BLOCK * REG_K_BLOCK) / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK);

                                *(uint32_t *)(shmem_b + lane_id * 4 + (i * ((SMEM_K_BLOCK - SMEM_K_BLOCK / REG_K_BLOCK * REG_K_BLOCK) / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK) + j) * WMMA_N_BLOCK * WMMA_K_BLOCK) = b_frags[i][real_j][0];
                                *(uint32_t *)(shmem_b + lane_id * 4 + (i * ((SMEM_K_BLOCK - SMEM_K_BLOCK / REG_K_BLOCK * REG_K_BLOCK) / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK) + j) * WMMA_N_BLOCK * WMMA_K_BLOCK + 2) = b_frags[i][real_j][1];
                            }
                        }
                    }

                    __syncthreads();

                    for (int i = 0; i < SUB_N_BLOCK / WMMA_N_BLOCK; i++)
                    {
                        for (int j = 0; j < (SMEM_K_BLOCK - SMEM_K_BLOCK / REG_K_BLOCK * REG_K_BLOCK) / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK; j++)
                        {
                            b_frags_compute[i][j][0] = *(uint32_t *)(shmem_b + lane_id * 4 + (i * ((SMEM_K_BLOCK - SMEM_K_BLOCK / REG_K_BLOCK * REG_K_BLOCK) / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK) + j) * WMMA_N_BLOCK * WMMA_K_BLOCK);
                            b_frags_compute[i][j][1] = *(uint32_t *)(shmem_b + lane_id * 4 + (i * ((SMEM_K_BLOCK - SMEM_K_BLOCK / REG_K_BLOCK * REG_K_BLOCK) / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK) + j) * WMMA_N_BLOCK * WMMA_K_BLOCK + 2);
                        }
                    }

                    __syncthreads();

                    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
                    {
                        for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; ++j)
                        {
                            for (int k = 0; k < (SMEM_K_BLOCK - SMEM_K_BLOCK / REG_K_BLOCK * REG_K_BLOCK) / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK; ++k)
                            {
                                int real_k = k + idx_pipe * ((SMEM_K_BLOCK - SMEM_K_BLOCK / REG_K_BLOCK * REG_K_BLOCK) / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK);
                                mma_m16n8k16_fp16(c_frags[i][j + idx_stage * (SUB_N_BLOCK / WMMA_N_BLOCK)], a_frags[i][real_k], b_frags_compute[j][k]);
                            }
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; i++)
    {
        for (int j = 0; j < N_BLOCK / WMMA_N_BLOCK; j++)
        {
            int block_row = i;
            int block_col = j;

            int group_id = lane_id >> 2;
            int thread_in_group = lane_id % 4;
            int row_upper = group_id;
            int row_lower = group_id + 8;
            int col = thread_in_group * 2;
            int global_row_upper = block_row * WMMA_M_BLOCK + row_upper;
            int global_row_lower = block_row * WMMA_M_BLOCK + row_lower;
            int global_col = block_col * WMMA_N_BLOCK + col;

            int global_offset_upper = global_row_upper * N_BLOCK + global_col + warp_id * (SUB_M_BLOCK * N_BLOCK);
            int global_offset_lower = global_row_lower * N_BLOCK + global_col + warp_id * (SUB_M_BLOCK * N_BLOCK);

            *(uint32_t *)(&(C[global_offset_upper])) = c_frags[i][j][0];
            *(uint32_t *)(&(C[global_offset_lower])) = c_frags[i][j][1];
        }
    }
}

int main(int argc, char *argv[])
{
    int device_id = 0;
    cudaDeviceProp prop;
    cudaSetDevice(device_id);
    cudaGetDeviceProperties(&prop, device_id);
    std::cout << "GPU " << prop.pciDeviceID << " Model: " << prop.name << std::endl;

    half *h_A = (half *)malloc(M_BLOCK * K_BLOCK * sizeof(half));
    half *h_B = (half *)malloc(K_BLOCK * N_BLOCK * sizeof(half));

    for (int i = 0; i < M_BLOCK * K_BLOCK; i++)
    {
        h_A[i] = rand() % 3;
        // h_A[i] = 1;
        // h_A[i] = i;
    }

    for (int i = 0; i < K_BLOCK * N_BLOCK; i++)
    {
        h_B[i] = rand() % 3;
        // h_B[i] = 1;
        // h_B[i] = i;
    }

    // Print matrix B
    // std::cout << "Matrix B:" << std::endl;
    // for (int i = 0; i < K_BLOCK; i++)
    // {
    //     for (int j = 0; j < N_BLOCK; j++)
    //     {
    //         std::cout << (float)h_B[i * K_BLOCK + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    half *d_A, *d_B;
    half *d_C;

    cudaMalloc(&d_A, M_BLOCK * K_BLOCK * sizeof(half));
    cudaMalloc(&d_B, K_BLOCK * N_BLOCK * sizeof(half));
    cudaMalloc(&d_C, M_BLOCK * N_BLOCK * sizeof(half));

    cudaMemcpy(d_A, h_A, M_BLOCK * K_BLOCK * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K_BLOCK * N_BLOCK * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M_BLOCK * N_BLOCK * sizeof(half));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::cout << "Launching kernel with " << BLOCKS_PER_GRID << " blocks and "
              << NUM_ALLOC_RANK_BLOCK * WARP_SIZE << " threads per block and " << shmem_size_block_gemm << " bytes of shared memory" << std::endl;

    cudaFuncSetAttribute(block_gemm_1d_half_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size_block_gemm);

    cudaEventRecord(start);
    block_gemm_1d_half_mma<<<BLOCKS_PER_GRID, NUM_ALLOC_RANK_BLOCK * WARP_SIZE, shmem_size_block_gemm>>>(d_A, d_B, d_C, 1.0, 0.0);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    long long numOpsPerMatrix = 2LL * M_BLOCK * N_BLOCK * K_BLOCK;
    long long totalOps = numOpsPerMatrix * BLOCKS_PER_GRID * NUM_ITER;
    double gflops = static_cast<double>(totalOps) / (milliseconds * 1e6);
    double tflops = gflops / 1000.0f;

    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS (" << tflops << " TFLOPS)" << std::endl;
    std::cout << "[hemeng_log],1d," << M_BLOCK << "," << N_BLOCK << "," << K_BLOCK << "," << NUM_RANK_BLOCK << "," << NUM_ALLOC_RANK_BLOCK <<","<<K_SPLIT_BLOCK<<","<<REG_K_SPLIT_BLOCK<<","<<SMEM_K_SPLIT_BLOCK<<","<< milliseconds << "," << tflops << std::endl;
    // std::cout << "[hemeng_log],1d," << M_BLOCK << "," << NUM_RANK_BLOCK << "," << NUM_ALLOC_RANK_BLOCK << "," << milliseconds << "," << tflops << std::endl;

    half *h_C = (half *)malloc(M_BLOCK * N_BLOCK * sizeof(half));
    cudaMemcpy(h_C, d_C, M_BLOCK * N_BLOCK * sizeof(half), cudaMemcpyDeviceToHost);

    double *h_C_ref = (double *)malloc(M_BLOCK * N_BLOCK * sizeof(double));

    for (int i = 0; i < M_BLOCK; i++)
    {
        for (int j = 0; j < M_BLOCK; j++)
        {
            h_C_ref[i * N_BLOCK + j] = 0;
            for (int k = 0; k < K_BLOCK; k++)
            {
                h_C_ref[i * N_BLOCK + j] += (double)h_A[i * K_BLOCK + k] * (double)h_B[j * K_BLOCK + k];
            }
        }
    }

    // std::cout << "h_C\n";
    // for (int i = 0; i < M_BLOCK; i++)
    // {
    //     for (int j = 0; j < N_BLOCK; j++)
    //     {
    //         std::cout << (double)h_C[i * N_BLOCK + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "h_C_ref\n";
    // for (int i = 0; i < M_BLOCK; i++)
    // {
    //     for (int j = 0; j < N_BLOCK; j++)
    //     {
    //         std::cout << h_C_ref[i * N_BLOCK + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    int error = 0;
    for (int i = 0; i < M_BLOCK * N_BLOCK; i++)
    {
        if (fabs((double)h_C[i] / NUM_ITER - h_C_ref[i]) > 1e-6)
        {
            std::cout << "Error: " << i << " " << (double)h_C[i] << " " << h_C_ref[i] << std::endl;
            error = 1;
            break;
        }
    }

    if (!error)
    {
        std::cout << "Validation successful!" << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
