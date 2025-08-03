
#include <mma.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

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

#define NUM_RANK_ALL_BLOCK (NUM_RANK_BLOCK * NUM_RANK_BLOCK)

#define WMMA_M_BLOCK 16
#define WMMA_N_BLOCK 8
#define WMMA_K_BLOCK 16

#define SUB_M_BLOCK (M_BLOCK / NUM_RANK_BLOCK)
#define SUB_N_BLOCK (N_BLOCK / NUM_RANK_BLOCK)
#define SUB_K_BLOCK (K_BLOCK / NUM_RANK_BLOCK)

#ifndef BSR_M
#define BSR_M 16
#define BSR_K 16
#define BSR_N 16
#endif

#define BSR_M_BLOCK (M_BLOCK / BSR_M)
#define BSR_N_BLOCK (N_BLOCK / BSR_N)
#define BSR_K_BLOCK (K_BLOCK / BSR_K)

#define SUB_BSR_M_BLOCK (BSR_M_BLOCK / NUM_RANK_BLOCK)
#define SUB_BSR_N_BLOCK (BSR_N_BLOCK / NUM_RANK_BLOCK)
#define SUB_BSR_K_BLOCK (BSR_K_BLOCK / NUM_RANK_BLOCK)

const int shmem_size_block_val = (M_BLOCK * SUB_K_BLOCK + K_BLOCK * SUB_N_BLOCK) * sizeof(half);
const int shmem_size_block_row_ptr = ((SUB_BSR_M_BLOCK + 1) + (SUB_BSR_K_BLOCK + 1)) * NUM_RANK_BLOCK * sizeof(int);
const int shmem_size_block_col_idx = (SUB_BSR_M_BLOCK * SUB_BSR_K_BLOCK + SUB_BSR_K_BLOCK * SUB_BSR_N_BLOCK) * NUM_RANK_BLOCK * sizeof(int);
const int shmem_size_block_spgemm = shmem_size_block_val + shmem_size_block_row_ptr + shmem_size_block_col_idx;

#define NUM_ITER 1000

#define THREADS_PER_BLOCK (NUM_RANK_ALL_BLOCK * WARP_SIZE)

#define BLOCKS_PER_GRID 16384

__global__ void spgemm_symbolic_kernel(
    const int *RowPtrA, const int *ColIdxA,
    const int *RowPtrB, const int *ColIdxB,
    int *RowPtrC, int m, int k, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m)
    {
        char d_dense_row_column_flag[N_BLOCK / BSR_N] = {0};
        for (int i = RowPtrA[row]; i < RowPtrA[row + 1]; i++)
        {
            int col_a = ColIdxA[i];
            for (int j = RowPtrB[col_a]; j < RowPtrB[col_a + 1]; j++)
            {
                int key = ColIdxB[j];
                d_dense_row_column_flag[key] = 1;
            }
        }
        int nnzr = 0;
        for (int cid = 0; cid < N_BLOCK / BSR_N; cid++)
        {
            if (d_dense_row_column_flag[cid] == 1)
            {
                nnzr++;
            }
        }
        RowPtrC[row] = nnzr;
    }
}

__global__ void spgemm_symbolic_col_kernel(
    const int *RowPtrA, const int *ColIdxA,
    const int *RowPtrB, const int *ColIdxB,
    const int *RowPtrC, int *ColIdxC, int m, int k, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m)
    {
        char d_dense_row_column_flag[N_BLOCK / BSR_N] = {0};
        for (int i = RowPtrA[row]; i < RowPtrA[row + 1]; i++)
        {
            int col_a = ColIdxA[i];
            for (int j = RowPtrB[col_a]; j < RowPtrB[col_a + 1]; j++)
            {
                int key = ColIdxB[j];
                d_dense_row_column_flag[key] = 1;
            }
        }
        int nnzr = 0;
        int c_rpt = RowPtrC[row];
        for (int cid = 0; cid < N_BLOCK / BSR_N; cid++)
        {
            if (d_dense_row_column_flag[cid] == 1)
            {
                ColIdxC[c_rpt + nnzr++] = cid;
            }
        }
    }
}

template <typename T>
void exclusive_scan(T *input, int length)
{
    if (length == 0 || length == 1)
        return;

    T old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i - 1];
        old_val = new_val;
    }
}

__device__ void exclusive_scan_device(int *input, int length)
{
    if (length == 0 || length == 1)
        return;

    int old_val, new_val;

    old_val = input[0];
    input[0] = 0;
    for (int i = 1; i < length; i++)
    {
        new_val = input[i];
        input[i] = old_val + input[i - 1];
        old_val = new_val;
    }
}

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

__global__ void block_spgemm_2d_half_mma(const int *RowPtrA, const int *ColIdxA, const half *ValA,
                                         const int *RowPtrB, const int *ColIdxB, const half *ValB,
                                         const int *RowPtrC, const int *ColIdxC, half *ValC,
                                         const half alpha, const half beta)
{
    extern __shared__ __align__(16) half shmem[];

    half *shmem_a = shmem;
    half *shmem_b = shmem + M_BLOCK * SUB_K_BLOCK;

    int *shmem_a_row_ptr = (int *)(shmem + shmem_size_block_val / sizeof(half));
    int *shmem_b_row_ptr = shmem_a_row_ptr + (SUB_BSR_M_BLOCK + 1) * NUM_RANK_BLOCK;
    int *shmem_a_col_idx = shmem_b_row_ptr + (SUB_BSR_K_BLOCK + 1) * NUM_RANK_BLOCK;
    int *shmem_b_col_idx = shmem_a_col_idx + (SUB_BSR_M_BLOCK * SUB_BSR_K_BLOCK) * NUM_RANK_BLOCK;

    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31;

    const unsigned int warp_row = warp_id / NUM_RANK_BLOCK;
    const unsigned int warp_col = warp_id % NUM_RANK_BLOCK;

    int row_ptr_a_local[SUB_BSR_M_BLOCK + 1] = {0};
    int col_idx_a_local[SUB_BSR_M_BLOCK * SUB_BSR_K_BLOCK] = {0};
    int row_ptr_b_local[SUB_BSR_K_BLOCK + 1] = {0};
    int col_idx_b_local[SUB_BSR_K_BLOCK * SUB_BSR_N_BLOCK] = {0};
    int row_ptr_c_local[SUB_BSR_M_BLOCK + 1] = {0};
    int col_idx_c_local[SUB_BSR_M_BLOCK * SUB_BSR_N_BLOCK] = {0};

    int row_ptr_a_local_compute[SUB_BSR_M_BLOCK + 1] = {0};
    int col_idx_a_local_compute[SUB_BSR_M_BLOCK * SUB_BSR_K_BLOCK] = {0};
    int row_ptr_b_local_compute[SUB_BSR_K_BLOCK + 1] = {0};
    int col_idx_b_local_compute[SUB_BSR_K_BLOCK * SUB_BSR_N_BLOCK] = {0};

    uint32_t a_frags[SUB_BSR_M_BLOCK * SUB_BSR_K_BLOCK][BSR_M / WMMA_M_BLOCK][BSR_K / WMMA_K_BLOCK][4];
    uint32_t a_frags_compute[SUB_BSR_M_BLOCK * SUB_BSR_K_BLOCK][BSR_M / WMMA_M_BLOCK][BSR_K / WMMA_K_BLOCK][4];
    uint32_t b_frags[SUB_BSR_K_BLOCK * SUB_BSR_N_BLOCK][BSR_K / WMMA_K_BLOCK][BSR_N / WMMA_N_BLOCK][2];
    uint32_t b_frags_compute[SUB_BSR_K_BLOCK * SUB_BSR_N_BLOCK][BSR_K / WMMA_K_BLOCK][BSR_N / WMMA_N_BLOCK][2];
    uint32_t c_frags[SUB_BSR_M_BLOCK * SUB_BSR_N_BLOCK][BSR_M / WMMA_M_BLOCK][BSR_N / WMMA_N_BLOCK][2];

    {
        int row_start_bsr = warp_row * SUB_BSR_M_BLOCK;
        int row_end_bsr = row_start_bsr + SUB_BSR_M_BLOCK;
        int col_start_bsr = warp_col * SUB_BSR_K_BLOCK;
        int col_end_bsr = col_start_bsr + SUB_BSR_K_BLOCK;

        int ele_id_local = 0;

        for (int row_id = row_start_bsr; row_id < row_end_bsr; row_id++)
        {
            for (int ele_id = RowPtrA[row_id]; ele_id < RowPtrA[row_id + 1]; ele_id++)
            {
                int col_id = ColIdxA[ele_id];
                if (col_id >= col_start_bsr && col_id < col_end_bsr)
                {
                    int val_offset = ele_id * BSR_M * BSR_K;

                    int row_id_bsr_local = row_id - row_start_bsr;
                    int col_id_bsr_local = col_id - col_start_bsr;

                    row_ptr_a_local[row_id_bsr_local]++;
                    col_idx_a_local[ele_id_local] = col_id_bsr_local;

                    for (int i = 0; i < BSR_M / WMMA_M_BLOCK; i++)
                    {
                        for (int j = 0; j < BSR_K / WMMA_K_BLOCK; j++)
                        {
                            int lane_row_upper_offset = lane_id >> 2;
                            int lane_row_lower_offset = lane_row_upper_offset + 8;
                            int lane_col_left_offset = (lane_id % 4) * 2;
                            int lane_col_right_offset = lane_col_left_offset + 8;

                            a_frags[ele_id_local][i][j][0] = *(uint32_t *)(ValA + val_offset + (i * WMMA_M_BLOCK + lane_row_upper_offset) * BSR_K + lane_col_left_offset + j * WMMA_K_BLOCK);
                            a_frags[ele_id_local][i][j][1] = *(uint32_t *)(ValA + val_offset + (i * WMMA_M_BLOCK + lane_row_lower_offset) * BSR_K + lane_col_left_offset + j * WMMA_K_BLOCK);
                            a_frags[ele_id_local][i][j][2] = *(uint32_t *)(ValA + val_offset + (i * WMMA_M_BLOCK + lane_row_upper_offset) * BSR_K + lane_col_right_offset + j * WMMA_K_BLOCK);
                            a_frags[ele_id_local][i][j][3] = *(uint32_t *)(ValA + val_offset + (i * WMMA_M_BLOCK + lane_row_lower_offset) * BSR_K + lane_col_right_offset + j * WMMA_K_BLOCK);
                        }
                    }
                    ele_id_local++;
                }
            }
        }
        exclusive_scan_device(row_ptr_a_local, SUB_BSR_M_BLOCK + 1);
    }

    {
        int row_start_bsr = warp_row * SUB_BSR_K_BLOCK;
        int row_end_bsr = row_start_bsr + SUB_BSR_K_BLOCK;
        int col_start_bsr = warp_col * SUB_BSR_N_BLOCK;
        int col_end_bsr = col_start_bsr + SUB_BSR_N_BLOCK;

        int ele_id_local = 0;

        for (int row_id = row_start_bsr; row_id < row_end_bsr; row_id++)
        {
            for (int ele_id = RowPtrB[row_id]; ele_id < RowPtrB[row_id + 1]; ele_id++)
            {
                int col_id = ColIdxB[ele_id];
                if (col_id >= col_start_bsr && col_id < col_end_bsr)
                {
                    int val_offset = ele_id * BSR_K * BSR_N;

                    int row_id_bsr_local = row_id - row_start_bsr;
                    int col_id_bsr_local = col_id - col_start_bsr;

                    row_ptr_b_local[row_id_bsr_local]++;
                    col_idx_b_local[ele_id_local] = col_id_bsr_local;

                    for (int i = 0; i < BSR_K / WMMA_K_BLOCK; i++)
                    {
                        for (int j = 0; j < BSR_N / WMMA_N_BLOCK; j++)
                        {
                            int lane_row_upper_offset = (lane_id % 4) * 2;
                            int lane_row_lower_offset = lane_row_upper_offset + 8;
                            int lane_col_offset = lane_id >> 2;

                            half2 temp;
                            temp.x = *(ValB + val_offset + ((i * WMMA_K_BLOCK + lane_row_upper_offset) + 0) * BSR_N + lane_col_offset + j * WMMA_N_BLOCK);
                            temp.y = *(ValB + val_offset + ((i * WMMA_K_BLOCK + lane_row_upper_offset) + 1) * BSR_N + lane_col_offset + j * WMMA_N_BLOCK);

                            b_frags[ele_id_local][i][j][0] = *(uint32_t *)&temp;

                            temp.x = *(ValB + val_offset + ((i * WMMA_K_BLOCK + lane_row_lower_offset) + 0) * BSR_N + lane_col_offset + j * WMMA_N_BLOCK);
                            temp.y = *(ValB + val_offset + ((i * WMMA_K_BLOCK + lane_row_lower_offset) + 1) * BSR_N + lane_col_offset + j * WMMA_N_BLOCK);

                            b_frags[ele_id_local][i][j][1] = *(uint32_t *)&temp;
                        }
                    }

                    ele_id_local++;
                }
            }
        }

        exclusive_scan_device(row_ptr_b_local, SUB_BSR_K_BLOCK + 1);
    }

    {
        int row_start_bsr = warp_row * SUB_BSR_M_BLOCK;
        int row_end_bsr = row_start_bsr + SUB_BSR_M_BLOCK;
        int col_start_bsr = warp_col * SUB_BSR_N_BLOCK;
        int col_end_bsr = col_start_bsr + SUB_BSR_N_BLOCK;

        int ele_id_local = 0;

        for (int row_id = row_start_bsr; row_id < row_end_bsr; row_id++)
        {
            for (int ele_id = RowPtrC[row_id]; ele_id < RowPtrC[row_id + 1]; ele_id++)
            {
                int col_id = ColIdxC[ele_id];
                if (col_id >= col_start_bsr && col_id < col_end_bsr)
                {
                    int val_offset = ele_id * BSR_M * BSR_N;

                    int row_id_bsr_local = row_id - row_start_bsr;
                    int col_id_bsr_local = col_id - col_start_bsr;

                    row_ptr_c_local[row_id_bsr_local]++;
                    col_idx_c_local[ele_id_local] = col_id_bsr_local;

                    for (int i = 0; i < BSR_M / WMMA_M_BLOCK; i++)
                    {
                        for (int j = 0; j < BSR_N / WMMA_N_BLOCK; j++)
                        {
                            int lane_row_upper_offset = lane_id >> 2;
                            int lane_row_lower_offset = lane_row_upper_offset + 8;
                            int lane_col_offset = (lane_id % 4) * 2;

                            c_frags[ele_id_local][i][j][0] = *(uint32_t *)(ValC + val_offset + (i * BSR_N + lane_row_upper_offset) * BSR_N + lane_col_offset + j * WMMA_N_BLOCK);
                            c_frags[ele_id_local][i][j][1] = *(uint32_t *)(ValC + val_offset + (i * BSR_N + lane_row_lower_offset) * BSR_N + lane_col_offset + j * WMMA_N_BLOCK);

                            c_frags[ele_id_local][i][j][0] = 0;
                            c_frags[ele_id_local][i][j][1] = 0;
                        }
                    }
                    ele_id_local++;
                }
            }
        }

        exclusive_scan_device(row_ptr_c_local, SUB_BSR_M_BLOCK + 1);
    }

    __syncthreads();
    for (int idx_iter = 0; idx_iter < NUM_ITER; idx_iter++)
    {

        for (int idx_stage = 0; idx_stage < NUM_RANK_BLOCK; idx_stage++)
        {

            if (warp_col == idx_stage)
            {
                for (int i = 0; i < SUB_BSR_M_BLOCK; i++)
                {
                    for (int j = 0; j < SUB_BSR_K_BLOCK; j++)
                    {
                        int shmem_offset = (i * (SUB_BSR_K_BLOCK) + j) * BSR_M * BSR_K;

                        for (int k = 0; k < BSR_M / WMMA_M_BLOCK; k++)
                        {
                            for (int l = 0; l < BSR_K / WMMA_K_BLOCK; l++)
                            {
                                *(uint32_t *)(shmem_a + shmem_offset + lane_id * 8 + 0 + (k * (BSR_K / WMMA_K_BLOCK) + l) * WMMA_M_BLOCK * WMMA_K_BLOCK + warp_row * ((SUB_M_BLOCK) * (SUB_K_BLOCK))) = a_frags[i * SUB_BSR_K_BLOCK + j][k][l][0];
                                *(uint32_t *)(shmem_a + shmem_offset + lane_id * 8 + 2 + (k * (BSR_K / WMMA_K_BLOCK) + l) * WMMA_M_BLOCK * WMMA_K_BLOCK + warp_row * ((SUB_M_BLOCK) * (SUB_K_BLOCK))) = a_frags[i * SUB_BSR_K_BLOCK + j][k][l][1];
                                *(uint32_t *)(shmem_a + shmem_offset + lane_id * 8 + 4 + (k * (BSR_K / WMMA_K_BLOCK) + l) * WMMA_M_BLOCK * WMMA_K_BLOCK + warp_row * ((SUB_M_BLOCK) * (SUB_K_BLOCK))) = a_frags[i * SUB_BSR_K_BLOCK + j][k][l][2];
                                *(uint32_t *)(shmem_a + shmem_offset + lane_id * 8 + 6 + (k * (BSR_K / WMMA_K_BLOCK) + l) * WMMA_M_BLOCK * WMMA_K_BLOCK + warp_row * ((SUB_M_BLOCK) * (SUB_K_BLOCK))) = a_frags[i * SUB_BSR_K_BLOCK + j][k][l][3];
                            }
                        }
                    }
                }

                for (int i = 0; i < SUB_BSR_M_BLOCK + 1; i++)
                {
                    shmem_a_row_ptr[warp_row * (SUB_BSR_M_BLOCK + 1) + i] = row_ptr_a_local[i];
                }

                for (int i = 0; i < SUB_BSR_M_BLOCK * SUB_BSR_K_BLOCK; i++)
                {
                    shmem_a_col_idx[warp_row * (SUB_BSR_M_BLOCK * SUB_BSR_K_BLOCK) + i] = col_idx_a_local[i];
                }
            }

            if (warp_row == idx_stage)
            {
                for (int i = 0; i < SUB_BSR_K_BLOCK; i++)
                {
                    for (int j = 0; j < SUB_BSR_N_BLOCK; j++)
                    {
                        int shmem_offset = (i * (SUB_BSR_N_BLOCK) + j) * BSR_K * BSR_N;

                        for (int k = 0; k < BSR_K / WMMA_K_BLOCK; k++)
                        {
                            for (int l = 0; l < BSR_N / WMMA_N_BLOCK; l++)
                            {
                                *(uint32_t *)(shmem_b + shmem_offset + lane_id * 4 + 0 + (k * (BSR_N / WMMA_N_BLOCK) + l) * WMMA_K_BLOCK * WMMA_N_BLOCK + warp_col * ((SUB_K_BLOCK) * (SUB_N_BLOCK))) = b_frags[i * SUB_BSR_N_BLOCK + j][k][l][0];
                                *(uint32_t *)(shmem_b + shmem_offset + lane_id * 4 + 2 + (k * (BSR_N / WMMA_N_BLOCK) + l) * WMMA_K_BLOCK * WMMA_N_BLOCK + warp_col * ((SUB_K_BLOCK) * (SUB_N_BLOCK))) = b_frags[i * SUB_BSR_N_BLOCK + j][k][l][1];
                            }
                        }
                    }
                }

                for (int i = 0; i < SUB_BSR_K_BLOCK + 1; i++)
                {
                    shmem_b_row_ptr[warp_col * (SUB_BSR_K_BLOCK + 1) + i] = row_ptr_b_local[i];
                }

                for (int i = 0; i < SUB_BSR_K_BLOCK * SUB_BSR_N_BLOCK; i++)
                {
                    shmem_b_col_idx[warp_col * (SUB_BSR_K_BLOCK * SUB_BSR_N_BLOCK) + i] = col_idx_b_local[i];
                }
            }

            __syncthreads();

            for (int i = 0; i < SUB_BSR_M_BLOCK; i++)
            {
                for (int j = 0; j < SUB_BSR_K_BLOCK; j++)
                {
                    int shmem_offset = (i * (SUB_BSR_K_BLOCK) + j) * BSR_M * BSR_K;

                    for (int k = 0; k < BSR_M / WMMA_M_BLOCK; k++)
                    {
                        for (int l = 0; l < BSR_K / WMMA_K_BLOCK; l++)
                        {
                            a_frags_compute[i * SUB_BSR_K_BLOCK + j][k][l][0] = *(uint32_t *)(shmem_a + shmem_offset + lane_id * 8 + 0 + (k * (BSR_K / WMMA_K_BLOCK) + l) * WMMA_M_BLOCK * WMMA_K_BLOCK + warp_row * ((SUB_M_BLOCK) * (SUB_K_BLOCK)));
                            a_frags_compute[i * SUB_BSR_K_BLOCK + j][k][l][1] = *(uint32_t *)(shmem_a + shmem_offset + lane_id * 8 + 2 + (k * (BSR_K / WMMA_K_BLOCK) + l) * WMMA_M_BLOCK * WMMA_K_BLOCK + warp_row * ((SUB_M_BLOCK) * (SUB_K_BLOCK)));
                            a_frags_compute[i * SUB_BSR_K_BLOCK + j][k][l][2] = *(uint32_t *)(shmem_a + shmem_offset + lane_id * 8 + 4 + (k * (BSR_K / WMMA_K_BLOCK) + l) * WMMA_M_BLOCK * WMMA_K_BLOCK + warp_row * ((SUB_M_BLOCK) * (SUB_K_BLOCK)));
                            a_frags_compute[i * SUB_BSR_K_BLOCK + j][k][l][3] = *(uint32_t *)(shmem_a + shmem_offset + lane_id * 8 + 6 + (k * (BSR_K / WMMA_K_BLOCK) + l) * WMMA_M_BLOCK * WMMA_K_BLOCK + warp_row * ((SUB_M_BLOCK) * (SUB_K_BLOCK)));
                        }
                    }
                }
            }

            for (int i = 0; i < SUB_BSR_M_BLOCK + 1; i++)
            {
                row_ptr_a_local_compute[i] = shmem_a_row_ptr[warp_row * (SUB_BSR_M_BLOCK + 1) + i];
            }

            for (int i = 0; i < SUB_BSR_M_BLOCK * SUB_BSR_K_BLOCK; i++)
            {
                col_idx_a_local_compute[i] = shmem_a_col_idx[warp_row * (SUB_BSR_M_BLOCK * SUB_BSR_K_BLOCK) + i];
            }

            for (int i = 0; i < SUB_BSR_K_BLOCK; i++)
            {
                for (int j = 0; j < SUB_BSR_N_BLOCK; j++)
                {
                    int shmem_offset = (i * (SUB_BSR_N_BLOCK) + j) * BSR_K * BSR_N;

                    for (int k = 0; k < BSR_K / WMMA_K_BLOCK; k++)
                    {
                        for (int l = 0; l < BSR_N / WMMA_N_BLOCK; l++)
                        {
                            b_frags_compute[i * SUB_BSR_N_BLOCK + j][k][l][0] = *(uint32_t *)(shmem_b + shmem_offset + lane_id * 4 + 0 + (k * (BSR_N / WMMA_N_BLOCK) + l) * WMMA_K_BLOCK * WMMA_N_BLOCK + warp_col * ((SUB_K_BLOCK) * (SUB_N_BLOCK)));
                            b_frags_compute[i * SUB_BSR_N_BLOCK + j][k][l][1] = *(uint32_t *)(shmem_b + shmem_offset + lane_id * 4 + 2 + (k * (BSR_N / WMMA_N_BLOCK) + l) * WMMA_K_BLOCK * WMMA_N_BLOCK + warp_col * ((SUB_K_BLOCK) * (SUB_N_BLOCK)));
                        }
                    }
                }
            }

            for (int i = 0; i < SUB_BSR_K_BLOCK + 1; i++)
            {
                row_ptr_b_local_compute[i] = shmem_b_row_ptr[warp_col * (SUB_BSR_K_BLOCK + 1) + i];
            }

            for (int i = 0; i < SUB_BSR_K_BLOCK * SUB_BSR_N_BLOCK; i++)
            {
                col_idx_b_local_compute[i] = shmem_b_col_idx[warp_col * (SUB_BSR_K_BLOCK * SUB_BSR_N_BLOCK) + i];
            }

            __syncthreads();

            {

                for (int row_a_idx = 0; row_a_idx < SUB_BSR_M_BLOCK; row_a_idx++)
                {
                    for (int ele_a_idx = row_ptr_a_local_compute[row_a_idx]; ele_a_idx < row_ptr_a_local_compute[row_a_idx + 1]; ele_a_idx++)
                    {
                        int row_b_idx = col_idx_a_local_compute[ele_a_idx];

                        for (int ele_b_idx = row_ptr_b_local_compute[row_b_idx]; ele_b_idx < row_ptr_b_local_compute[row_b_idx + 1]; ele_b_idx++)
                        {
                            int col_c_idx = col_idx_b_local_compute[ele_b_idx];

                            for (int ele_c_idx = row_ptr_c_local[row_a_idx]; ele_c_idx < row_ptr_c_local[row_a_idx + 1]; ele_c_idx++)
                            {
                                if (col_idx_c_local[ele_c_idx] == col_c_idx)
                                {
                                    for (int i = 0; i < BSR_M / WMMA_M_BLOCK; i++)
                                    {
                                        for (int j = 0; j < BSR_N / WMMA_N_BLOCK; j++)
                                        {
                                            for (int k = 0; k < BSR_K / WMMA_K_BLOCK; k++)
                                            {
                                                mma_m16n8k16_fp16(c_frags[ele_c_idx][i][j], a_frags_compute[ele_a_idx][i][k], b_frags_compute[ele_b_idx][k][j]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    {
        int row_start_bsr = warp_row * SUB_BSR_M_BLOCK;
        int row_end_bsr = row_start_bsr + SUB_BSR_M_BLOCK;
        int col_start_bsr = warp_col * SUB_BSR_N_BLOCK;
        int col_end_bsr = col_start_bsr + SUB_BSR_N_BLOCK;

        int ele_id_local = 0;

        for (int row_id = row_start_bsr; row_id < row_end_bsr; row_id++)
        {
            for (int ele_id = RowPtrC[row_id]; ele_id < RowPtrC[row_id + 1]; ele_id++)
            {
                int col_id = ColIdxC[ele_id];
                if (col_id >= col_start_bsr && col_id < col_end_bsr)
                {
                    int val_offset = ele_id * BSR_M * BSR_N;

                    int row_id_bsr_local = row_id - row_start_bsr;
                    int col_id_bsr_local = col_id - col_start_bsr;

                    for (int i = 0; i < BSR_M / WMMA_M_BLOCK; i++)
                    {
                        for (int j = 0; j < BSR_N / WMMA_N_BLOCK; j++)
                        {
                            int lane_row_upper_offset = lane_id >> 2;
                            int lane_row_lower_offset = lane_row_upper_offset + 8;
                            int lane_col_offset = (lane_id % 4) * 2;

                            *(uint32_t *)(ValC + val_offset + (i * BSR_N + lane_row_upper_offset) * BSR_N + lane_col_offset + j * WMMA_N_BLOCK) = c_frags[ele_id_local][i][j][0];
                            *(uint32_t *)(ValC + val_offset + (i * BSR_N + lane_row_lower_offset) * BSR_N + lane_col_offset + j * WMMA_N_BLOCK) = c_frags[ele_id_local][i][j][1];
                        }
                    }

                    ele_id_local++;
                }
            }
        }
    }
}

void spgemm_spa(const int *RowPtrA, const int *ColIdxA, const half *ValA, const int mA, const int nA, const int nnzA,
                const int *RowPtrB, const int *ColIdxB, const half *ValB, const int mB, const int nB, const int nnzB,
                int *RowPtrC, int *ColIdxC, half *ValC, const int mC, const int nC, int *nnzC, const int get_nnzC_only)
{
    if (get_nnzC_only == 1)
    {
#pragma omp parallel for
        for (int iid = 0; iid < mA; iid++)
        {
            char *d_dense_row_column_flag = (char *)malloc((nB) * sizeof(char));
            memset(d_dense_row_column_flag, 0, (nB) * sizeof(char));
            for (int i = RowPtrA[iid]; i < RowPtrA[iid + 1]; i++)
            {
                int col = ColIdxA[i];
                for (int l = RowPtrB[col]; l < RowPtrB[col + 1]; l++)
                {
                    const int key = ColIdxB[l];
                    d_dense_row_column_flag[key] = 1;
                }
            }
            int nnzr = 0;
            for (int cid = 0; cid < nB; cid++)
            {
                if (d_dense_row_column_flag[cid] == 1)
                {
                    nnzr++;
                }
            }
            RowPtrC[iid] = nnzr;
            free(d_dense_row_column_flag);
        }
        exclusive_scan<int>(RowPtrC, mC + 1);
        *nnzC = RowPtrC[mC] * BSR_M * BSR_N;
    }
    else
    {
#pragma omp parallel for
        for (int iid = 0; iid < mA; iid++)
        {
            char *d_dense_row_column_flag = (char *)malloc((nB) * sizeof(char));
            half *d_dense_row_value = (half *)malloc((BSR_M * BSR_N * nB) * sizeof(half));

            memset(d_dense_row_column_flag, 0, (nB) * sizeof(char));
            memset(d_dense_row_value, 0, (BSR_M * BSR_N * nB) * sizeof(half));

            for (int i = RowPtrA[iid]; i < RowPtrA[iid + 1]; i++)
            {
                int col = ColIdxA[i];
                const half *a = &ValA[i * BSR_M * BSR_K];
                for (int l = RowPtrB[col]; l < RowPtrB[col + 1]; l++)
                {

                    const int key = ColIdxB[l];
                    d_dense_row_column_flag[key] = 1;
                    const half *b = &ValB[l * BSR_K * BSR_N];
                    for (int ii = 0; ii < BSR_M; ii++)
                    {
                        for (int jj = 0; jj < BSR_N; jj++)
                        {
                            for (int kk = 0; kk < BSR_K; kk++)
                            {
                                d_dense_row_value[key * BSR_M * BSR_N + ii * BSR_N + jj] += (double)a[ii * BSR_K + kk] * (double)b[kk * BSR_N + jj];
                            }
                        }
                    }
                }
            }

            int nnzr = RowPtrC[iid];

            for (int cid = 0; cid < nB; cid++)
            {
                if (d_dense_row_column_flag[cid] == 1)
                {
                    for (int i = 0; i < BSR_M * BSR_N; i++)
                    {
                        ValC[nnzr * BSR_M * BSR_N + i] = d_dense_row_value[cid * BSR_M * BSR_N + i];
                    }

                    ColIdxC[nnzr] = cid;

                    nnzr++;
                }
            }

            free(d_dense_row_column_flag);
            free(d_dense_row_value);
        }
    }
}

void generate_random_bsr_matrix(int rows, int cols, float sparsity, int *row_ptr, int *col_idx, half *values, int *nnz)
{
    int i, j;
    int count = 0;
    float r;
    srand(0);
    row_ptr[0] = 0;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            r = (float)rand() / (float)RAND_MAX;
            if (r > sparsity)
            {
                col_idx[count] = j;
                for (int len = 0; len < BSR_M * BSR_K; len++)
                {
                    values[count * BSR_M * BSR_K + len] = rand() % 3 + 1.0;
                }
                count++;
            }
        }
        row_ptr[i + 1] = count;
    }
    *nnz = count * BSR_M * BSR_K;
}

void print_bcsr(int *row_ptr, int *col_idx, half *values, int rows, int cols)
{
    int i, j, k;
    for (i = 0; i < rows; i++)
    {
        for (j = row_ptr[i]; j < row_ptr[i + 1]; j++)
        {
            printf("Row %d, Col %d: ", i, col_idx[j]);
            for (k = 0; k < BSR_M * BSR_K; k++)
            {
                printf("%10.f ", (float)values[j * BSR_M * BSR_K + k]);
            }
            printf("\n");
        }
    }

    printf("BSR Bitmap:\n");

    int bitmap[rows][cols];

    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++)
            bitmap[i][j] = 0;

    for (i = 0; i < rows; i++)
    {
        for (j = row_ptr[i]; j < row_ptr[i + 1]; j++)
        {
            int br = i;
            int bc = col_idx[j];
            bitmap[br][bc] = 1;
        }
    }

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            printf("%d ", bitmap[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "GPU " << prop.pciDeviceID << " Model: " << prop.name << std::endl;
    std::cout << "GPU Model: " << prop.name << std::endl;

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = BLOCKS_PER_GRID;

    float sparsity = 0.5;
    int nnzA = 0;
    int max_nnzA = (M_BLOCK / BSR_M) * (K_BLOCK / BSR_K);
    int *RowPtrA = (int *)malloc((M_BLOCK / BSR_M + 1) * sizeof(int));
    int *ColIdxA = (int *)malloc(max_nnzA * sizeof(int));
    half *ValA = (half *)malloc(max_nnzA * BSR_M * BSR_K * sizeof(half));
    generate_random_bsr_matrix(M_BLOCK / BSR_M, K_BLOCK / BSR_K, sparsity, RowPtrA, ColIdxA, ValA, &nnzA);

    int nnzB = 0;
    int max_nnzB = (K_BLOCK / BSR_K) * (N_BLOCK / BSR_N);
    int *RowPtrB = (int *)malloc((K_BLOCK / BSR_K + 1) * sizeof(int));
    int *ColIdxB = (int *)malloc(max_nnzB * sizeof(int));
    half *ValB = (half *)malloc(max_nnzB * BSR_K * BSR_N * sizeof(half));
    generate_random_bsr_matrix(K_BLOCK / BSR_K, N_BLOCK / BSR_N, sparsity, RowPtrB, ColIdxB, ValB, &nnzB);

    int nnzC = 0;
    int *RowPtrC = (int *)malloc((M_BLOCK / BSR_M + 1) * sizeof(int));
    int *ColIdxC;
    half *ValC;
    spgemm_spa(RowPtrA, ColIdxA, ValA, M_BLOCK / BSR_M, K_BLOCK / BSR_K, nnzA,
               RowPtrB, ColIdxB, ValB, K_BLOCK / BSR_K, N_BLOCK / BSR_N, nnzB,
               RowPtrC, ColIdxC, ValC, M_BLOCK / BSR_M, N_BLOCK / BSR_N, &nnzC, 1);

    printf("nnzA = %d, nnzB = %d, nnzC = %d\n", nnzA / BSR_M / BSR_K, nnzB / BSR_K / BSR_N, nnzC / BSR_M / BSR_N);
    ColIdxC = (int *)malloc(nnzC / (BSR_M * BSR_N) * sizeof(int));
    ValC = (half *)malloc(nnzC * sizeof(half));

    spgemm_spa(RowPtrA, ColIdxA, ValA, M_BLOCK / BSR_M, K_BLOCK / BSR_K, nnzA,
               RowPtrB, ColIdxB, ValB, K_BLOCK / BSR_K, N_BLOCK / BSR_N, nnzB,
               RowPtrC, ColIdxC, ValC, M_BLOCK / BSR_M, N_BLOCK / BSR_N, &nnzC, 0);

    unsigned long long int nnzCub = 0;
    for (int i = 0; i < nnzA / (BSR_M * BSR_K); i++)
    {
        int rowB = ColIdxA[i];
        nnzCub += RowPtrB[rowB + 1] - RowPtrB[rowB];
    }

    int *d_RowPtrA, *d_ColIdxA, *d_RowPtrB, *d_ColIdxB;
    half *d_ValA, *d_ValB;

    cudaMalloc(&d_RowPtrA, (M_BLOCK / BSR_M + 1) * sizeof(int));
    cudaMalloc(&d_ColIdxA, nnzA / (BSR_M * BSR_K) * sizeof(int));
    cudaMalloc(&d_ValA, nnzA * sizeof(half));

    cudaMalloc(&d_RowPtrB, (M_BLOCK / BSR_M + 1) * sizeof(int));
    cudaMalloc(&d_ColIdxB, nnzB / (BSR_K * BSR_N) * sizeof(int));
    cudaMalloc(&d_ValB, nnzB * sizeof(half));

    cudaMemcpy(d_RowPtrA, RowPtrA, (M_BLOCK / BSR_M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ColIdxA, ColIdxA, nnzA / (BSR_M * BSR_K) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ValA, ValA, nnzA * sizeof(half), cudaMemcpyHostToDevice);

    cudaMemcpy(d_RowPtrB, RowPtrB, (K_BLOCK / BSR_K + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ColIdxB, ColIdxB, nnzB / (BSR_K * BSR_N) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ValB, ValB, nnzB * sizeof(half), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaFuncSetAttribute(block_spgemm_2d_half_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size_block_spgemm);
    std::cout << "Launching kernel with " << blocksPerGrid << " blocks and "
              << threadsPerBlock << " threads per block and " << shmem_size_block_spgemm << " bytes of shared memory" << std::endl;

    int *d_RowPtrC, *d_ColIdxC;
    half *d_ValC;
    cudaMalloc(&d_RowPtrC, (M_BLOCK / BSR_M + 1) * sizeof(int));
    cudaMemset(d_RowPtrC, 0, (M_BLOCK / BSR_M + 1) * sizeof(int));

    cudaEventRecord(start);
    int block_size = threadsPerBlock;
    int grid_size = (M_BLOCK / BSR_M + block_size - 1) / block_size;
    spgemm_symbolic_kernel<<<grid_size, block_size>>>(d_RowPtrA, d_ColIdxA, d_RowPtrB, d_ColIdxB, d_RowPtrC, M_BLOCK / BSR_M, K_BLOCK / BSR_K, N_BLOCK / BSR_N);
    thrust::exclusive_scan(thrust::device, d_RowPtrC, d_RowPtrC + M_BLOCK / BSR_M + 1, d_RowPtrC);
    int d_nnzC = 0;
    cudaMemcpy(&d_nnzC, d_RowPtrC + M_BLOCK / BSR_M, sizeof(int), cudaMemcpyDeviceToHost);
    d_nnzC = d_nnzC * BSR_M * BSR_N;
    cudaMalloc(&d_ColIdxC, d_nnzC / (BSR_M * BSR_N) * sizeof(int));
    cudaMalloc(&d_ValC, d_nnzC * sizeof(half));
    spgemm_symbolic_col_kernel<<<grid_size, block_size>>>(d_RowPtrA, d_ColIdxA, d_RowPtrB, d_ColIdxB, d_RowPtrC, d_ColIdxC, M_BLOCK / BSR_M, K_BLOCK / BSR_K, N_BLOCK / BSR_N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float symbolic_time = 0;
    cudaEventElapsedTime(&symbolic_time, start, stop);
    std::cout << "Symbolic time: " << symbolic_time << " ms" << std::endl;

    cudaEventRecord(start);
    block_spgemm_2d_half_mma<<<blocksPerGrid, threadsPerBlock, shmem_size_block_spgemm>>>(d_RowPtrA, d_ColIdxA, d_ValA,
                                                                                          d_RowPtrB, d_ColIdxB, d_ValB,
                                                                                          d_RowPtrC, d_ColIdxC, d_ValC,
                                                                                          1.0, 0.0);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    long long numOpsPerMatrix = 2LL * nnzCub * (BSR_M * BSR_K * BSR_N);
    long long totalOps = numOpsPerMatrix * blocksPerGrid * NUM_ITER;
    double gflops = static_cast<double>(totalOps) / (milliseconds * 1e6);
    double tflops = gflops / 1000.0f;

    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS (" << tflops << " TFLOPS)" << std::endl;
    std::cout << "[hemeng_log],2d," << nnzCub << "," << M_BLOCK << "," << N_BLOCK << "," << K_BLOCK << "," << NUM_RANK_ALL_BLOCK << "," << tflops << "," << THREADS_PER_BLOCK << std::endl;

    half *h_ValC = (half *)malloc(nnzC * sizeof(half));
    cudaMemcpy(h_ValC, d_ValC, nnzC * sizeof(half), cudaMemcpyDeviceToHost);

    int error = 0;
    for (int i = 0; i < nnzC; i++)
    {
        if (fabs((double)h_ValC[i] / NUM_ITER - (double)ValC[i]) > 1e-6)
        {

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