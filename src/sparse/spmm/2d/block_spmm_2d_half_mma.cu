
#include <mma.h>
#include <cuda_runtime.h>
#include <iostream>

#define WARP_SIZE 32

using namespace nvcuda;

#ifndef BSR_M
#define BSR_M 16
#define BSR_K 16
#endif

#ifndef M_BLOCK
#define M_BLOCK 16
#define N_BLOCK 16
#define K_BLOCK 16
#endif

#ifndef BSR_ROWS
#define BSR_ROWS (M_BLOCK / BSR_M)
#define BSR_COLS (K_BLOCK / BSR_K)
#endif

#ifndef NUM_RANK_BLOCK
#define NUM_RANK_BLOCK 1
#endif

#define NUM_RANK_ALL_BLOCK (NUM_RANK_BLOCK * NUM_RANK_BLOCK)

#define WMMA_M_BLOCK 16
#define WMMA_N_BLOCK 8
#define WMMA_K_BLOCK 16

#define SUB_M_BLOCK (M_BLOCK / NUM_RANK_BLOCK)
#define SUB_N_BLOCK (N_BLOCK / NUM_RANK_BLOCK)
#define SUB_K_BLOCK (K_BLOCK / NUM_RANK_BLOCK)

#define NUM_PIPE_BLOCK 1
#define NUMS_ROW (BSR_ROWS / NUM_RANK_BLOCK)
#define NUMS_COL (BSR_COLS / NUM_RANK_BLOCK / NUM_PIPE_BLOCK)

const int shmem_size_block_spmm = ((M_BLOCK * (SUB_K_BLOCK / NUM_PIPE_BLOCK)) + (K_BLOCK / NUM_PIPE_BLOCK * SUB_N_BLOCK)) * sizeof(half);

#ifndef NUM_ITER
#define NUM_ITER 1000
#endif

#define THREADS_PER_BLOCK (NUM_RANK_ALL_BLOCK * WARP_SIZE)

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

__global__ void block_spmm_2d_half_mma(const int *blk_rowptr, const int *blk_colidx, const half *bsr_vals, const half *B, half *C, const half alpha, const half beta)
{
    extern __shared__ __align__(16) half shmem[];

    half *shmem_a = shmem;
    half *shmem_b = shmem + M_BLOCK * SUB_K_BLOCK / NUM_PIPE_BLOCK;

    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31;

    const unsigned int warp_row = warp_id / NUM_RANK_BLOCK;
    const unsigned int warp_col = warp_id % NUM_RANK_BLOCK;

    uint32_t a_frags[SUB_M_BLOCK / WMMA_M_BLOCK][SUB_K_BLOCK / WMMA_K_BLOCK][4];
    uint32_t b_frags[SUB_N_BLOCK / WMMA_N_BLOCK][SUB_K_BLOCK / WMMA_K_BLOCK][2];

    uint32_t b_frags_compute[SUB_N_BLOCK / WMMA_N_BLOCK][SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK][2];
    uint32_t c_frags[SUB_M_BLOCK / WMMA_M_BLOCK][SUB_N_BLOCK / WMMA_N_BLOCK][2];

    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
    {
        for (int l = 0; l < NUM_PIPE_BLOCK; ++l)
        {
            int row_start = (warp_id * NUM_PIPE_BLOCK + l) * NUMS_ROW;
            for (int j = blk_rowptr[row_start + i]; j < blk_rowptr[row_start + i + 1]; ++j)
            {
                int col = blk_colidx[j];
                const half *A = &bsr_vals[j * BSR_M * BSR_K];
                int group_id = lane_id >> 2;
                int thread_in_group = lane_id % 4;
                int row_upper = group_id;
                int row_lower = group_id + 8;
                int col_left = thread_in_group * 2;
                int col_right = thread_in_group * 2 + 8;

                int global_offset_upper_left = row_upper * WMMA_K_BLOCK + col_left;
                int global_offset_lower_left = row_lower * WMMA_K_BLOCK + col_left;
                int global_offset_upper_right = row_upper * WMMA_K_BLOCK + col_right;
                int global_offset_lower_right = row_lower * WMMA_K_BLOCK + col_right;

                a_frags[i][col + l * NUMS_COL][0] = *(uint32_t *)(&(A[global_offset_upper_left]));
                a_frags[i][col + l * NUMS_COL][1] = *(uint32_t *)(&(A[global_offset_lower_left]));
                a_frags[i][col + l * NUMS_COL][2] = *(uint32_t *)(&(A[global_offset_upper_right]));
                a_frags[i][col + l * NUMS_COL][3] = *(uint32_t *)(&(A[global_offset_lower_right]));
            }
        }
    }

    for (int i = 0; i < SUB_N_BLOCK / WMMA_N_BLOCK; ++i)
    {
        for (int j = 0; j < SUB_K_BLOCK / WMMA_K_BLOCK; ++j)
        {
            int block_row = warp_col * (SUB_N_BLOCK / WMMA_N_BLOCK) + i;
            int block_col = warp_row * (SUB_K_BLOCK / WMMA_K_BLOCK) + j;
            int real_row = block_row * WMMA_N_BLOCK + (lane_id >> 2);
            int real_col_left = block_col * WMMA_K_BLOCK + (lane_id % 4) * 2;
            int real_col_right = real_col_left + 8;
            int global_offset_left = real_row * K_BLOCK + real_col_left;
            int global_offset_right = real_row * K_BLOCK + real_col_right;

            b_frags[i][j][0] = *(uint32_t *)(B + global_offset_left);
            b_frags[i][j][1] = *(uint32_t *)(B + global_offset_right);
        }
    }

    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
    {
        for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; ++j)
        {
            c_frags[i][j][0] = 0.0;
            c_frags[i][j][1] = 0.0;
        }
    }

    __syncthreads();

    for (int idx_iter = 0; idx_iter < NUM_ITER; ++idx_iter)
    {
        for (int idx_stage = 0; idx_stage < NUM_RANK_BLOCK; ++idx_stage)
        {
            for (int idx_pipe = 0; idx_pipe < NUM_PIPE_BLOCK; ++idx_pipe)
            {

                if (warp_col == idx_stage)
                {
                    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; i++)
                    {
                        for (int j = 0; j < SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK; j++)
                        {
                            int real_j = j + idx_pipe * (SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK);
                            *(uint32_t *)(shmem_a + lane_id * 8 + 0 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK) + j) * WMMA_M_BLOCK * WMMA_K_BLOCK + warp_row * ((SUB_M_BLOCK) * (SUB_K_BLOCK / NUM_PIPE_BLOCK))) = a_frags[i][real_j][0];
                            *(uint32_t *)(shmem_a + lane_id * 8 + 2 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK) + j) * WMMA_M_BLOCK * WMMA_K_BLOCK + warp_row * ((SUB_M_BLOCK) * (SUB_K_BLOCK / NUM_PIPE_BLOCK))) = a_frags[i][real_j][1];
                            *(uint32_t *)(shmem_a + lane_id * 8 + 4 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK) + j) * WMMA_M_BLOCK * WMMA_K_BLOCK + warp_row * ((SUB_M_BLOCK) * (SUB_K_BLOCK / NUM_PIPE_BLOCK))) = a_frags[i][real_j][2];
                            *(uint32_t *)(shmem_a + lane_id * 8 + 6 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK) + j) * WMMA_M_BLOCK * WMMA_K_BLOCK + warp_row * ((SUB_M_BLOCK) * (SUB_K_BLOCK / NUM_PIPE_BLOCK))) = a_frags[i][real_j][3];
                        }
                    }
                }

                if (warp_row == idx_stage)
                {
                    for (int i = 0; i < SUB_N_BLOCK / WMMA_N_BLOCK; i++)
                    {
                        for (int j = 0; j < SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK; j++)
                        {
                            int real_j = j + idx_pipe * (SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK);
                            *(uint32_t *)(shmem_b + lane_id * 4 + 0 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK) + j) * WMMA_K_BLOCK * WMMA_N_BLOCK + warp_col * ((SUB_K_BLOCK / NUM_PIPE_BLOCK) * (SUB_N_BLOCK))) = b_frags[i][real_j][0];
                            *(uint32_t *)(shmem_b + lane_id * 4 + 2 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK) + j) * WMMA_K_BLOCK * WMMA_N_BLOCK + warp_col * ((SUB_K_BLOCK / NUM_PIPE_BLOCK) * (SUB_N_BLOCK))) = b_frags[i][real_j][1];
                        }
                    }
                }

                __syncthreads();

                for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; j++)
                {
                    for (int k = 0; k < SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK; k++)
                    {
                        b_frags_compute[j][k][0] = *(uint32_t *)(shmem_b + lane_id * 4 + 0 + (j * (SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK) + k) * WMMA_K_BLOCK * WMMA_N_BLOCK + warp_col * ((SUB_K_BLOCK / NUM_PIPE_BLOCK) * (SUB_N_BLOCK)));
                        b_frags_compute[j][k][1] = *(uint32_t *)(shmem_b + lane_id * 4 + 2 + (j * (SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK) + k) * WMMA_K_BLOCK * WMMA_N_BLOCK + warp_col * ((SUB_K_BLOCK / NUM_PIPE_BLOCK) * (SUB_N_BLOCK)));
                    }
                }

                for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
                {
                    int row_start = warp_row * (BSR_COLS / NUMS_COL) * NUMS_ROW + (idx_stage * NUM_PIPE_BLOCK + idx_pipe) * NUMS_ROW;
                    for (int k = blk_rowptr[row_start + i]; k < blk_rowptr[row_start + i + 1]; ++k)
                    {
                        int col = blk_colidx[k];
                        uint32_t a_frags_compute[4];
                        a_frags_compute[0] = *(uint32_t *)(shmem_a + lane_id * 8 + 0 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK) + col) * WMMA_M_BLOCK * WMMA_K_BLOCK + warp_row * ((SUB_M_BLOCK) * (SUB_K_BLOCK / NUM_PIPE_BLOCK)));
                        a_frags_compute[1] = *(uint32_t *)(shmem_a + lane_id * 8 + 2 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK) + col) * WMMA_M_BLOCK * WMMA_K_BLOCK + warp_row * ((SUB_M_BLOCK) * (SUB_K_BLOCK / NUM_PIPE_BLOCK)));
                        a_frags_compute[2] = *(uint32_t *)(shmem_a + lane_id * 8 + 4 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK) + col) * WMMA_M_BLOCK * WMMA_K_BLOCK + warp_row * ((SUB_M_BLOCK) * (SUB_K_BLOCK / NUM_PIPE_BLOCK)));
                        a_frags_compute[3] = *(uint32_t *)(shmem_a + lane_id * 8 + 6 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK) + col) * WMMA_M_BLOCK * WMMA_K_BLOCK + warp_row * ((SUB_M_BLOCK) * (SUB_K_BLOCK / NUM_PIPE_BLOCK)));

                        for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; ++j)
                        {
                            mma_m16n8k16_fp16(c_frags[i][j], a_frags_compute, b_frags_compute[j][col]);
                        }
                    }
                }
                __syncthreads();
            }
        }
    }

    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; i++)
    {
        for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; j++)
        {
            int block_row = warp_row * (SUB_M_BLOCK / WMMA_M_BLOCK) + i;
            int block_col = warp_col * (SUB_N_BLOCK / WMMA_N_BLOCK) + j;

            int real_row_upper = block_row * WMMA_M_BLOCK + (lane_id >> 2);
            int real_row_lower = real_row_upper + 8;
            int real_col = block_col * WMMA_N_BLOCK + (lane_id % 4) * 2;

            int global_offset_upper = real_row_upper * N_BLOCK + real_col;
            int global_offset_lower = real_row_lower * N_BLOCK + real_col;

            *(uint32_t *)(C + global_offset_upper) = c_frags[i][j][0];
            *(uint32_t *)(C + global_offset_lower) = c_frags[i][j][1];
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
                    values[count * BSR_M * BSR_K + len] = rand() % 2 + 1.0;
                }
                count++;
            }
        }
        row_ptr[i + 1] = count;
    }
    *nnz = count * BSR_M * BSR_K;
}

void bsr2tilebsr(int rows, int cols, int bsr_nnz, float sparsity, int *bsr_rowptr, int *bsr_colidx, int *blk_rowptr, int *blk_colidx)
{

    blk_rowptr[0] = 0;
    int count = 0;

    printf("NUMS_ROW = %d, NUMS_COL = %d\n", NUMS_ROW, NUMS_COL);
    for (int i = 0; i < NUM_RANK_BLOCK; i++)
    {
        for (int j = 0; j < BSR_COLS / NUMS_COL; j++)
        {
            int start = j * NUMS_COL;
            int end = (j + 1) * NUMS_COL;
            for (int ii = 0; ii < NUMS_ROW; ii++)
            {
                for (int jj = bsr_rowptr[i * NUMS_ROW + ii]; jj < bsr_rowptr[i * NUMS_ROW + ii + 1]; jj++)
                {
                    int col = bsr_colidx[jj];
                    if (col >= start && col < end)
                    {
                        blk_colidx[count++] = col % NUMS_COL;
                    }
                }

                blk_rowptr[i * NUMS_ROW * (BSR_COLS / NUMS_COL) + j * NUMS_ROW + ii + 1] = count;
            }
        }
    }
}
int main(int argc, char *argv[])
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "GPU " << prop.pciDeviceID << " Model: " << prop.name << std::endl;

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = BLOCKS_PER_GRID;

    float sparsity = 0.5;
    int max_nnz = BSR_ROWS * BSR_COLS;
    int bsr_nnz = 0;
    int *bsr_rowptr = (int *)malloc((BSR_ROWS + 1) * sizeof(int));
    int *bsr_colidx = (int *)malloc(max_nnz * sizeof(int));
    half *bsr_val = (half *)malloc(max_nnz * BSR_M * BSR_K * sizeof(half));
    generate_random_bsr_matrix(BSR_ROWS, BSR_COLS, sparsity, bsr_rowptr, bsr_colidx, bsr_val, &bsr_nnz);

    int *blk_rowptr = (int *)malloc(((BSR_ROWS / NUMS_ROW) * (BSR_COLS / NUMS_COL) * NUMS_ROW + 1) * sizeof(int));
    int *blk_colidx = (int *)malloc(bsr_nnz / (BSR_M * BSR_K) * sizeof(int));
    bsr2tilebsr(BSR_ROWS, BSR_COLS, bsr_nnz, sparsity, bsr_rowptr, bsr_colidx, blk_rowptr, blk_colidx);

    half *h_B = (half *)malloc(K_BLOCK * N_BLOCK * sizeof(half));
    for (int i = 0; i < K_BLOCK * N_BLOCK; i++)
    {

        h_B[i] = rand() % 2 + 1.0;
    }

    int *d_blk_rowptr, *d_blk_colidx;
    half *d_bsr_val;
    half *d_B;
    half *d_C;

    cudaMalloc(&d_blk_rowptr, ((BSR_ROWS / NUMS_ROW) * (BSR_COLS / NUMS_COL) * NUMS_ROW + 1) * sizeof(int));
    cudaMalloc(&d_blk_colidx, bsr_nnz / (BSR_M * BSR_K) * sizeof(int));
    cudaMalloc(&d_bsr_val, bsr_nnz * sizeof(half));
    cudaMalloc(&d_B, K_BLOCK * N_BLOCK * sizeof(half));
    cudaMalloc(&d_C, M_BLOCK * N_BLOCK * sizeof(half));

    cudaMemcpy(d_blk_rowptr, blk_rowptr, ((BSR_ROWS / NUMS_ROW) * (BSR_COLS / NUMS_COL) * NUMS_ROW + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_blk_colidx, blk_colidx, bsr_nnz / (BSR_M * BSR_K) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bsr_val, bsr_val, bsr_nnz * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K_BLOCK * N_BLOCK * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M_BLOCK * N_BLOCK * sizeof(half));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaFuncSetAttribute(block_spmm_2d_half_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size_block_spmm);

    std::cout << "Launching kernel with " << blocksPerGrid << " blocks and "
              << threadsPerBlock << " threads per block and " << shmem_size_block_spmm << " bytes of shared memory" << std::endl;

    cudaEventRecord(start);
    block_spmm_2d_half_mma<<<blocksPerGrid, threadsPerBlock, shmem_size_block_spmm>>>(d_blk_rowptr, d_blk_colidx, d_bsr_val, d_B, d_C, 1.0, 0.0);
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

    long long numOpsPerMatrix = 2LL * bsr_nnz * K_BLOCK;
    long long totalOps = numOpsPerMatrix * blocksPerGrid * NUM_ITER;
    double gflops = static_cast<double>(totalOps) / (milliseconds * 1e6);
    double tflops = gflops / 1000.0f;

    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS (" << tflops << " TFLOPS)" << std::endl;

    std::cout << "[hemeng_log],2d," << M_BLOCK << "," << NUM_RANK_ALL_BLOCK << "," << milliseconds << "," << tflops << std::endl;

    half *h_C = (half *)malloc(M_BLOCK * N_BLOCK * sizeof(half));
    cudaMemcpy(h_C, d_C, M_BLOCK * N_BLOCK * sizeof(half), cudaMemcpyDeviceToHost);

    double *h_C_ref = (double *)malloc(M_BLOCK * N_BLOCK * sizeof(double));
    cudaMemset(h_C_ref, 0, M_BLOCK * N_BLOCK * sizeof(double));

    for (int i = 0; i < NUM_RANK_BLOCK; i++)
    {
        for (int j = 0; j < BSR_COLS / NUMS_COL; j++)
        {
            for (int ii = 0; ii < NUMS_ROW; ii++)
            {
                int ptr_offset = i * NUMS_ROW * (BSR_COLS / NUMS_COL) + j * NUMS_ROW + ii;
                for (int jj = blk_rowptr[ptr_offset]; jj < blk_rowptr[ptr_offset + 1]; jj++)
                {
                    int col = blk_colidx[jj] + j * NUMS_COL;
                    half *a = &bsr_val[jj * BSR_M * BSR_K];
                    for (int iii = 0; iii < BSR_M; iii++)
                    {
                        for (int jjj = 0; jjj < BSR_K; jjj++)
                        {
                            for (int k = 0; k < N_BLOCK; k++)
                            {
                                h_C_ref[(i * NUMS_ROW * BSR_M + ii * BSR_M + iii) * N_BLOCK + k] +=
                                    (double)a[iii * BSR_K + jjj] * (double)h_B[k * K_BLOCK + (col * BSR_K + jjj)];
                            }
                        }
                    }
                }
            }
        }
    }

    int error = 0;
    for (int i = 0; i < M_BLOCK * N_BLOCK; i++)
    {
        if (fabs((double)h_C[i] / NUM_ITER - (double)h_C_ref[i]) > 1e-6)
        {
            std::cout << "Error: " << i << " " << (double)h_C[i] << " " << (double)h_C_ref[i] << std::endl;
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