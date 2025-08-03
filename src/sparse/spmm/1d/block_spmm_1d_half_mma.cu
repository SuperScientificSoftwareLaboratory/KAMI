
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
#define M_BLOCK 64
#define N_BLOCK 64
#define K_BLOCK 64
#endif

#ifndef BSR_ROWS
#define BSR_ROWS (M_BLOCK / BSR_M)
#define BSR_COLS (K_BLOCK / BSR_K)
#endif

#ifndef NUM_RANK_BLOCK
#define NUM_RANK_BLOCK 4
#endif

#define WMMA_M_BLOCK 16
#define WMMA_N_BLOCK 8
#define WMMA_K_BLOCK 16

#define SUB_M_BLOCK (M_BLOCK / NUM_RANK_BLOCK)
#define SUB_N_BLOCK (N_BLOCK / NUM_RANK_BLOCK)
#define SUB_K_BLOCK (K_BLOCK / NUM_RANK_BLOCK)

const int shmem_size_block_spmm = SUB_N_BLOCK * K_BLOCK * sizeof(half);

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

__device__ __forceinline__ half load_half_from_global(const half *a)
{
    ushort r;
    asm volatile("ld.global.cs.u16 %0, [%1];" : "=h"(r) : "l"(a));
    half *r_half = reinterpret_cast<half *>(&r);
    return *r_half;
}

__device__ __forceinline__ void store_half_to_global(const half *a, half v)
{
    ushort *v_u = reinterpret_cast<ushort *>(&v);
    asm volatile("st.global.cs.u16 [%0], %1;" ::"l"(a), "h"(*v_u));
}

__global__ void block_spmm_1d_half_mma(const int *bsr_rowptr, const int *bsr_colidx, const half *bsr_vals, const half *B, half *C, const half alpha, const half beta)
{
    extern __shared__ half shmem_b[];

    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31;

    uint32_t a_frags[SUB_M_BLOCK / WMMA_M_BLOCK][K_BLOCK / WMMA_K_BLOCK][4];
    uint32_t b_frags[SUB_N_BLOCK / WMMA_N_BLOCK][K_BLOCK / WMMA_K_BLOCK][2];
    uint32_t b_frags_compute[2];
    uint32_t c_frags[SUB_M_BLOCK / WMMA_M_BLOCK][N_BLOCK / WMMA_N_BLOCK][2];

    int start = warp_id * (SUB_M_BLOCK / BSR_M);
    int stop = (warp_id + 1) * (SUB_M_BLOCK / BSR_M);

    for (int i = start; i < stop; ++i)
    {
        for (int j = bsr_rowptr[i]; j < bsr_rowptr[i + 1]; ++j)
        {
            const half *A = &bsr_vals[j * BSR_M * BSR_K];
            int col = bsr_colidx[j];
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

            a_frags[i - start][col][0] = *(uint32_t *)(&(A[global_offset_upper_left]));
            a_frags[i - start][col][1] = *(uint32_t *)(&(A[global_offset_lower_left]));
            a_frags[i - start][col][2] = *(uint32_t *)(&(A[global_offset_upper_right]));
            a_frags[i - start][col][3] = *(uint32_t *)(&(A[global_offset_lower_right]));
        }
    }

    for (int i = 0; i < SUB_N_BLOCK / WMMA_N_BLOCK; ++i)
    {
        for (int j = 0; j < K_BLOCK / WMMA_K_BLOCK; ++j)
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
        for (int j = 0; j < N_BLOCK / WMMA_N_BLOCK; ++j)
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

            if (warp_id == idx_stage)
            {
                for (int i = 0; i < SUB_N_BLOCK / WMMA_N_BLOCK; i++)
                {
                    for (int j = 0; j < K_BLOCK / WMMA_K_BLOCK; j++)
                    {
                        *(uint32_t *)(shmem_b + lane_id * 4 + (i * (K_BLOCK / WMMA_K_BLOCK) + j) * WMMA_N_BLOCK * WMMA_K_BLOCK) = b_frags[i][j][0];
                        *(uint32_t *)(shmem_b + lane_id * 4 + (i * (K_BLOCK / WMMA_K_BLOCK) + j) * WMMA_N_BLOCK * WMMA_K_BLOCK + 2) = b_frags[i][j][1];
                    }
                }
            }

            __syncthreads();

            for (int i = start; i < stop; ++i)
            {
                for (int k = bsr_rowptr[i]; k < bsr_rowptr[i + 1]; ++k)
                {
                    int col = bsr_colidx[k];
                    for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; ++j)
                    {
                        uint32_t b_frags_compute[2];
                        b_frags_compute[0] = *(uint32_t *)(shmem_b + lane_id * 4 + (j * (K_BLOCK / WMMA_K_BLOCK) + col) * WMMA_N_BLOCK * WMMA_K_BLOCK);
                        b_frags_compute[1] = *(uint32_t *)(shmem_b + lane_id * 4 + (j * (K_BLOCK / WMMA_K_BLOCK) + col) * WMMA_N_BLOCK * WMMA_K_BLOCK + 2);
                        mma_m16n8k16_fp16(c_frags[i - start][j + idx_stage * (SUB_N_BLOCK / WMMA_N_BLOCK)], a_frags[i - start][col], b_frags_compute);
                    }
                }
            }
            __syncthreads();
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
                    values[count * BSR_M * BSR_K + len] = rand() % 3;
                }
                count++;
            }
        }
        row_ptr[i + 1] = count;
    }
    *nnz = count * BSR_M * BSR_K;
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
    int max_nnz = BSR_ROWS * BSR_COLS;
    int bsr_nnz = 0;
    int *bsr_rowptr = (int *)malloc((BSR_ROWS + 1) * sizeof(int));
    int *bsr_colidx = (int *)malloc(max_nnz * sizeof(int));
    half *bsr_val = (half *)malloc(max_nnz * BSR_M * BSR_K * sizeof(half));
    generate_random_bsr_matrix(BSR_ROWS, BSR_COLS, sparsity, bsr_rowptr, bsr_colidx, bsr_val, &bsr_nnz);

    half *h_B = (half *)malloc(K_BLOCK * N_BLOCK * sizeof(half));
    for (int i = 0; i < K_BLOCK * N_BLOCK; i++)
    {

        h_B[i] = rand() % 3;
    }
    int *d_bsr_rowptr, *d_bsr_colidx;
    half *d_bsr_val;
    half *d_B;
    half *d_C;

    cudaMalloc(&d_bsr_rowptr, (BSR_ROWS + 1) * sizeof(int));
    cudaMalloc(&d_bsr_colidx, bsr_nnz * sizeof(int));
    cudaMalloc(&d_bsr_val, bsr_nnz * sizeof(half));
    cudaMalloc(&d_B, K_BLOCK * N_BLOCK * sizeof(half));
    cudaMalloc(&d_C, M_BLOCK * N_BLOCK * sizeof(half));

    cudaMemcpy(d_bsr_rowptr, bsr_rowptr, (BSR_ROWS + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bsr_colidx, bsr_colidx, bsr_nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bsr_val, bsr_val, bsr_nnz * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K_BLOCK * N_BLOCK * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M_BLOCK * N_BLOCK * sizeof(half));

    printf("bsr_nnz = %d\n", bsr_nnz);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaFuncSetAttribute(block_spmm_1d_half_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size_block_spmm);
    std::cout << "Launching kernel with " << blocksPerGrid << " blocks and "
              << threadsPerBlock << " threads per block and " << shmem_size_block_spmm << " bytes of shared memory" << std::endl;

    cudaEventRecord(start);
    block_spmm_1d_half_mma<<<blocksPerGrid, threadsPerBlock, shmem_size_block_spmm>>>(d_bsr_rowptr, d_bsr_colidx, d_bsr_val, d_B, d_C, 1.0, 0.0);
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

    std::cout << "[hemeng_log],1d," << M_BLOCK << "," << NUM_RANK_BLOCK << "," << milliseconds << "," << tflops << std::endl;

    half *h_C = (half *)malloc(M_BLOCK * N_BLOCK * sizeof(half));
    cudaMemcpy(h_C, d_C, M_BLOCK * N_BLOCK * sizeof(half), cudaMemcpyDeviceToHost);

    double *h_C_ref = (double *)malloc(M_BLOCK * N_BLOCK * sizeof(double));
    cudaMemset(h_C_ref, 0, M_BLOCK * N_BLOCK * sizeof(double));

    for (int i = 0; i < BSR_ROWS; i++)
    {
        int rows = i * BSR_M;
        for (int j = bsr_rowptr[i]; j < bsr_rowptr[i + 1]; j++)
        {
            int cols = bsr_colidx[j] * BSR_K;
            half *a = &bsr_val[j * BSR_M * BSR_K];
            for (int ii = 0; ii < BSR_M; ii++)
            {
                for (int jj = 0; jj < BSR_K; jj++)
                {
                    for (int k = 0; k < N_BLOCK; k++)
                    {
                        h_C_ref[(rows + ii) * N_BLOCK + k] += (double)a[ii * BSR_K + jj] * (double)h_B[k * K_BLOCK + (cols + jj)];
                    }
                }
            }
        }
    }

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