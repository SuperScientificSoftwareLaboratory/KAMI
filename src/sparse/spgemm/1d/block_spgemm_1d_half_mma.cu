
#include <mma.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#define WARP_SIZE 32

using namespace nvcuda;

#ifndef BSR_M
#define BSR_M 16
#define BSR_K 16
#define BSR_N 16
#endif

#ifndef M_BLOCK
#define M_BLOCK 128
#define N_BLOCK 128
#define K_BLOCK 128
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

const int shmem_size_block_gemm = SUB_N_BLOCK * K_BLOCK * sizeof(half);

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

__global__ void spgemm_symbolic_kernel(
    const int *RowPtrA, const int *ColIdxA,
    const int *RowPtrB, const int *ColIdxB,
    int *RowPtrC, int m, int k, int n)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < m)
    {
        int d_dense_row_column_flag[N_BLOCK / BSR_N];
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

__global__ void block_spgemm_1d_half_mma(const int *RowPtrA, const int *ColIdxA, const half *ValA,
                                         const int *RowPtrB, const int *ColIdxB, half *ValB,
                                         int *RowPtrC, int *ColIdxC, half *ValC,
                                         const half alpha, const half beta)
{
    extern __shared__ half shmem_b[];

    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31;

    uint32_t a_frags[SUB_M_BLOCK / WMMA_M_BLOCK][K_BLOCK / WMMA_K_BLOCK][4];
    half b_frags[K_BLOCK / WMMA_K_BLOCK][SUB_N_BLOCK / WMMA_N_BLOCK][4];
    uint32_t c_frags[SUB_M_BLOCK / WMMA_M_BLOCK][N_BLOCK / WMMA_N_BLOCK][2];
    char flag[SUB_M_BLOCK / WMMA_M_BLOCK][N_BLOCK / BSR_N];

    int startA = warp_id * (SUB_M_BLOCK / BSR_M);
    int stopA = (warp_id + 1) * (SUB_M_BLOCK / BSR_M);
    for (int i = startA; i < stopA; ++i)
    {
        for (int j = RowPtrA[i]; j < RowPtrA[i + 1]; ++j)
        {
            int col = ColIdxA[j];
            const half *A = &ValA[j * BSR_M * BSR_K];
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

            a_frags[i - startA][col][0] = *(uint32_t *)(&(A[global_offset_upper_left]));
            a_frags[i - startA][col][1] = *(uint32_t *)(&(A[global_offset_lower_left]));
            a_frags[i - startA][col][2] = *(uint32_t *)(&(A[global_offset_upper_right]));
            a_frags[i - startA][col][3] = *(uint32_t *)(&(A[global_offset_lower_right]));
        }
    }

    int startB = warp_id * (SUB_N_BLOCK / BSR_N);
    int stopB = (warp_id + 1) * (SUB_N_BLOCK / BSR_N);
    for (int i = 0; i < K_BLOCK / WMMA_K_BLOCK; ++i)
    {
        for (int j = RowPtrB[i]; j < RowPtrB[i + 1]; ++j)
        {
            int col = ColIdxB[j];
            const half *B = &ValB[j * BSR_K * BSR_N];
            if (col >= startB && col < stopB)
            {
                for (int k = 0; k < (BSR_N / WMMA_N_BLOCK); ++k)
                {
                    int group_id = lane_id >> 2;
                    int thread_in_group = lane_id % 4;
                    int row_upper = thread_in_group * 2;
                    int row_lower = thread_in_group * 2 + 8;
                    b_frags[i][(col - startB) * (BSR_N / WMMA_N_BLOCK) + k][0] = B[(row_upper + 0) * BSR_N + group_id + k * WMMA_N_BLOCK];
                    b_frags[i][(col - startB) * (BSR_N / WMMA_N_BLOCK) + k][1] = B[(row_upper + 1) * BSR_N + group_id + k * WMMA_N_BLOCK];
                    b_frags[i][(col - startB) * (BSR_N / WMMA_N_BLOCK) + k][2] = B[(row_lower + 0) * BSR_N + group_id + k * WMMA_N_BLOCK];
                    b_frags[i][(col - startB) * (BSR_N / WMMA_N_BLOCK) + k][3] = B[(row_lower + 1) * BSR_N + group_id + k * WMMA_N_BLOCK];
                }
            }
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

    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
    {
        for (int j = 0; j < N_BLOCK / BSR_N; ++j)
        {
            flag[i][j] = 0;
        }
    }
    __syncthreads();

    for (int idx_iter = 0; idx_iter < NUM_ITER; ++idx_iter)
    {
        for (int idx_stage = 0; idx_stage < NUM_RANK_BLOCK; ++idx_stage)
        {

            if (warp_id == idx_stage)
            {
                for (int i = 0; i < K_BLOCK / WMMA_K_BLOCK; ++i)
                {
                    for (int j = RowPtrB[i]; j < RowPtrB[i + 1]; ++j)
                    {
                        int col = ColIdxB[j];
                        if (col >= startB && col < stopB)
                        {
                            for (int k = 0; k < (BSR_N / WMMA_N_BLOCK); ++k)
                            {
                                *(uint32_t *)(shmem_b + lane_id * 4 + (i * (SUB_N_BLOCK / WMMA_N_BLOCK) + (col - startB) * (BSR_N / WMMA_N_BLOCK) + k) * WMMA_K_BLOCK * WMMA_N_BLOCK) = *(uint32_t *)(&b_frags[i][(col - startB) * (BSR_N / WMMA_N_BLOCK) + k][0]);
                                *(uint32_t *)(shmem_b + lane_id * 4 + (i * (SUB_N_BLOCK / WMMA_N_BLOCK) + (col - startB) * (BSR_N / WMMA_N_BLOCK) + k) * WMMA_K_BLOCK * WMMA_N_BLOCK + 2) = *(uint32_t *)(&b_frags[i][(col - startB) * (BSR_N / WMMA_N_BLOCK) + k][2]);
                            }
                        }
                    }
                }
            }
            __syncthreads();

            int begin = idx_stage * (SUB_N_BLOCK / BSR_N);
            int end = (idx_stage + 1) * (SUB_N_BLOCK / BSR_N);
            for (int i = startA; i < stopA; ++i)
            {
                for (int l = RowPtrA[i]; l < RowPtrA[i + 1]; ++l)
                {
                    int colA = ColIdxA[l];
                    for (int j = RowPtrB[colA]; j < RowPtrB[colA + 1]; j++)
                    {
                        int colB = ColIdxB[j];
                        if (colB >= begin && colB < end)
                        {
                            flag[i - startA][colB] = 1;
                            for (int k = 0; k < (BSR_N / WMMA_N_BLOCK); ++k)
                            {
                                uint32_t b_frags_compute[2];
                                b_frags_compute[0] = *(uint32_t *)(shmem_b + lane_id * 4 + (colA * (SUB_N_BLOCK / WMMA_N_BLOCK) + (colB - begin) * (BSR_N / WMMA_N_BLOCK) + k) * WMMA_K_BLOCK * WMMA_N_BLOCK);
                                b_frags_compute[1] = *(uint32_t *)(shmem_b + lane_id * 4 + (colA * (SUB_N_BLOCK / WMMA_N_BLOCK) + (colB - begin) * (BSR_N / WMMA_N_BLOCK) + k) * WMMA_K_BLOCK * WMMA_N_BLOCK + 2);
                                mma_m16n8k16_fp16(c_frags[i - startA][colB * (BSR_N / WMMA_N_BLOCK) + k], a_frags[i - startA][colA], b_frags_compute);
                            }
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    int startC = warp_id * (SUB_M_BLOCK / BSR_M);
    int stopC = (warp_id + 1) * (SUB_M_BLOCK / BSR_M);
    for (int i = startC; i < stopC; ++i)
    {
        int nnzr = RowPtrC[i];
        for (int j = 0; j < N_BLOCK / BSR_N; j++)
        {
            if (flag[i - startC][j] == 1)
            {
                int group_id = lane_id >> 2;
                int thread_in_group = lane_id % 4;
                int row_upper = group_id;
                int row_lower = group_id + 8;
                int col = thread_in_group * 2;
                *(uint32_t *)(&(ValC[nnzr * BSR_M * BSR_N + row_upper * BSR_N + col])) = c_frags[i - startC][j * 2][0];
                *(uint32_t *)(&(ValC[nnzr * BSR_M * BSR_N + row_lower * BSR_N + col])) = c_frags[i - startC][j * 2][1];
                *(uint32_t *)(&(ValC[nnzr * BSR_M * BSR_N + row_upper * BSR_N + col + WMMA_N_BLOCK])) = c_frags[i - startC][j * 2 + 1][0];
                *(uint32_t *)(&(ValC[nnzr * BSR_M * BSR_N + row_lower * BSR_N + col + WMMA_N_BLOCK])) = c_frags[i - startC][j * 2 + 1][1];
                ColIdxC[nnzr] = j;
                nnzr++;
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
    cudaFuncSetAttribute(block_spgemm_1d_half_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size_block_gemm);
    std::cout << "Launching kernel with " << blocksPerGrid << " blocks and "
              << threadsPerBlock << " threads per block and " << shmem_size_block_gemm << " bytes of shared memory" << std::endl;

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
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float symbolic_time = 0;
    cudaEventElapsedTime(&symbolic_time, start, stop);
    std::cout << "Symbolic time: " << symbolic_time << " ms" << std::endl;

    cudaEventRecord(start);
    block_spgemm_1d_half_mma<<<blocksPerGrid, threadsPerBlock, shmem_size_block_gemm>>>(d_RowPtrA, d_ColIdxA, d_ValA, d_RowPtrB, d_ColIdxB, d_ValB, d_RowPtrC, d_ColIdxC, d_ValC, 1.0, 0.0);
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
    std::cout << "[hemeng_log],1d," << nnzCub << "," << M_BLOCK << "," << N_BLOCK << "," << K_BLOCK << "," << NUM_RANK_BLOCK << "," << tflops << "," << THREADS_PER_BLOCK << std::endl;

    half *h_ValC = (half *)malloc(nnzC * sizeof(half));
    cudaMemcpy(h_ValC, d_ValC, nnzC * sizeof(half), cudaMemcpyDeviceToHost);

    int error = 0;
    int error_count = 0;
    for (int i = 0; i < nnzC; i++)
    {
        if (fabs((double)h_ValC[i] / NUM_ITER - (double)ValC[i]) > 1e-6)
        {

            error = 1;
            error_count++;
            if (error_count > 100)
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