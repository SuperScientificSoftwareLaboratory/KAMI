#include <mma.h>
#include <cuda_runtime.h>
#include <iostream>

#define WARP_SIZE 32

using namespace nvcuda;

#ifndef M_BLOCK
#define M_BLOCK 64
#define N_BLOCK 64
#define K_BLOCK 64
#endif

#ifndef NUM_RANK_BLOCK
#define NUM_RANK_BLOCK 2
#endif

#define WMMA_M_BLOCK 16
#define WMMA_N_BLOCK 8
#define WMMA_K_BLOCK 32

#define SUB_M_BLOCK (M_BLOCK / NUM_RANK_BLOCK)
#define SUB_N_BLOCK (N_BLOCK / NUM_RANK_BLOCK)
#define SUB_K_BLOCK (K_BLOCK / NUM_RANK_BLOCK)

#define NUM_PIPE_BLOCK 1

#ifndef NUM_ELE_PER_REG
#define NUM_ELE_PER_REG 4
#endif

const int shmem_size_block_gemm = SUB_N_BLOCK * K_BLOCK / NUM_PIPE_BLOCK * sizeof(float) / NUM_ELE_PER_REG;

#ifndef NUM_ITER
#define NUM_ITER 1000
#endif

#define THREADS_PER_BLOCK (NUM_RANK_BLOCK * WARP_SIZE)

#ifndef BLOCKS_PER_GRID
#define BLOCKS_PER_GRID 16384
#endif

__device__ __forceinline__ void mma_m16n8k32_e5m2_fp8_fp32acc(uint32_t *acc, uint32_t *frag_a, uint32_t *frag_b)
{
    uint32_t const *A = reinterpret_cast<uint32_t const *>(&frag_a[0]);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&frag_b[0]);
    uint32_t *C = reinterpret_cast<uint32_t *>(&acc[0]);

    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32"
        " { %0, %1, %2, %3 }, "
        " { %4, %5, %6, %7 }, "
        " { %8, %9 }, "
        " { %10, %11, %12, %13 };"
        : "=r"(C[0]), "=r"(C[1]), "=r"(C[2]), "=r"(C[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3]));
}

__global__ void block_gemm_1d_e5m2_mma(const uint8_t *A, const uint8_t *B, float *C, const float alpha, const float beta)
{
    extern __shared__ uint8_t shmem_b[];

    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31;

    uint32_t a_frags[SUB_M_BLOCK / WMMA_M_BLOCK][K_BLOCK / WMMA_K_BLOCK][4];
    uint32_t b_frags[SUB_N_BLOCK / WMMA_N_BLOCK][K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK][2];
    uint32_t c_frags[SUB_M_BLOCK / WMMA_M_BLOCK][N_BLOCK / WMMA_N_BLOCK][4];

    uint32_t (*b_frags_compute)[K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK][2 * WARP_SIZE] = reinterpret_cast<uint32_t (*)[K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK][2 * WARP_SIZE]>(shmem_b);

    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
    {
        for (int j = 0; j < K_BLOCK / WMMA_K_BLOCK; ++j)
        {
            int group_id = lane_id >> 2;
            int thread_in_group = lane_id % 4;
            int row_upper = group_id;
            int row_lower = group_id + 8;
            int col_left = thread_in_group * 4;
            int col_right = thread_in_group * 4 + 16;

            int global_offset_upper_left = warp_id * (SUB_M_BLOCK * K_BLOCK) + (i * WMMA_M_BLOCK + row_upper) * K_BLOCK + j * WMMA_K_BLOCK + col_left;
            int global_offset_lower_left = warp_id * (SUB_M_BLOCK * K_BLOCK) + (i * WMMA_M_BLOCK + row_lower) * K_BLOCK + j * WMMA_K_BLOCK + col_left;

            int global_offset_upper_right = warp_id * (SUB_M_BLOCK * K_BLOCK) + (i * WMMA_M_BLOCK + row_upper) * K_BLOCK + j * WMMA_K_BLOCK + col_right;
            int global_offset_lower_right = warp_id * (SUB_M_BLOCK * K_BLOCK) + (i * WMMA_M_BLOCK + row_lower) * K_BLOCK + j * WMMA_K_BLOCK + col_right;

            a_frags[i][j][0] = *(uint32_t *)(A + global_offset_upper_left);
            a_frags[i][j][1] = *(uint32_t *)(A + global_offset_lower_left);
            a_frags[i][j][2] = *(uint32_t *)(A + global_offset_upper_right);
            a_frags[i][j][3] = *(uint32_t *)(A + global_offset_lower_right);
        }
    }

    for (int i = 0; i < SUB_N_BLOCK / WMMA_N_BLOCK; ++i)
    {
        for (int j = 0; j < K_BLOCK / WMMA_K_BLOCK; ++j)
        {
            int row = lane_id >> 2;
            int col_left = (lane_id % 4) * 4;
            int col_right = (lane_id % 4) * 4 + 16;
            int global_offset_left = warp_id * (SUB_N_BLOCK * K_BLOCK) + (i * WMMA_N_BLOCK + row) * K_BLOCK + j * WMMA_K_BLOCK + col_left;
            int global_offset_right = warp_id * (SUB_N_BLOCK * K_BLOCK) + (i * WMMA_N_BLOCK + row) * K_BLOCK + j * WMMA_K_BLOCK + col_right;

            b_frags[i][j][0] = *(uint32_t *)(B + global_offset_left);
            b_frags[i][j][1] = *(uint32_t *)(B + global_offset_right);
        }
    }

    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
    {
        for (int j = 0; j < N_BLOCK / WMMA_N_BLOCK; ++j)
        {
            c_frags[i][j][0] = 0.0;
            c_frags[i][j][1] = 0.0;
            c_frags[i][j][2] = 0.0;
            c_frags[i][j][3] = 0.0;
        }
    }

    __syncthreads();

    for (int idx_iter = 0; idx_iter < NUM_ITER; ++idx_iter)
    {
        for (int idx_stage = 0; idx_stage < NUM_RANK_BLOCK; ++idx_stage)
        {
            for (int idx_pipe = 0; idx_pipe < NUM_PIPE_BLOCK; ++idx_pipe)
            {
                if (warp_id == idx_stage)
                {
                    for (int i = 0; i < SUB_N_BLOCK / WMMA_N_BLOCK; i++)
                    {
                        for (int j = 0; j < K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK; j++)
                        {
                            int real_j = j + idx_pipe * (K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK);

                            b_frags_compute[i][j][lane_id * 2 + 0] = b_frags[i][real_j][0];
                            b_frags_compute[i][j][lane_id * 2 + 1] = b_frags[i][real_j][1];
                        }
                    }
                }

                __syncthreads();

#pragma unroll
                for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
                {
#pragma unroll
                    for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; ++j)
                    {
#pragma unroll
                        for (int k = 0; k < K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK; ++k)
                        {
                            int real_k = k + idx_pipe * (K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_BLOCK);
                            mma_m16n8k32_e5m2_fp8_fp32acc(c_frags[i][j + idx_stage * (SUB_N_BLOCK / WMMA_N_BLOCK)], a_frags[i][real_k], &b_frags_compute[j][k][lane_id * 2]);
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
            *(uint32_t *)(&(C[global_offset_upper + 1])) = c_frags[i][j][1];
            *(uint32_t *)(&(C[global_offset_lower])) = c_frags[i][j][2];
            *(uint32_t *)(&(C[global_offset_lower + 1])) = c_frags[i][j][3];
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

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = BLOCKS_PER_GRID;

    uint8_t *h_A = (uint8_t *)malloc(M_BLOCK * K_BLOCK * sizeof(uint8_t));
    uint8_t *h_B = (uint8_t *)malloc(K_BLOCK * N_BLOCK * sizeof(uint8_t));

    for (int i = 0; i < M_BLOCK * K_BLOCK; i++)
    {
        h_A[i] = 0x3C;
    }

    for (int i = 0; i < K_BLOCK * N_BLOCK; i++)
    {
        h_B[i] = 0x3C;
    }

    uint8_t *d_A, *d_B;
    float *d_C;
    cudaMalloc(&d_A, M_BLOCK * K_BLOCK * sizeof(uint8_t));
    cudaMalloc(&d_B, K_BLOCK * N_BLOCK * sizeof(uint8_t));
    cudaMalloc(&d_C, M_BLOCK * N_BLOCK * sizeof(float));

    cudaMemcpy(d_A, h_A, M_BLOCK * K_BLOCK * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K_BLOCK * N_BLOCK * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M_BLOCK * N_BLOCK * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaFuncSetAttribute(block_gemm_1d_e5m2_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size_block_gemm);

    std::cout << "Launching kernel with " << blocksPerGrid << " blocks and "
              << threadsPerBlock << " threads per block and " << shmem_size_block_gemm << " bytes of shared memory" << std::endl;

    cudaEventRecord(start);
    block_gemm_1d_e5m2_mma<<<blocksPerGrid, threadsPerBlock, shmem_size_block_gemm>>>(d_A, d_B, d_C, 1.0, 0.0);
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

    long long numOpsPerMatrix = 2LL * M_BLOCK * N_BLOCK * K_BLOCK;
    long long totalOps = numOpsPerMatrix * blocksPerGrid * NUM_ITER;
    double gflops = static_cast<double>(totalOps) / (milliseconds * 1e6);
    double tflops = gflops / 1000.0f;

    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS (" << tflops << " TFLOPS)" << std::endl;

    std::cout << "[hemeng_log],1d," << M_BLOCK << "," << N_BLOCK << "," << K_BLOCK << "," << NUM_RANK_BLOCK << "," << tflops << "," << THREADS_PER_BLOCK << std::endl;

    float *h_C = (float *)malloc(M_BLOCK * N_BLOCK * sizeof(float));
    cudaMemcpy(h_C, d_C, M_BLOCK * N_BLOCK * sizeof(float), cudaMemcpyDeviceToHost);

    double *h_C_ref = (double *)malloc(M_BLOCK * N_BLOCK * sizeof(double));

    for (int i = 0; i < M_BLOCK; i++)
    {
        for (int j = 0; j < M_BLOCK; j++)
        {
            h_C_ref[i * N_BLOCK + j] = 0;
            for (int k = 0; k < K_BLOCK; k++)
            {
                h_C_ref[i * N_BLOCK + j] += 1.0 * 1.0;
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
