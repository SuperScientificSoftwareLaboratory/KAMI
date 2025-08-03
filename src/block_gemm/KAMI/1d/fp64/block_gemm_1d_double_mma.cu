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
#define NUM_RANK_BLOCK 4
#endif

#ifndef NUM_ALLOC_RANK_BLOCK
#define NUM_ALLOC_RANK_BLOCK NUM_RANK_BLOCK
#endif

#define WMMA_M_BLOCK 16
#define WMMA_N_BLOCK 8
#define WMMA_K_BLOCK 8

#define SUB_M_BLOCK (M_BLOCK / NUM_RANK_BLOCK)
#define SUB_N_BLOCK (N_BLOCK / NUM_RANK_BLOCK)
#define SUB_K_BLOCK (K_BLOCK / NUM_RANK_BLOCK)

#define NUM_PIPE_N_BLOCK 1
#define NUM_PIPE_K_BLOCK 1

const int shmem_size_block_gemm = (SUB_N_BLOCK / NUM_PIPE_N_BLOCK) * (K_BLOCK / NUM_PIPE_K_BLOCK) * sizeof(double);

#define NUM_ITER 1000

#define THREADS_PER_BLOCK (NUM_RANK_BLOCK * WARP_SIZE)

#define BLOCKS_PER_GRID 16384

__device__ __forceinline__ void mma_m16n8k8(double *acc, const double *frag_a, const double *frag_b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f64.f64.f64.f64"
        "{%0,  %1,  %2,  %3},"
        "{%4,  %5,  %6,  %7},"
        "{%8,  %9},"
        "{%0,  %1,  %2,  %3};\n"
        : "+d"(acc[0]), "+d"(acc[1]), "+d"(acc[2]), "+d"(acc[3])
        : "d"(frag_a[0]), "d"(frag_a[1]), "d"(frag_a[2]), "d"(frag_a[3]),
          "d"(frag_b[0]), "d"(frag_b[1]));
}

__global__ void block_gemm_1d_double_mma(const double *A, const double *B, double *C, const double alpha, const double beta)
{
    extern __shared__ double shmem_b[];

    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31;

    if (warp_id >= NUM_RANK_BLOCK)
    {
        return;
    }

    for (int i = 0; i < (SUB_N_BLOCK / NUM_PIPE_N_BLOCK) * (K_BLOCK / NUM_PIPE_K_BLOCK); i++)

    {
        shmem_b[i] = 0;
    }

    double a_frags[SUB_M_BLOCK / WMMA_M_BLOCK][K_BLOCK / WMMA_K_BLOCK][4];
    double b_frags[SUB_N_BLOCK / WMMA_N_BLOCK][K_BLOCK / WMMA_K_BLOCK][2];

    double(*b_compute)[K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK][2 * WARP_SIZE] = reinterpret_cast<double(*)[K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK][2 * WARP_SIZE]>(shmem_b);

    double c_frags[SUB_M_BLOCK / WMMA_M_BLOCK][N_BLOCK / WMMA_N_BLOCK][4];

    for (int warp_row = 0; warp_row < SUB_M_BLOCK / WMMA_M_BLOCK; warp_row++)
    {
        for (int warp_col = 0; warp_col < K_BLOCK / WMMA_K_BLOCK; warp_col++)
        {
            int warp_row_offset = warp_id * SUB_M_BLOCK + warp_row * WMMA_M_BLOCK;
            int warp_col_offset = warp_col * WMMA_K_BLOCK;

            int lane_row_upper_offset = lane_id >> 2;
            int lane_row_lower_offset = lane_row_upper_offset + 8;
            int lane_col_left_offset = lane_id % 4;
            int lane_col_right_offset = lane_col_left_offset + 4;

            int global_offset_upper_left = (warp_row_offset + lane_row_upper_offset) * K_BLOCK + warp_col_offset + lane_col_left_offset;
            int global_offset_upper_right = (warp_row_offset + lane_row_upper_offset) * K_BLOCK + warp_col_offset + lane_col_right_offset;
            int global_offset_lower_left = (warp_row_offset + lane_row_lower_offset) * K_BLOCK + warp_col_offset + lane_col_left_offset;
            int global_offset_lower_right = (warp_row_offset + lane_row_lower_offset) * K_BLOCK + warp_col_offset + lane_col_right_offset;

            a_frags[warp_row][warp_col][0] = A[global_offset_upper_left];
            a_frags[warp_row][warp_col][1] = A[global_offset_lower_left];
            a_frags[warp_row][warp_col][2] = A[global_offset_upper_right];
            a_frags[warp_row][warp_col][3] = A[global_offset_lower_right];
        }
    }

    __syncthreads();

    for (int warp_row = 0; warp_row < SUB_N_BLOCK / WMMA_N_BLOCK; warp_row++)
    {
        for (int warp_col = 0; warp_col < K_BLOCK / WMMA_K_BLOCK; warp_col++)
        {
            int warp_row_offset = warp_id * SUB_N_BLOCK + warp_row * WMMA_N_BLOCK;
            int warp_col_offset = warp_col * WMMA_K_BLOCK;

            int lane_row_offset = lane_id >> 2;
            int lane_col_left_offset = lane_id % 4;
            int lane_col_right_offset = lane_col_left_offset + 4;

            int global_offset_left = (warp_row_offset + lane_row_offset) * K_BLOCK + warp_col_offset + lane_col_left_offset;
            int global_offset_right = (warp_row_offset + lane_row_offset) * K_BLOCK + warp_col_offset + lane_col_right_offset;

            b_frags[warp_row][warp_col][0] = B[global_offset_left];
            b_frags[warp_row][warp_col][1] = B[global_offset_right];
        }
    }

    __syncthreads();

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

    for (int idx_iter = 0; idx_iter < NUM_ITER; idx_iter++)
    {
        for (int idx_stage = 0; idx_stage < NUM_RANK_BLOCK; idx_stage++)
        {
            for (int idx_pipe_n = 0; idx_pipe_n < NUM_PIPE_N_BLOCK; idx_pipe_n++)
            {
                for (int idx_pipe_k = 0; idx_pipe_k < NUM_PIPE_K_BLOCK; idx_pipe_k++)
                {
                    if (warp_id == idx_stage)
                    {
#pragma unroll
                        for (int i = 0; i < (SUB_N_BLOCK / NUM_PIPE_N_BLOCK) / WMMA_N_BLOCK; i++)
                        {
#pragma unroll
                            for (int j = 0; j < (K_BLOCK / NUM_PIPE_K_BLOCK) / WMMA_K_BLOCK; j++)
                            {
                                int real_i = i + idx_pipe_n * ((SUB_N_BLOCK / NUM_PIPE_N_BLOCK) / WMMA_N_BLOCK);

                                int real_j = j + idx_pipe_k * ((K_BLOCK / NUM_PIPE_K_BLOCK) / WMMA_K_BLOCK);

                                b_compute[i][j][lane_id * 2 + 0] = b_frags[real_i][real_j][0];
                                b_compute[i][j][lane_id * 2 + 1] = b_frags[real_i][real_j][1];
                            }
                        }
                    }

                    __syncthreads();

#pragma unroll
                    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; i++)
                    {
#pragma unroll
                        for (int j = 0; j < (SUB_N_BLOCK / NUM_PIPE_N_BLOCK) / WMMA_N_BLOCK; j++)
                        {
#pragma unroll
                            for (int k = 0; k < (K_BLOCK / NUM_PIPE_K_BLOCK) / WMMA_K_BLOCK; k++)
                            {
                                int real_j = j + idx_pipe_n * ((SUB_N_BLOCK / NUM_PIPE_N_BLOCK) / WMMA_N_BLOCK);

                                int real_k = k + idx_pipe_k * ((K_BLOCK / NUM_PIPE_K_BLOCK) / WMMA_K_BLOCK);

                                mma_m16n8k8(c_frags[i][real_j + idx_stage * (SUB_N_BLOCK / WMMA_N_BLOCK)], a_frags[i][real_k], &b_compute[j][k][lane_id * 2]);
                            }
                        }
                    }

                    __syncthreads();
                }
            }
        }
    }
    for (int warp_row = 0; warp_row < SUB_M_BLOCK / WMMA_M_BLOCK; warp_row++)
    {
        for (int warp_col = 0; warp_col < N_BLOCK / WMMA_N_BLOCK; warp_col++)
        {
            int warp_row_offset = warp_id * SUB_M_BLOCK + warp_row * WMMA_M_BLOCK;
            int warp_col_offset = warp_col * WMMA_N_BLOCK;

            int lane_row_upper_offset = lane_id >> 2;
            int lane_row_lower_offset = lane_row_upper_offset + 8;
            int lane_col_offset = (lane_id % 4) * 2;

            int global_offset_upper = (warp_row_offset + lane_row_upper_offset) * N_BLOCK + warp_col_offset + lane_col_offset;
            int global_offset_lower = (warp_row_offset + lane_row_lower_offset) * N_BLOCK + warp_col_offset + lane_col_offset;

            C[global_offset_upper] = c_frags[warp_row][warp_col][0];
            C[global_offset_upper + 1] = c_frags[warp_row][warp_col][1];
            C[global_offset_lower] = c_frags[warp_row][warp_col][2];
            C[global_offset_lower + 1] = c_frags[warp_row][warp_col][3];
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

    double *h_A = (double *)malloc(M_BLOCK * K_BLOCK * sizeof(double));
    double *h_B = (double *)malloc(K_BLOCK * N_BLOCK * sizeof(double));

    for (int i = 0; i < M_BLOCK * K_BLOCK; i++)
    {
        h_A[i] = 1;

        h_A[i] = i;
    }

    for (int i = 0; i < K_BLOCK * N_BLOCK; i++)
    {
        h_B[i] = 1;

        h_B[i] = i;
    }

    double *d_A, *d_B;
    double *d_C;

    cudaMalloc(&d_A, M_BLOCK * K_BLOCK * sizeof(double));
    cudaMalloc(&d_B, K_BLOCK * N_BLOCK * sizeof(double));
    cudaMalloc(&d_C, M_BLOCK * N_BLOCK * sizeof(double));

    cudaMemcpy(d_A, h_A, M_BLOCK * K_BLOCK * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K_BLOCK * N_BLOCK * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, M_BLOCK * N_BLOCK * sizeof(double));

    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaFuncSetAttribute(block_gemm_1d_double_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size_block_gemm);

    std::cout << "Launching kernel with " << BLOCKS_PER_GRID << " blocks and "
              << NUM_ALLOC_RANK_BLOCK * WARP_SIZE << " threads per block and " << shmem_size_block_gemm << " bytes of shared memory" << std::endl;

    block_gemm_1d_double_mma<<<BLOCKS_PER_GRID, NUM_ALLOC_RANK_BLOCK * WARP_SIZE, shmem_size_block_gemm>>>(d_A, d_B, d_C, 1.0, 0.0);

    cudaEventRecord(start);
    block_gemm_1d_double_mma<<<BLOCKS_PER_GRID, NUM_ALLOC_RANK_BLOCK * WARP_SIZE, shmem_size_block_gemm>>>(d_A, d_B, d_C, 1.0, 0.0);
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
    long long totalOps = numOpsPerMatrix * BLOCKS_PER_GRID * NUM_ITER;
    double gflops = static_cast<double>(totalOps) / (milliseconds * 1e6);
    double tflops = gflops / 1000.0f;

    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS (" << tflops << " TFLOPS)" << std::endl;

    std::cout << "[hemeng_log],1d," << M_BLOCK << "," << N_BLOCK << "," << K_BLOCK << "," << NUM_RANK_BLOCK << "," << NUM_ALLOC_RANK_BLOCK << "," << tflops << "," << NUM_ALLOC_RANK_BLOCK * 32 << std::endl;

    double *h_C = (double *)malloc(M_BLOCK * N_BLOCK * sizeof(double));
    cudaMemcpy(h_C, d_C, M_BLOCK * N_BLOCK * sizeof(double), cudaMemcpyDeviceToHost);

    double *h_C_ref = (double *)malloc(M_BLOCK * N_BLOCK * sizeof(double));

    for (int i = 0; i < M_BLOCK; i++)
    {
        for (int j = 0; j < M_BLOCK; j++)
        {
            h_C_ref[i * N_BLOCK + j] = 0;
            for (int k = 0; k < K_BLOCK; k++)
            {
                h_C_ref[i * N_BLOCK + j] += h_A[i * K_BLOCK + k] * h_B[j * K_BLOCK + k];
            }
        }
    }

    int error = 0;
    for (int i = 0; i < M_BLOCK * N_BLOCK; i++)
    {
        if (fabs(h_C[i] / NUM_ITER - h_C_ref[i]) > 1e-6)
        {
            std::cout << "Error: " << i << " " << h_C[i] << " " << h_C_ref[i] << std::endl;
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
