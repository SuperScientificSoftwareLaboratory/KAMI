#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_ext.h>

#include <rocwmma/rocwmma.hpp>
#include <iostream>

#define WARP_SIZE 32

#ifndef M_BLOCK
#define M_BLOCK 64
#define N_BLOCK 64
#define K_BLOCK 64
#endif

#ifndef NUM_RANK_BLOCK
#define NUM_RANK_BLOCK 2
#endif

#define NUM_RANK_ALL_BLOCK (NUM_RANK_BLOCK * NUM_RANK_BLOCK * NUM_RANK_BLOCK)

#ifndef NUM_ALLOC_RANK_BLOCK
#define NUM_ALLOC_RANK_BLOCK NUM_RANK_ALL_BLOCK
#endif

#define WMMA_M_BLOCK 16
#define WMMA_N_BLOCK 16
#define WMMA_K_BLOCK 16

#define SUB_M_BLOCK (M_BLOCK / NUM_RANK_BLOCK)
#define SUB_N_BLOCK (N_BLOCK / NUM_RANK_BLOCK)
#define SUB_K_BLOCK (K_BLOCK / NUM_RANK_BLOCK / NUM_RANK_BLOCK)

const int shmem_size_block_gemm = ((SUB_M_BLOCK * SUB_K_BLOCK * NUM_RANK_BLOCK * NUM_RANK_BLOCK) + (SUB_K_BLOCK * SUB_N_BLOCK * NUM_RANK_BLOCK * NUM_RANK_BLOCK)) * sizeof(half);

#ifndef NUM_ITER
#define NUM_ITER 1000
#endif

#define THREADS_PER_BLOCK (NUM_RANK_ALL_BLOCK * WARP_SIZE)

#define BLOCKS_PER_GRID 16384

__global__ void block_gemm_3d_half_wmma(const half *A, const half *B, half *C, const half alpha, const half beta)
{
    extern __shared__ __align__(16) half shmem[];

    half *shmem_a = shmem;
    half *shmem_b = shmem + SUB_M_BLOCK * SUB_K_BLOCK * NUM_RANK_BLOCK * NUM_RANK_BLOCK;

    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31;

    const unsigned int warp_row = warp_id / NUM_RANK_BLOCK / NUM_RANK_BLOCK;
    const unsigned int warp_col = warp_id / NUM_RANK_BLOCK % NUM_RANK_BLOCK;
    const unsigned int warp_dep = warp_id % NUM_RANK_BLOCK;

    if (warp_row >= NUM_RANK_BLOCK || warp_col >= NUM_RANK_BLOCK)
    {
        return;
    }

    rocwmma::fragment<rocwmma::matrix_a, WMMA_M_BLOCK, WMMA_N_BLOCK, WMMA_K_BLOCK, half, rocwmma::row_major> a_frags[SUB_M_BLOCK / WMMA_M_BLOCK][SUB_K_BLOCK / WMMA_K_BLOCK];
    rocwmma::fragment<rocwmma::matrix_b, WMMA_M_BLOCK, WMMA_N_BLOCK, WMMA_K_BLOCK, half, rocwmma::row_major> b_frags[SUB_K_BLOCK / WMMA_K_BLOCK][SUB_N_BLOCK / WMMA_N_BLOCK];
    rocwmma::fragment<rocwmma::matrix_a, WMMA_M_BLOCK, WMMA_N_BLOCK, WMMA_K_BLOCK, half, rocwmma::row_major> a_frags_compute[SUB_M_BLOCK / WMMA_M_BLOCK][SUB_K_BLOCK / WMMA_K_BLOCK];
    rocwmma::fragment<rocwmma::matrix_b, WMMA_M_BLOCK, WMMA_N_BLOCK, WMMA_K_BLOCK, half, rocwmma::row_major> b_frags_compute[SUB_K_BLOCK / WMMA_K_BLOCK][SUB_N_BLOCK / WMMA_N_BLOCK];
    rocwmma::fragment<rocwmma::accumulator, WMMA_M_BLOCK, WMMA_N_BLOCK, WMMA_K_BLOCK, half> c_frags[SUB_M_BLOCK / WMMA_M_BLOCK][SUB_N_BLOCK / WMMA_N_BLOCK];

    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
    {
        for (int j = 0; j < SUB_K_BLOCK / WMMA_K_BLOCK; ++j)
        {
            int real_row = warp_row * (SUB_M_BLOCK / WMMA_M_BLOCK) + i;
            int real_col = (warp_col * NUM_RANK_BLOCK + warp_dep) * (SUB_K_BLOCK / WMMA_K_BLOCK) + j;
            int real_idx = real_row * WMMA_M_BLOCK * K_BLOCK + real_col * WMMA_K_BLOCK;
            rocwmma::load_matrix_sync(a_frags[i][j], A + real_idx, K_BLOCK);
        }
    }

    for (int i = 0; i < SUB_K_BLOCK / WMMA_K_BLOCK; ++i)
    {
        for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; ++j)
        {
            int real_row = (warp_row * NUM_RANK_BLOCK + warp_dep) * (SUB_K_BLOCK / WMMA_K_BLOCK) + i;
            int real_col = warp_col * (SUB_N_BLOCK / WMMA_N_BLOCK) + j;
            int real_idx = real_row * WMMA_K_BLOCK * N_BLOCK + real_col * WMMA_N_BLOCK;
            rocwmma::load_matrix_sync(b_frags[i][j], B + real_idx, N_BLOCK);
        }
    }

    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
    {
        for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; ++j)
        {
            rocwmma::fill_fragment(c_frags[i][j], half(0.0));
        }
    }

    __syncthreads();

    for (int idx_iter = 0; idx_iter < NUM_ITER; ++idx_iter)
    {
        for (int idx_stage = 0; idx_stage < NUM_RANK_BLOCK; ++idx_stage)
        {

            if (warp_col == idx_stage)
            {
                for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
                {
                    for (int j = 0; j < SUB_K_BLOCK / WMMA_K_BLOCK; ++j)
                    {
                        int dep_offset = warp_dep * WMMA_M_BLOCK * WMMA_K_BLOCK;
                        int row_offset = warp_row * SUB_M_BLOCK * SUB_K_BLOCK * NUM_RANK_BLOCK;
                        for (int k = 0; k < 8; k++)
                        {
                            shmem_a[lane_id * 8 + k + (i * (SUB_K_BLOCK / WMMA_K_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_M_BLOCK * WMMA_K_BLOCK + dep_offset + row_offset] = a_frags[i][j].x[k];
                        }
                    }
                }
            }

            if (warp_row == idx_stage)
            {
                for (int i = 0; i < SUB_K_BLOCK / WMMA_K_BLOCK; ++i)
                {
                    for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; ++j)
                    {
                        int dep_offset = warp_dep * WMMA_K_BLOCK * WMMA_N_BLOCK;
                        int col_offset = warp_col * SUB_K_BLOCK * SUB_N_BLOCK * NUM_RANK_BLOCK;
                        for (int k = 0; k < 8; k++)
                        {
                            shmem_b[lane_id * 8 + k + (i * (SUB_N_BLOCK / WMMA_N_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_K_BLOCK * WMMA_N_BLOCK + dep_offset + col_offset] = b_frags[i][j].x[k];
                        }
                    }
                }
            }

            __syncthreads();

            for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
            {
                for (int j = 0; j < SUB_K_BLOCK / WMMA_K_BLOCK; ++j)
                {
                    int dep_offset = warp_dep * WMMA_M_BLOCK * WMMA_K_BLOCK;
                    int row_offset = warp_row * SUB_M_BLOCK * SUB_K_BLOCK * NUM_RANK_BLOCK;
                    for (int k = 0; k < 8; k++)
                    {
                        a_frags_compute[i][j].x[k] = shmem_a[lane_id * 8 + k + (i * (SUB_K_BLOCK / WMMA_K_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_M_BLOCK * WMMA_K_BLOCK + dep_offset + row_offset];
                    }
                }
            }

            for (int i = 0; i < SUB_K_BLOCK / WMMA_K_BLOCK; ++i)
            {
                for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; ++j)
                {
                    int dep_offset = warp_dep * WMMA_K_BLOCK * WMMA_N_BLOCK;
                    int col_offset = warp_col * SUB_K_BLOCK * SUB_N_BLOCK * NUM_RANK_BLOCK;
                    for (int k = 0; k < 8; k++)
                    {
                        b_frags_compute[i][j].x[k] = shmem_b[lane_id * 8 + k + (i * (SUB_N_BLOCK / WMMA_N_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_K_BLOCK * WMMA_N_BLOCK + dep_offset + col_offset];
                    }
                }
            }

            __syncthreads();

            for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
            {
                for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; ++j)
                {
                    for (int k = 0; k < SUB_K_BLOCK / WMMA_K_BLOCK; ++k)
                    {
                        rocwmma::mma_sync(c_frags[i][j], a_frags_compute[i][k], b_frags_compute[k][j], c_frags[i][j]);
                    }
                }
            }
        }
    }

    __syncthreads();
    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
    {
        for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; ++j)
        {
            int real_row = warp_row * (SUB_M_BLOCK / WMMA_M_BLOCK) + i;
            int real_col = warp_col * (SUB_N_BLOCK / WMMA_N_BLOCK) + j;
            int real_idx = real_row * WMMA_M_BLOCK * N_BLOCK + real_col * WMMA_N_BLOCK;
            int global_idx = real_idx + warp_dep * M_BLOCK * N_BLOCK;

            rocwmma::store_matrix_sync(C + global_idx, c_frags[i][j], N_BLOCK, rocwmma::mem_row_major);
        }
    }
}

int main(int argc, char *argv[])
{
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);

    std::cout << "GPU " << prop.pciDeviceID << " Model: " << prop.name << std::endl;

    int threadsPerBlock = THREADS_PER_BLOCK;
    int blocksPerGrid = BLOCKS_PER_GRID;

    half *h_A = (half *)malloc(M_BLOCK * K_BLOCK * sizeof(half));
    half *h_B = (half *)malloc(K_BLOCK * N_BLOCK * sizeof(half));

    for (int i = 0; i < M_BLOCK * K_BLOCK; i++)
    {

        h_A[i] = rand() % 2 + 1.0;
    }

    for (int i = 0; i < K_BLOCK * N_BLOCK; i++)
    {

        h_B[i] = rand() % 2 + 1.0;
    }

    half *d_A, *d_B;
    half *d_C;
    hipMalloc(&d_A, M_BLOCK * K_BLOCK * sizeof(half));
    hipMalloc(&d_B, K_BLOCK * N_BLOCK * sizeof(half));
    hipMalloc(&d_C, M_BLOCK * N_BLOCK * NUM_RANK_BLOCK * sizeof(half));

    hipMemcpy(d_A, h_A, M_BLOCK * K_BLOCK * sizeof(half), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, K_BLOCK * N_BLOCK * sizeof(half), hipMemcpyHostToDevice);
    hipMemset(d_C, 0, M_BLOCK * N_BLOCK * NUM_RANK_BLOCK * sizeof(half));

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipFuncSetAttribute(reinterpret_cast<const void *>(block_gemm_3d_half_wmma), hipFuncAttributeMaxDynamicSharedMemorySize, shmem_size_block_gemm);

    std::cout << "Launching kernel with " << BLOCKS_PER_GRID << " blocks and "
              << NUM_ALLOC_RANK_BLOCK * WARP_SIZE << " threads per block and " << shmem_size_block_gemm << " bytes of shared memory" << std::endl;
    hipEventRecord(start);
    block_gemm_3d_half_wmma<<<blocksPerGrid, NUM_ALLOC_RANK_BLOCK * WARP_SIZE, shmem_size_block_gemm>>>(d_A, d_B, d_C, 1.0, 0.0);
    hipEventRecord(stop);

    hipDeviceSynchronize();

    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
    {
        std::cerr << "CUDA Error: " << hipGetErrorString(err) << std::endl;
        return -1;
    }

    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);
    long long numOpsPerMatrix = 2LL * M_BLOCK * N_BLOCK * K_BLOCK;
    long long totalOps = numOpsPerMatrix * blocksPerGrid * NUM_ITER;
    double gflops = static_cast<double>(totalOps) / (milliseconds * 1e6);
    double tflops = gflops / 1000.0;

    std::cout << "Average performance: " << gflops << " GFLOPS (" << tflops << " TFLOPS)" << std::endl;

    std::cout << "[hemeng_log],3d," << M_BLOCK << "," << N_BLOCK << "," << K_BLOCK << "," << NUM_RANK_BLOCK << "," << NUM_ALLOC_RANK_BLOCK << "," << tflops << "," << NUM_ALLOC_RANK_BLOCK * WARP_SIZE << std::endl;

    half *h_C = (half *)malloc(M_BLOCK * N_BLOCK * NUM_RANK_BLOCK * sizeof(half));
    hipMemcpy(h_C, d_C, M_BLOCK * N_BLOCK * NUM_RANK_BLOCK * sizeof(half), hipMemcpyDeviceToHost);

    for (int i = 0; i < M_BLOCK; i++)
    {
        for (int j = 0; j < N_BLOCK; j++)
        {
            for (int k = 1; k < NUM_RANK_BLOCK; k++)
            {
                h_C[i * N_BLOCK + j] = (double)h_C[i * N_BLOCK + j] + (double)h_C[i * N_BLOCK + j + k * M_BLOCK * N_BLOCK];
            }
        }
    }

    half *h_C_ref = (half *)malloc(M_BLOCK * N_BLOCK * sizeof(half));

    for (int i = 0; i < M_BLOCK; i++)
    {
        for (int j = 0; j < N_BLOCK; j++)
        {
            h_C_ref[i * N_BLOCK + j] = 0;
            for (int k = 0; k < K_BLOCK; k++)
            {
                h_C_ref[i * N_BLOCK + j] = (double)h_C_ref[i * N_BLOCK + j] + (double)h_A[i * K_BLOCK + k] * (double)h_B[k * N_BLOCK + j];
            }
        }
    }

    int error = 0;
    for (int i = 0; i < M_BLOCK * N_BLOCK; i++)
    {
        if (fabs((double)h_C[i] / NUM_ITER - (double)h_C_ref[i]) > 1e-6)
        {
            error = 1;
            break;
        }
    }

    if (!error)
    {
        std::cout << "Validation successful!" << std::endl;
    }

    hipEventDestroy(start);
    hipEventDestroy(stop);

    return 0;
}
