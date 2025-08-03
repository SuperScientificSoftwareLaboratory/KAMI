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
#define NUM_RANK_BLOCK 4
#endif

#ifndef NUM_ALLOC_RANK_BLOCK
#define NUM_ALLOC_RANK_BLOCK NUM_RANK_BLOCK
#endif

#define WMMA_M_BLOCK 16
#define WMMA_N_BLOCK 16
#define WMMA_K_BLOCK 16

#define SUB_M_BLOCK (M_BLOCK / NUM_RANK_BLOCK)
#define SUB_N_BLOCK (N_BLOCK / NUM_RANK_BLOCK)
#define SUB_K_BLOCK (K_BLOCK / NUM_RANK_BLOCK)

const int shmem_size_block_gemm = SUB_K_BLOCK * N_BLOCK * sizeof(half);

#define FETCH_INT4(pointer) (reinterpret_cast<int4 *>(&(pointer))[0])

#define NUM_ITER 1000

#define THREADS_PER_BLOCK (NUM_RANK_BLOCK * WARP_SIZE)

#define BLOCKS_PER_GRID 16384

__global__ void block_gemm_1d_half_wmma(const half *A, const half *B, half *C, const half alpha, const half beta)
{
    extern __shared__ __align__(16) half shmem_b[];

    const unsigned int warp_id = threadIdx.x >> 5;
    const unsigned int lane_id = threadIdx.x & 31;

    if (warp_id >= NUM_RANK_BLOCK)
    {
        return;
    }

    rocwmma::fragment<rocwmma::matrix_a, WMMA_M_BLOCK, WMMA_N_BLOCK, WMMA_K_BLOCK, half, rocwmma::row_major> a_frags[SUB_M_BLOCK / WMMA_M_BLOCK][K_BLOCK / WMMA_K_BLOCK];
    rocwmma::fragment<rocwmma::matrix_b, WMMA_M_BLOCK, WMMA_N_BLOCK, WMMA_K_BLOCK, half, rocwmma::row_major> b_frags[SUB_K_BLOCK / WMMA_K_BLOCK][N_BLOCK / WMMA_N_BLOCK];
    rocwmma::fragment<rocwmma::matrix_b, WMMA_M_BLOCK, WMMA_N_BLOCK, WMMA_K_BLOCK, half, rocwmma::row_major> b_frags_compute[SUB_K_BLOCK / WMMA_K_BLOCK][N_BLOCK / WMMA_N_BLOCK];
    rocwmma::fragment<rocwmma::accumulator, WMMA_M_BLOCK, WMMA_N_BLOCK, WMMA_K_BLOCK, half> c_frags[SUB_M_BLOCK / WMMA_M_BLOCK][N_BLOCK / WMMA_N_BLOCK];

    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
    {
        for (int j = 0; j < K_BLOCK / WMMA_K_BLOCK; ++j)
        {

            rocwmma::load_matrix_sync(a_frags[i][j], A + warp_id * (SUB_M_BLOCK * K_BLOCK) + i * WMMA_M_BLOCK * K_BLOCK + j * WMMA_K_BLOCK, K_BLOCK);
        }
    }

    for (int i = 0; i < SUB_K_BLOCK / WMMA_K_BLOCK; ++i)
    {
        for (int j = 0; j < N_BLOCK / WMMA_N_BLOCK; ++j)
        {

            rocwmma::load_matrix_sync(b_frags[i][j], B + warp_id * (SUB_K_BLOCK * N_BLOCK) + i * WMMA_K_BLOCK * N_BLOCK + j * WMMA_N_BLOCK, N_BLOCK);
        }
    }

    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
    {
        for (int j = 0; j < N_BLOCK / WMMA_N_BLOCK; ++j)
        {
            rocwmma::fill_fragment(c_frags[i][j], half(0.0));
        }
    }

    __syncthreads();

    for (int idx_iter = 0; idx_iter < NUM_ITER; ++idx_iter)
    {
        for (int idx_stage = 0; idx_stage < NUM_RANK_BLOCK; ++idx_stage)
        {

            if (warp_id == idx_stage)
            {

                for (int i = 0; i < SUB_K_BLOCK / WMMA_K_BLOCK; i++)
                {
                    for (int j = 0; j < N_BLOCK / WMMA_N_BLOCK; j++)
                    {
                        for (int k = 0; k < 8; k++)
                        {
                            shmem_b[lane_id * 8 + k + (i * (N_BLOCK / WMMA_N_BLOCK) + j) * WMMA_K_BLOCK * WMMA_N_BLOCK] = b_frags[i][j].x[k];
                        }
                    }
                }
            }

            __syncthreads();

            for (int i = 0; i < SUB_K_BLOCK / WMMA_K_BLOCK; i++)
            {
                for (int j = 0; j < N_BLOCK / WMMA_N_BLOCK; j++)
                {

                    for (int k = 0; k < 8; k++)
                    {
                        b_frags_compute[i][j].x[k] = shmem_b[lane_id * 8 + k + (i * (N_BLOCK / WMMA_N_BLOCK) + j) * WMMA_K_BLOCK * WMMA_N_BLOCK];
                    }
                }
            }

            __syncthreads();

            for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
            {
                for (int j = 0; j < N_BLOCK / WMMA_N_BLOCK; ++j)
                {
                    for (int k = 0; k < SUB_K_BLOCK / WMMA_K_BLOCK; ++k)
                    {
                        rocwmma::mma_sync(c_frags[i][j], a_frags[i][k + idx_stage * (SUB_K_BLOCK / WMMA_K_BLOCK)], b_frags_compute[k][j], c_frags[i][j]);
                    }
                }
            }
        }
    }

    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; i++)
    {
        for (int j = 0; j < N_BLOCK / WMMA_N_BLOCK; j++)
        {
            rocwmma::store_matrix_sync(C + warp_id * (SUB_M_BLOCK * N_BLOCK) + i * WMMA_M_BLOCK * N_BLOCK + j * WMMA_N_BLOCK, c_frags[i][j], N_BLOCK, rocwmma::mem_row_major);
        }
    }
}

int main(int argc, char *argv[])
{
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    int numSM = prop.multiProcessorCount;

    std::cout << "GPU Model: " << prop.name << std::endl;

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
    hipMalloc(&d_C, M_BLOCK * N_BLOCK * sizeof(half));

    hipMemcpy(d_A, h_A, M_BLOCK * K_BLOCK * sizeof(half), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B, K_BLOCK * N_BLOCK * sizeof(half), hipMemcpyHostToDevice);
    hipMemset(d_C, 0, M_BLOCK * N_BLOCK * sizeof(half));

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    hipFuncSetAttribute(reinterpret_cast<const void *>(block_gemm_1d_half_wmma), hipFuncAttributeMaxDynamicSharedMemorySize, shmem_size_block_gemm);

    std::cout << "Launching kernel with " << BLOCKS_PER_GRID << " blocks and "
              << NUM_ALLOC_RANK_BLOCK * WARP_SIZE << " threads per block and " << shmem_size_block_gemm << " bytes of shared memory" << std::endl;

    hipEventRecord(start);
    block_gemm_1d_half_wmma<<<blocksPerGrid, NUM_ALLOC_RANK_BLOCK * WARP_SIZE, shmem_size_block_gemm>>>(d_A, d_B, d_C, 1.0, 0.0);
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

    std::cout << "[hemeng_log],1d," << M_BLOCK << "," << N_BLOCK << "," << K_BLOCK << "," << NUM_RANK_BLOCK << "," << NUM_ALLOC_RANK_BLOCK << "," << tflops << "," << NUM_ALLOC_RANK_BLOCK * WARP_SIZE << std::endl;

    half *h_C = (half *)malloc(M_BLOCK * N_BLOCK * sizeof(half));
    hipMemcpy(h_C, d_C, M_BLOCK * N_BLOCK * sizeof(half), hipMemcpyDeviceToHost);

    half *h_C_ref = (half *)malloc(M_BLOCK * N_BLOCK * sizeof(half));

    for (int i = 0; i < M_BLOCK; i++)
    {
        for (int j = 0; j < M_BLOCK; j++)
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
