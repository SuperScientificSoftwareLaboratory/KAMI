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

#define NUM_RANK_ALL_BLOCK (NUM_RANK_BLOCK * NUM_RANK_BLOCK * NUM_RANK_BLOCK)

#ifndef NUM_ALLOC_RANK_BLOCK
#define NUM_ALLOC_RANK_BLOCK NUM_RANK_ALL_BLOCK
#endif

#define WMMA_M_BLOCK 16
#define WMMA_N_BLOCK 8
#define WMMA_K_BLOCK 8

#define SUB_M_BLOCK (M_BLOCK / NUM_RANK_BLOCK)
#define SUB_N_BLOCK (N_BLOCK / NUM_RANK_BLOCK)
#define SUB_K_BLOCK (K_BLOCK / NUM_RANK_BLOCK / NUM_RANK_BLOCK)

#define NUM_PIPE_M_BLOCK 1
#define NUM_PIPE_N_BLOCK 1
#define NUM_PIPE_K_BLOCK 1

const int shmem_size_block_gemm = (((SUB_M_BLOCK / NUM_PIPE_M_BLOCK) * (SUB_K_BLOCK / NUM_PIPE_K_BLOCK) * NUM_RANK_BLOCK * NUM_RANK_BLOCK) + ((SUB_N_BLOCK / NUM_PIPE_N_BLOCK) * (SUB_K_BLOCK / NUM_PIPE_K_BLOCK) * NUM_RANK_BLOCK * NUM_RANK_BLOCK)) * sizeof(double);

#ifndef NUM_ITER
#define NUM_ITER 1
#endif

#define THREADS_PER_BLOCK (NUM_RANK_ALL_BLOCK * WARP_SIZE)

#ifndef NUM_BATCHES
#define NUM_BATCHES 16384
#endif

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

#define FETCH_INT4(pointer) (reinterpret_cast<int4 *>(&(pointer))[0])

__global__ void batched_gemm_3d_double_mma(const double *A, const double *B, double *C, const double alpha, const double beta)
{
    extern __shared__ __align__(16) double shmem[];

    double *shmem_a = shmem;
    double *shmem_b = shmem + (SUB_M_BLOCK / NUM_PIPE_M_BLOCK) * (SUB_K_BLOCK / NUM_PIPE_K_BLOCK) * NUM_RANK_BLOCK * NUM_RANK_BLOCK;

    const unsigned char warp_id = threadIdx.x >> 5;
    const unsigned char lane_id = threadIdx.x & 31;

    const unsigned char warp_id_row = warp_id / NUM_RANK_BLOCK / NUM_RANK_BLOCK;
    const unsigned char warp_id_col = warp_id / NUM_RANK_BLOCK % NUM_RANK_BLOCK;
    const unsigned char warp_id_dep = warp_id % NUM_RANK_BLOCK;

    if (warp_id_row >= NUM_RANK_BLOCK || warp_id_col >= NUM_RANK_BLOCK)
    {
        return;
    }

    const unsigned int block_id = blockIdx.x;
    const double *A_block = A + block_id * (M_BLOCK * K_BLOCK);
    const double *B_block = B + block_id * (K_BLOCK * N_BLOCK);
    double *C_block = C + block_id * (M_BLOCK * N_BLOCK * NUM_RANK_BLOCK);

    const unsigned int dep_offset_a = warp_id_dep * WMMA_M_BLOCK * WMMA_K_BLOCK;
    const unsigned int row_offset_a = warp_id_row * (SUB_M_BLOCK / NUM_PIPE_M_BLOCK) * (SUB_K_BLOCK / NUM_PIPE_K_BLOCK) * NUM_RANK_BLOCK;

    const unsigned int dep_offset_b = warp_id_dep * WMMA_N_BLOCK * WMMA_K_BLOCK;
    const unsigned int col_offset_b = warp_id_col * (SUB_N_BLOCK / NUM_PIPE_N_BLOCK) * (SUB_K_BLOCK / NUM_PIPE_K_BLOCK) * NUM_RANK_BLOCK;

    double a_frags[SUB_M_BLOCK / WMMA_M_BLOCK][SUB_K_BLOCK / WMMA_K_BLOCK][4];
    double b_frags[SUB_N_BLOCK / WMMA_N_BLOCK][SUB_K_BLOCK / WMMA_K_BLOCK][2];
    double a_frags_compute[SUB_M_BLOCK / NUM_PIPE_M_BLOCK / WMMA_M_BLOCK][SUB_K_BLOCK / NUM_PIPE_K_BLOCK / WMMA_K_BLOCK][4];
    double b_frags_compute[SUB_N_BLOCK / NUM_PIPE_N_BLOCK / WMMA_N_BLOCK][SUB_K_BLOCK / NUM_PIPE_K_BLOCK / WMMA_K_BLOCK][2];
    double c_frags[SUB_M_BLOCK / WMMA_M_BLOCK][SUB_N_BLOCK / WMMA_N_BLOCK][4];

    for (int warp_row = 0; warp_row < SUB_M_BLOCK / WMMA_M_BLOCK; warp_row++)
    {
        for (int warp_col = 0; warp_col < SUB_K_BLOCK / WMMA_K_BLOCK; warp_col++)
        {
            int warp_row_offset = warp_id_row * SUB_M_BLOCK + warp_row * WMMA_M_BLOCK;
            int warp_col_offset = (warp_id_col * NUM_RANK_BLOCK + warp_id_dep) * SUB_K_BLOCK + warp_col * WMMA_K_BLOCK;

            int lane_row_upper_offset = lane_id >> 2;
            int lane_row_lower_offset = lane_row_upper_offset + 8;
            int lane_col_left_offset = lane_id % 4;
            int lane_col_right_offset = lane_col_left_offset + 4;

            int global_offset_upper_left = (warp_row_offset + lane_row_upper_offset) * K_BLOCK + warp_col_offset + lane_col_left_offset;
            int global_offset_upper_right = (warp_row_offset + lane_row_upper_offset) * K_BLOCK + warp_col_offset + lane_col_right_offset;
            int global_offset_lower_left = (warp_row_offset + lane_row_lower_offset) * K_BLOCK + warp_col_offset + lane_col_left_offset;
            int global_offset_lower_right = (warp_row_offset + lane_row_lower_offset) * K_BLOCK + warp_col_offset + lane_col_right_offset;

            a_frags[warp_row][warp_col][0] = A_block[global_offset_upper_left];
            a_frags[warp_row][warp_col][1] = A_block[global_offset_lower_left];
            a_frags[warp_row][warp_col][2] = A_block[global_offset_upper_right];
            a_frags[warp_row][warp_col][3] = A_block[global_offset_lower_right];
        }
    }

    for (int warp_row = 0; warp_row < SUB_N_BLOCK / WMMA_N_BLOCK; warp_row++)
    {
        for (int warp_col = 0; warp_col < SUB_K_BLOCK / WMMA_K_BLOCK; warp_col++)
        {
            int warp_row_offset = warp_id_col * SUB_N_BLOCK + warp_row * WMMA_N_BLOCK;
            int warp_col_offset = (warp_id_row * NUM_RANK_BLOCK + warp_id_dep) * SUB_K_BLOCK + warp_col * WMMA_K_BLOCK;

            int lane_row_offset = lane_id >> 2;
            int lane_col_left_offset = lane_id % 4;
            int lane_col_right_offset = lane_col_left_offset + 4;

            int global_offset_left = (warp_row_offset + lane_row_offset) * K_BLOCK + warp_col_offset + lane_col_left_offset;
            int global_offset_right = (warp_row_offset + lane_row_offset) * K_BLOCK + warp_col_offset + lane_col_right_offset;

            b_frags[warp_row][warp_col][0] = B_block[global_offset_left];
            b_frags[warp_row][warp_col][1] = B_block[global_offset_right];
        }
    }

    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
    {
        for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; ++j)
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
            for (int idx_pipe_k = 0; idx_pipe_k < NUM_PIPE_K_BLOCK; idx_pipe_k++)
            {
#pragma unroll
                for (int idx_pipe_n = 0; idx_pipe_n < NUM_PIPE_N_BLOCK; idx_pipe_n++)
                {
                    for (int idx_pipe_m = 0; idx_pipe_m < NUM_PIPE_M_BLOCK; idx_pipe_m++)
                    {
                        if (warp_id_col == idx_stage)
                        {
#pragma unroll
                            for (int i = 0; i < (SUB_M_BLOCK / NUM_PIPE_M_BLOCK) / WMMA_M_BLOCK; ++i)
                            {
#pragma unroll
                                for (int j = 0; j < SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK; ++j)
                                {
                                    int real_i = i + idx_pipe_m * (SUB_M_BLOCK / WMMA_M_BLOCK / NUM_PIPE_M_BLOCK);
                                    int real_j = j + idx_pipe_k * (SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK);

                                    int smem_offset = lane_id * 4 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_M_BLOCK * WMMA_K_BLOCK + dep_offset_a + row_offset_a;

                                    shmem_a[smem_offset] = a_frags[real_i][real_j][0];
                                    shmem_a[smem_offset + 1] = a_frags[real_i][real_j][1];
                                    shmem_a[smem_offset + 2] = a_frags[real_i][real_j][2];
                                    shmem_a[smem_offset + 3] = a_frags[real_i][real_j][3];
                                }
                            }
                        }

                        if (warp_id_row == idx_stage)
                        {
#pragma unroll
                            for (int i = 0; i < (SUB_N_BLOCK / NUM_PIPE_N_BLOCK) / WMMA_N_BLOCK; ++i)
                            {
#pragma unroll
                                for (int j = 0; j < SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK; ++j)
                                {
                                    int real_i = i + idx_pipe_n * (SUB_N_BLOCK / WMMA_N_BLOCK / NUM_PIPE_N_BLOCK);
                                    int real_j = j + idx_pipe_k * (SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK);

                                    int smem_offset = lane_id * 2 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK) + j) * (NUM_RANK_BLOCK * WMMA_N_BLOCK * WMMA_K_BLOCK) + dep_offset_b + col_offset_b;

                                    shmem_b[smem_offset] = b_frags[real_i][real_j][0];
                                    shmem_b[smem_offset + 1] = b_frags[real_i][real_j][1];
                                }
                            }
                        }

                        __syncthreads();

#pragma unroll
                        for (int i = 0; i < (SUB_M_BLOCK / NUM_PIPE_M_BLOCK) / WMMA_M_BLOCK; ++i)
                        {
#pragma unroll
                            for (int j = 0; j < SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK; ++j)
                            {

                                int smem_offset = lane_id * 4 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_M_BLOCK * WMMA_K_BLOCK + dep_offset_a + row_offset_a;

                                a_frags_compute[i][j][0] = shmem_a[smem_offset];
                                a_frags_compute[i][j][1] = shmem_a[smem_offset + 1];
                                a_frags_compute[i][j][2] = shmem_a[smem_offset + 2];
                                a_frags_compute[i][j][3] = shmem_a[smem_offset + 3];
                            }
                        }

#pragma unroll
                        for (int i = 0; i < (SUB_N_BLOCK / NUM_PIPE_N_BLOCK) / WMMA_N_BLOCK; ++i)
                        {
#pragma unroll
                            for (int j = 0; j < SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK; ++j)
                            {

                                int smem_offset = lane_id * 2 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK) + j) * (NUM_RANK_BLOCK * WMMA_N_BLOCK * WMMA_K_BLOCK) + dep_offset_b + col_offset_b;

                                b_frags_compute[i][j][0] = shmem_b[smem_offset];
                                b_frags_compute[i][j][1] = shmem_b[smem_offset + 1];
                            }
                        }

                        __syncthreads();

#pragma unroll
                        for (int i = 0; i < (SUB_M_BLOCK / NUM_PIPE_M_BLOCK) / WMMA_M_BLOCK; i++)
                        {
#pragma unroll
                            for (int j = 0; j < (SUB_N_BLOCK / NUM_PIPE_N_BLOCK) / WMMA_N_BLOCK; j++)
                            {
#pragma unroll
                                for (int k = 0; k < SUB_K_BLOCK / WMMA_K_BLOCK / NUM_PIPE_K_BLOCK; k++)
                                {

                                    int real_i = i + idx_pipe_m * (SUB_M_BLOCK / WMMA_M_BLOCK / NUM_PIPE_M_BLOCK);
                                    int real_j = j + idx_pipe_n * (SUB_N_BLOCK / WMMA_N_BLOCK / NUM_PIPE_N_BLOCK);
                                    mma_m16n8k8(c_frags[real_i][real_j], a_frags_compute[i][k], b_frags_compute[j][k]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    __syncthreads();

    for (int warp_row = 0; warp_row < SUB_M_BLOCK / WMMA_M_BLOCK; warp_row++)
    {
        for (int warp_col = 0; warp_col < SUB_N_BLOCK / WMMA_N_BLOCK; warp_col++)
        {
            int warp_row_offset = warp_id_row * SUB_M_BLOCK + warp_row * WMMA_M_BLOCK;
            int warp_col_offset = warp_id_col * SUB_N_BLOCK + warp_col * WMMA_N_BLOCK;

            int lane_row_upper_offset = lane_id >> 2;
            int lane_row_lower_offset = lane_row_upper_offset + 8;
            int lane_col_offset = (lane_id % 4) * 2;

            int global_offset_upper = (warp_row_offset + lane_row_upper_offset) * N_BLOCK + warp_col_offset + lane_col_offset + warp_id_dep * M_BLOCK * N_BLOCK;
            int global_offset_lower = (warp_row_offset + lane_row_lower_offset) * N_BLOCK + warp_col_offset + lane_col_offset + warp_id_dep * M_BLOCK * N_BLOCK;

            C_block[global_offset_upper] = c_frags[warp_row][warp_col][0];
            C_block[global_offset_upper + 1] = c_frags[warp_row][warp_col][1];
            C_block[global_offset_lower] = c_frags[warp_row][warp_col][2];
            C_block[global_offset_lower + 1] = c_frags[warp_row][warp_col][3];
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

    double *h_A = (double *)malloc(sizeof(double) * M_BLOCK * K_BLOCK * NUM_BATCHES);
    double *h_B = (double *)malloc(sizeof(double) * K_BLOCK * N_BLOCK * NUM_BATCHES);

#pragma omp parallel for
    for (unsigned long i = 0; i < (unsigned long)M_BLOCK * K_BLOCK * NUM_BATCHES; i++)
    {
        h_A[i] = 1;
        unsigned int local_seed = i;
        h_A[i] = rand_r(&local_seed) % 3;
    }

#pragma omp parallel for
    for (unsigned long i = 0; i < (unsigned long)K_BLOCK * N_BLOCK * NUM_BATCHES; i++)
    {
        h_B[i] = 1;
        unsigned int local_seed = i;
        h_B[i] = rand_r(&local_seed) % 3;
    }

    double *d_A, *d_B;
    double *d_C;
    cudaMalloc(&d_A, sizeof(double) * M_BLOCK * K_BLOCK * NUM_BATCHES);
    cudaMalloc(&d_B, sizeof(double) * K_BLOCK * N_BLOCK * NUM_BATCHES);
    cudaMalloc(&d_C, sizeof(double) * M_BLOCK * N_BLOCK * NUM_BATCHES * NUM_RANK_BLOCK);

    cudaMemcpy(d_A, h_A, sizeof(double) * M_BLOCK * K_BLOCK * NUM_BATCHES, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(double) * K_BLOCK * N_BLOCK * NUM_BATCHES, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, sizeof(double) * M_BLOCK * N_BLOCK * NUM_BATCHES * NUM_RANK_BLOCK);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaFuncSetAttribute(batched_gemm_3d_double_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size_block_gemm);

    std::cout << "Launching kernel with " << NUM_BATCHES << " blocks and "
              << NUM_ALLOC_RANK_BLOCK * WARP_SIZE << " threads per block and " << shmem_size_block_gemm << " bytes of shared memory" << std::endl;

    cudaEventRecord(start);
    batched_gemm_3d_double_mma<<<NUM_BATCHES, NUM_ALLOC_RANK_BLOCK * WARP_SIZE, shmem_size_block_gemm>>>(d_A, d_B, d_C, 1.0, 0.0);
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
    long long totalOps = numOpsPerMatrix * NUM_BATCHES * NUM_ITER;
    double gflops = static_cast<double>(totalOps) / (milliseconds * 1e6);
    double tflops = gflops / 1000.0f;

    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS (" << tflops << " TFLOPS)" << std::endl;
    std::cout << "[hemeng_log],3d," << M_BLOCK << "," << N_BLOCK << "," << K_BLOCK << "," << NUM_BATCHES << "," << tflops << "," << THREADS_PER_BLOCK << std::endl;

    double *h_C = (double *)malloc(sizeof(double) * M_BLOCK * N_BLOCK * NUM_BATCHES * NUM_RANK_BLOCK);
    cudaMemcpy(h_C, d_C, sizeof(double) * M_BLOCK * N_BLOCK * NUM_BATCHES * NUM_RANK_BLOCK, cudaMemcpyDeviceToHost);

#pragma omp parallel for
    for (unsigned long batch = 0; batch < NUM_BATCHES; batch++)
    {
        for (int i = 0; i < M_BLOCK; i++)
        {
            for (int j = 0; j < N_BLOCK; j++)
            {
                for (int k = 1; k < NUM_RANK_BLOCK; k++)
                {
                    h_C[i * N_BLOCK + j + batch * M_BLOCK * N_BLOCK * NUM_RANK_BLOCK] += h_C[i * N_BLOCK + j + k * M_BLOCK * N_BLOCK + batch * M_BLOCK * N_BLOCK * NUM_RANK_BLOCK];
                }
            }
        }
    }

    double *h_C_ref = (double *)malloc(sizeof(double) * M_BLOCK * N_BLOCK * NUM_BATCHES);

#pragma omp parallel for
    for (unsigned long batch = 0; batch < NUM_BATCHES; batch++)
    {
        for (unsigned long i = 0; i < M_BLOCK; i++)
        {
            for (unsigned long j = 0; j < M_BLOCK; j++)
            {
                h_C_ref[i * N_BLOCK + j + batch * M_BLOCK * N_BLOCK] = 0;
                for (unsigned long k = 0; k < K_BLOCK; k++)
                {
                    h_C_ref[i * N_BLOCK + j + batch * M_BLOCK * N_BLOCK] += h_A[i * K_BLOCK + k + batch * M_BLOCK * K_BLOCK] * h_B[j * K_BLOCK + k + batch * N_BLOCK * K_BLOCK];
                }
            }
        }
    }

    int error = 0;
#pragma omp parallel for
    for (unsigned long i = 0; i < (unsigned long)NUM_BATCHES; i++)
    {
        for (int j = 0; j < M_BLOCK * N_BLOCK; j++)
        {
            if (fabs(h_C[j + i * M_BLOCK * N_BLOCK * NUM_RANK_BLOCK] / NUM_ITER - h_C_ref[j + i * M_BLOCK * N_BLOCK]) > 1e-6)
            {

                error = 1;
            }
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
