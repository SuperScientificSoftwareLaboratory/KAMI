
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

#define NUM_RANK_ALL_BLOCK (NUM_RANK_BLOCK * NUM_RANK_BLOCK * NUM_RANK_BLOCK)

#ifndef NUM_ALLOC_RANK_BLOCK
#define NUM_ALLOC_RANK_BLOCK NUM_RANK_ALL_BLOCK
#endif

#define WMMA_M_BLOCK 16
#define WMMA_N_BLOCK 8
#define WMMA_K_BLOCK 16

#define SUB_M_BLOCK (M_BLOCK / NUM_RANK_BLOCK)
#define SUB_N_BLOCK (N_BLOCK / NUM_RANK_BLOCK)
#define SUB_K_BLOCK (K_BLOCK / NUM_RANK_BLOCK / NUM_RANK_BLOCK)

const int shmem_size_block_gemm = ((SUB_M_BLOCK * SUB_K_BLOCK * NUM_RANK_BLOCK * NUM_RANK_BLOCK) + (SUB_K_BLOCK * SUB_N_BLOCK * NUM_RANK_BLOCK * NUM_RANK_BLOCK)) * sizeof(half);

#define NUM_ITER 10000

#define THREADS_PER_BLOCK (NUM_RANK_ALL_BLOCK * WARP_SIZE)

#define BLOCKS_PER_GRID 1

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

__global__ void block_gemm_3d_half_mma(const half *A, const half *B, half *C, const half alpha, const half beta)
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

  uint32_t a_frags[SUB_M_BLOCK / WMMA_M_BLOCK][SUB_K_BLOCK / WMMA_K_BLOCK][4];
  uint32_t b_frags[SUB_N_BLOCK / WMMA_N_BLOCK][SUB_K_BLOCK / WMMA_K_BLOCK][2];
  uint32_t a_frags_compute[SUB_M_BLOCK / WMMA_M_BLOCK][SUB_K_BLOCK / WMMA_K_BLOCK][4];
  uint32_t b_frags_compute[SUB_N_BLOCK / WMMA_N_BLOCK][SUB_K_BLOCK / WMMA_K_BLOCK][2];
  uint32_t c_frags[SUB_M_BLOCK / WMMA_M_BLOCK][SUB_N_BLOCK / WMMA_N_BLOCK][2];

  for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
  {
    for (int j = 0; j < SUB_K_BLOCK / WMMA_K_BLOCK; ++j)
    {
      int block_row = warp_row * (SUB_M_BLOCK / WMMA_M_BLOCK) + i;
      int block_col = (warp_col * NUM_RANK_BLOCK + warp_dep) * (SUB_K_BLOCK / WMMA_K_BLOCK) + j;

      int lane_row_upper = lane_id >> 2;
      int lane_row_lower = lane_row_upper + 8;
      int lane_col_left = (lane_id % 4) * 2;
      int lane_col_right = lane_col_left + 8;

      int global_offset_upper_left = (block_row * WMMA_M_BLOCK + lane_row_upper) * K_BLOCK + block_col * WMMA_K_BLOCK + lane_col_left;
      int global_offset_lower_left = (block_row * WMMA_M_BLOCK + lane_row_lower) * K_BLOCK + block_col * WMMA_K_BLOCK + lane_col_left;
      int global_offset_upper_right = (block_row * WMMA_M_BLOCK + lane_row_upper) * K_BLOCK + block_col * WMMA_K_BLOCK + lane_col_right;
      int global_offset_lower_right = (block_row * WMMA_M_BLOCK + lane_row_lower) * K_BLOCK + block_col * WMMA_K_BLOCK + lane_col_right;

      a_frags[i][j][0] = *(uint32_t *)(A + global_offset_upper_left);
      a_frags[i][j][1] = *(uint32_t *)(A + global_offset_lower_left);
      a_frags[i][j][2] = *(uint32_t *)(A + global_offset_upper_right);
      a_frags[i][j][3] = *(uint32_t *)(A + global_offset_lower_right);
    }
  }

  for (int i = 0; i < SUB_N_BLOCK / WMMA_N_BLOCK; ++i)
  {
    for (int j = 0; j < SUB_K_BLOCK / WMMA_K_BLOCK; ++j)
    {
      int block_row = warp_col * (SUB_N_BLOCK / WMMA_N_BLOCK) + i;
      int block_col = (warp_row * NUM_RANK_BLOCK + warp_dep) * (SUB_K_BLOCK / WMMA_K_BLOCK) + j;

      int lane_row = lane_id >> 2;
      int lane_col_left = (lane_id % 4) * 2;
      int lane_col_right = lane_col_left + 8;

      int global_offset_left = (block_row * WMMA_N_BLOCK + lane_row) * K_BLOCK + block_col * WMMA_K_BLOCK + lane_col_left;
      int global_offset_right = (block_row * WMMA_N_BLOCK + lane_row) * K_BLOCK + block_col * WMMA_K_BLOCK + lane_col_right;

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

  unsigned int comm_comp_total_cycles = 0, comm_comm_total_cycles = 0, comm_total_cycles = 0, comp_cudacore_total_cycles = 0, comp_tensorcore_total_cycles = 0, comp_total_cycles = 0;

  for (int idx_iter = 0; idx_iter < NUM_ITER; ++idx_iter)
  {
    for (int idx_stage = 0; idx_stage < NUM_RANK_BLOCK; ++idx_stage)
    {

      unsigned int comm_start_cycle = clock();
      if (warp_col == idx_stage)
      {
        for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
        {
          for (int j = 0; j < SUB_K_BLOCK / WMMA_K_BLOCK; ++j)
          {

            int dep_offset = warp_dep * WMMA_M_BLOCK * WMMA_K_BLOCK;
            int row_offset = warp_row * SUB_M_BLOCK * SUB_K_BLOCK * NUM_RANK_BLOCK;

            uint32_t *a1 = (uint32_t *)(shmem_a + lane_id * 8 + 0 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_M_BLOCK * WMMA_K_BLOCK + dep_offset + row_offset);
            uint32_t *a2 = (uint32_t *)(shmem_a + lane_id * 8 + 2 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_M_BLOCK * WMMA_K_BLOCK + dep_offset + row_offset);
            uint32_t *a3 = (uint32_t *)(shmem_a + lane_id * 8 + 4 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_M_BLOCK * WMMA_K_BLOCK + dep_offset + row_offset);
            uint32_t *a4 = (uint32_t *)(shmem_a + lane_id * 8 + 6 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_M_BLOCK * WMMA_K_BLOCK + dep_offset + row_offset);

            *a1 = a_frags[i][j][0];
            *a2 = a_frags[i][j][1];
            *a3 = a_frags[i][j][2];
            *a4 = a_frags[i][j][3];
          }
        }
      }

      if (warp_row == idx_stage)
      {
        for (int i = 0; i < SUB_N_BLOCK / WMMA_N_BLOCK; ++i)
        {
          for (int j = 0; j < SUB_K_BLOCK / WMMA_K_BLOCK; ++j)
          {

            int dep_offset = warp_dep * WMMA_N_BLOCK * WMMA_K_BLOCK;
            int col_offset = warp_col * SUB_K_BLOCK * SUB_N_BLOCK * NUM_RANK_BLOCK;

            uint32_t *b1 = (uint32_t *)(shmem_b + lane_id * 4 + 0 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_N_BLOCK * WMMA_K_BLOCK + dep_offset + col_offset);
            uint32_t *b2 = (uint32_t *)(shmem_b + lane_id * 4 + 2 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_N_BLOCK * WMMA_K_BLOCK + dep_offset + col_offset);

            *b1 = b_frags[i][j][0];
            *b2 = b_frags[i][j][1];
          }
        }
      }

      __syncthreads();

      unsigned int start_cycle = clock();

      for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
      {
        for (int j = 0; j < SUB_K_BLOCK / WMMA_K_BLOCK; ++j)
        {

          int dep_offset = warp_dep * WMMA_M_BLOCK * WMMA_K_BLOCK;
          int row_offset = warp_row * SUB_M_BLOCK * SUB_K_BLOCK * NUM_RANK_BLOCK;

          uint32_t *a1 = (uint32_t *)(shmem_a + lane_id * 8 + 0 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_M_BLOCK * WMMA_K_BLOCK + dep_offset + row_offset);
          uint32_t *a2 = (uint32_t *)(shmem_a + lane_id * 8 + 2 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_M_BLOCK * WMMA_K_BLOCK + dep_offset + row_offset);
          uint32_t *a3 = (uint32_t *)(shmem_a + lane_id * 8 + 4 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_M_BLOCK * WMMA_K_BLOCK + dep_offset + row_offset);
          uint32_t *a4 = (uint32_t *)(shmem_a + lane_id * 8 + 6 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_M_BLOCK * WMMA_K_BLOCK + dep_offset + row_offset);

          a_frags_compute[i][j][0] = *a1;
          a_frags_compute[i][j][1] = *a2;
          a_frags_compute[i][j][2] = *a3;
          a_frags_compute[i][j][3] = *a4;
        }
      }

      for (int i = 0; i < SUB_N_BLOCK / WMMA_N_BLOCK; ++i)
      {
        for (int j = 0; j < SUB_K_BLOCK / WMMA_K_BLOCK; ++j)
        {

          int dep_offset = warp_dep * WMMA_N_BLOCK * WMMA_K_BLOCK;
          int col_offset = warp_col * SUB_K_BLOCK * SUB_N_BLOCK * NUM_RANK_BLOCK;
          uint32_t *b1 = (uint32_t *)(shmem_b + lane_id * 4 + 0 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_N_BLOCK * WMMA_K_BLOCK + dep_offset + col_offset);
          uint32_t *b2 = (uint32_t *)(shmem_b + lane_id * 4 + 2 + (i * (SUB_K_BLOCK / WMMA_K_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_N_BLOCK * WMMA_K_BLOCK + dep_offset + col_offset);

          b_frags_compute[i][j][0] = *b1;
          b_frags_compute[i][j][1] = *b2;
        }
      }
      comm_total_cycles += clock() - comm_start_cycle;
      __syncthreads();

      unsigned int comp_start_cycle = clock();
      for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
      {
        for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; ++j)
        {
          for (int k = 0; k < SUB_K_BLOCK / WMMA_K_BLOCK; ++k)
          {
            unsigned int comp_tensorcore_start_cycle = clock();
            mma_m16n8k16_fp16(c_frags[i][j], a_frags_compute[i][k], b_frags_compute[j][k]);
            comp_tensorcore_total_cycles += clock() - comp_tensorcore_start_cycle;
          }
        }
      }
      comp_total_cycles += clock() - comp_start_cycle;
    }
  }
  if (lane_id == 0)
  {
    printf("warp_id: 3d,%d,%d,%d\n", warp_id, comm_total_cycles / (NUM_ITER), comp_total_cycles / (NUM_ITER));
  }

  for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; i++)
  {
    for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; ++j)
    {
      int block_row = warp_row * (SUB_M_BLOCK / WMMA_M_BLOCK) + i;
      int block_col = warp_col * (SUB_N_BLOCK / WMMA_N_BLOCK) + j;

      int real_row_upper = block_row * WMMA_M_BLOCK + (lane_id >> 2);
      int real_row_lower = real_row_upper + 8;
      int real_col = block_col * WMMA_N_BLOCK + (lane_id % 4) * 2;

      int global_offset_upper = real_row_upper * N_BLOCK + real_col + warp_dep * M_BLOCK * N_BLOCK;
      int global_offset_lower = real_row_lower * N_BLOCK + real_col + warp_dep * M_BLOCK * N_BLOCK;

      *(uint32_t *)(C + global_offset_upper) = c_frags[i][j][0];
      *(uint32_t *)(C + global_offset_lower) = c_frags[i][j][1];
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
    h_A[i] = 1;
    h_A[i] = rand() % 2 + 1.0;
  }

  for (int i = 0; i < K_BLOCK * N_BLOCK; i++)
  {
    h_B[i] = 1;
    h_B[i] = rand() % 2 + 1.0;
  }

  half *d_A, *d_B;
  half *d_C;

  cudaMalloc(&d_A, M_BLOCK * K_BLOCK * sizeof(half));
  cudaMalloc(&d_B, K_BLOCK * N_BLOCK * sizeof(half));
  cudaMalloc(&d_C, M_BLOCK * N_BLOCK * NUM_RANK_BLOCK * sizeof(half));

  cudaMemcpy(d_A, h_A, M_BLOCK * K_BLOCK * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, K_BLOCK * N_BLOCK * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemset(d_C, 0, M_BLOCK * N_BLOCK * NUM_RANK_BLOCK * sizeof(half));

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaFuncSetAttribute(block_gemm_3d_half_mma, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size_block_gemm);

  std::cout << "Launching kernel with " << BLOCKS_PER_GRID << " blocks and "
            << NUM_ALLOC_RANK_BLOCK * WARP_SIZE << " threads per block and " << shmem_size_block_gemm << " bytes of shared memory" << std::endl;

  cudaEventRecord(start);
  block_gemm_3d_half_mma<<<BLOCKS_PER_GRID, NUM_ALLOC_RANK_BLOCK * WARP_SIZE, shmem_size_block_gemm>>>(d_A, d_B, d_C, 1.0, 0.0);
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

  std::cout << "[hemeng_log],3d," << M_BLOCK << "," << NUM_RANK_ALL_BLOCK << "," << NUM_ALLOC_RANK_BLOCK << "," << milliseconds << "," << tflops << std::endl;

  half *h_C = (half *)malloc(M_BLOCK * N_BLOCK * NUM_RANK_BLOCK * sizeof(half));
  cudaMemcpy(h_C, d_C, M_BLOCK * N_BLOCK * NUM_RANK_BLOCK * sizeof(half), cudaMemcpyDeviceToHost);

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
        h_C_ref[i * N_BLOCK + j] = (double)h_C_ref[i * N_BLOCK + j] + (double)h_A[i * K_BLOCK + k] * (double)h_B[j * K_BLOCK + k];
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
