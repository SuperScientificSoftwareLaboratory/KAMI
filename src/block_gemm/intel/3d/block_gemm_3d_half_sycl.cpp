
#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <iostream>
#include <cmath>

#define WARP_SIZE 16

#ifndef M_BLOCK
#define M_BLOCK 128
#define N_BLOCK 128
#define K_BLOCK 128
#endif

#ifndef NUM_RANK_BLOCK
#define NUM_RANK_BLOCK 2
#endif

#define NUM_RANK_ALL_BLOCK (NUM_RANK_BLOCK * NUM_RANK_BLOCK * NUM_RANK_BLOCK)

#define WMMA_M_BLOCK 8
#define WMMA_N_BLOCK 16
#define WMMA_K_BLOCK 16

#define SUB_M_BLOCK (M_BLOCK / NUM_RANK_BLOCK)
#define SUB_N_BLOCK (N_BLOCK / NUM_RANK_BLOCK)
#define SUB_K_BLOCK (K_BLOCK / NUM_RANK_BLOCK / NUM_RANK_BLOCK)

const int shmem_size_block_gemm = ((SUB_M_BLOCK * SUB_K_BLOCK * NUM_RANK_BLOCK * NUM_RANK_BLOCK) + (SUB_K_BLOCK * SUB_N_BLOCK * NUM_RANK_BLOCK * NUM_RANK_BLOCK));

#define NUM_ITER 1000

#define THREADS_PER_BLOCK (WARP_SIZE * NUM_RANK_ALL_BLOCK)

#define BLOCKS_PER_GRID 1000

void block_gemm_3d_half_sycl(const sycl::half *A, const sycl::half *B,
                             float *C, const sycl::half alpha,
                             const sycl::half beta,
                             const sycl::nd_item<3> &item_ct1,
                             sycl::half *dpct_local,
                             int *dpct_warpsize)
{
    auto shmem_a = (sycl::half *)dpct_local;
    auto shmem_b = (sycl::half *)dpct_local + SUB_M_BLOCK * SUB_K_BLOCK * NUM_RANK_BLOCK * NUM_RANK_BLOCK;

    const unsigned int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
    const unsigned int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;

    const unsigned int warp_row = warp_id / NUM_RANK_BLOCK / NUM_RANK_BLOCK;
    const unsigned int warp_col = warp_id / NUM_RANK_BLOCK % NUM_RANK_BLOCK;
    const unsigned int warp_dep = warp_id % NUM_RANK_BLOCK;

    *dpct_warpsize = item_ct1.get_sub_group().get_local_range().get(0);

    dpct::experimental::matrix::joint_matrix<
        dpct::experimental::matrix::a, WMMA_M_BLOCK, WMMA_N_BLOCK, WMMA_K_BLOCK,
        sycl::half, dpct::experimental::matrix::row_major>
        a_frags[SUB_M_BLOCK / WMMA_M_BLOCK][SUB_K_BLOCK / WMMA_K_BLOCK];

    dpct::experimental::matrix::joint_matrix<
        dpct::experimental::matrix::a, WMMA_M_BLOCK, WMMA_N_BLOCK, WMMA_K_BLOCK,
        sycl::half, dpct::experimental::matrix::row_major>
        a_frags_compute[SUB_M_BLOCK / WMMA_M_BLOCK][SUB_K_BLOCK / WMMA_K_BLOCK];

    dpct::experimental::matrix::joint_matrix<
        dpct::experimental::matrix::b, WMMA_M_BLOCK, WMMA_N_BLOCK, WMMA_K_BLOCK,
        sycl::half, dpct::experimental::matrix::row_major>
        b_frags[SUB_K_BLOCK / WMMA_K_BLOCK][SUB_N_BLOCK / WMMA_N_BLOCK];

    dpct::experimental::matrix::joint_matrix<
        dpct::experimental::matrix::b, WMMA_M_BLOCK, WMMA_N_BLOCK, WMMA_K_BLOCK,
        sycl::half, dpct::experimental::matrix::row_major>
        b_frags_compute[SUB_K_BLOCK / WMMA_K_BLOCK][SUB_N_BLOCK / WMMA_N_BLOCK];

    dpct::experimental::matrix::joint_matrix<
        dpct::experimental::matrix::accumulator, WMMA_M_BLOCK, WMMA_N_BLOCK,
        WMMA_K_BLOCK, float>
        c_frags[SUB_M_BLOCK / WMMA_M_BLOCK][SUB_N_BLOCK / WMMA_N_BLOCK];

    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
    {
        for (int j = 0; j < SUB_K_BLOCK / WMMA_K_BLOCK; ++j)
        {
            int real_row = warp_row * (SUB_M_BLOCK / WMMA_M_BLOCK) + i;
            int real_col = (warp_col * NUM_RANK_BLOCK + warp_dep) * (SUB_K_BLOCK / WMMA_K_BLOCK) + j;
            int real_idx = real_row * WMMA_M_BLOCK * K_BLOCK + real_col * WMMA_K_BLOCK;

            sycl::ext::oneapi::experimental::matrix::joint_matrix_load(
                item_ct1.get_sub_group(), a_frags[i][j].get(),
                sycl::address_space_cast<
                    sycl::access::address_space::generic_space,
                    sycl::access::decorated::no, const sycl::half>(
                    A + real_idx),
                K_BLOCK);
        }
    }

    for (int i = 0; i < SUB_K_BLOCK / WMMA_K_BLOCK; ++i)
    {
        for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; ++j)
        {
            int real_row = (warp_row * NUM_RANK_BLOCK + warp_dep) * (SUB_K_BLOCK / WMMA_K_BLOCK) + i;
            int real_col = warp_col * (SUB_N_BLOCK / WMMA_N_BLOCK) + j;
            int real_idx = real_row * WMMA_K_BLOCK * N_BLOCK + real_col * WMMA_N_BLOCK;

            sycl::ext::oneapi::experimental::matrix::joint_matrix_load(
                item_ct1.get_sub_group(), b_frags[i][j].get(),
                sycl::address_space_cast<
                    sycl::access::address_space::generic_space,
                    sycl::access::decorated::no, const sycl::half>(
                    B + real_idx),
                N_BLOCK);
        }
    }

    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
    {
        for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; ++j)
        {
            sycl::ext::oneapi::experimental::matrix::joint_matrix_fill(
                item_ct1.get_sub_group(), c_frags[i][j].get(), 0.0);
        }
    }

    item_ct1.barrier();

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
                        for (int k = 0; k < (WMMA_K_BLOCK * WMMA_M_BLOCK / WARP_SIZE); k++)
                        {
                            shmem_a[lane_id * (WMMA_K_BLOCK * WMMA_M_BLOCK / WARP_SIZE) + k + (i * (SUB_K_BLOCK / WMMA_K_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_M_BLOCK * WMMA_K_BLOCK + dep_offset + row_offset] = a_frags[i][j].x[k];
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
                        for (int k = 0; k < (WMMA_K_BLOCK * WMMA_N_BLOCK / WARP_SIZE); k++)
                        {
                            shmem_b[lane_id * (WMMA_K_BLOCK * WMMA_N_BLOCK / WARP_SIZE) + k + (i * (SUB_N_BLOCK / WMMA_N_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_K_BLOCK * WMMA_N_BLOCK + dep_offset + col_offset] = b_frags[i][j].x[k];
                        }
                    }
                }
            }

            item_ct1.barrier();

            for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
            {
                for (int j = 0; j < SUB_K_BLOCK / WMMA_K_BLOCK; ++j)
                {
                    int dep_offset = warp_dep * WMMA_M_BLOCK * WMMA_K_BLOCK;
                    int row_offset = warp_row * SUB_M_BLOCK * SUB_K_BLOCK * NUM_RANK_BLOCK;
                    for (int k = 0; k < (WMMA_M_BLOCK * WMMA_K_BLOCK / WARP_SIZE); k++)
                    {
                        a_frags_compute[i][j].x[k] = shmem_a[lane_id * (WMMA_M_BLOCK * WMMA_K_BLOCK / WARP_SIZE) + k + (i * (SUB_K_BLOCK / WMMA_K_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_M_BLOCK * WMMA_K_BLOCK + dep_offset + row_offset];
                    }
                }
            }

            for (int i = 0; i < SUB_K_BLOCK / WMMA_K_BLOCK; ++i)
            {
                for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; ++j)
                {
                    int dep_offset = warp_dep * WMMA_K_BLOCK * WMMA_N_BLOCK;
                    int col_offset = warp_col * SUB_K_BLOCK * SUB_N_BLOCK * NUM_RANK_BLOCK;
                    for (int k = 0; k < (WMMA_N_BLOCK * WMMA_K_BLOCK / WARP_SIZE); k++)
                    {
                        b_frags_compute[i][j].x[k] = shmem_b[lane_id * (WMMA_N_BLOCK * WMMA_K_BLOCK / WARP_SIZE) + k + (i * (SUB_N_BLOCK / WMMA_N_BLOCK) + j) * NUM_RANK_BLOCK * WMMA_K_BLOCK * WMMA_N_BLOCK + dep_offset + col_offset];
                    }
                }
            }

            item_ct1.barrier();

#pragma unroll
            for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; ++i)
            {
#pragma unroll
                for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; ++j)
                {
#pragma unroll
                    for (int k = 0; k < SUB_K_BLOCK / WMMA_K_BLOCK; ++k)
                    {
                        sycl::ext::oneapi::experimental::matrix::
                            joint_matrix_mad(
                                item_ct1.get_sub_group(),
                                c_frags[i][j].get(),
                                a_frags_compute[i][k].get(),
                                b_frags_compute[k][j].get(),
                                c_frags[i][j].get());
                    }
                }
            }
        }
    }

    for (int i = 0; i < SUB_M_BLOCK / WMMA_M_BLOCK; i++)
    {
        for (int j = 0; j < SUB_N_BLOCK / WMMA_N_BLOCK; j++)
        {

            int real_row = warp_row * (SUB_M_BLOCK / WMMA_M_BLOCK) + i;
            int real_col = warp_col * (SUB_N_BLOCK / WMMA_N_BLOCK) + j;
            int real_idx = real_row * WMMA_M_BLOCK * N_BLOCK + real_col * WMMA_N_BLOCK;
            int global_idx = real_idx + warp_dep * M_BLOCK * N_BLOCK;

            sycl::ext::oneapi::experimental::matrix::joint_matrix_store(
                item_ct1.get_sub_group(), c_frags[i][j].get(),
                sycl::address_space_cast<
                    sycl::access::address_space::generic_space,
                    sycl::access::decorated::no, float>(
                    C + global_idx),
                N_BLOCK,
                sycl::ext::oneapi::experimental::matrix::layout::row_major);
        }
    }
}

int main(int argc, char *argv[])
{
    int device_id = 0;
    dpct::device_info prop;

    dpct::select_device(device_id);
    dpct::get_device(device_id).get_device_info(prop);
    std::cout << "GPU " << prop.get_device_id() << " Model: " << prop.get_name()
              << std::endl;

    sycl::half *h_A =
        (sycl::half *)malloc(M_BLOCK * K_BLOCK * sizeof(sycl::half));
    sycl::half *h_B =
        (sycl::half *)malloc(K_BLOCK * N_BLOCK * sizeof(sycl::half));

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

    int *d_warpsize = sycl::malloc_device<int>(1, dpct::get_in_order_queue());

    sycl::half *d_A, *d_B;
    float *d_C;

    d_A = sycl::malloc_device<sycl::half>(M_BLOCK * K_BLOCK,
                                          dpct::get_in_order_queue());
    d_B = sycl::malloc_device<sycl::half>(K_BLOCK * N_BLOCK,
                                          dpct::get_in_order_queue());
    d_C = sycl::malloc_device<float>(M_BLOCK * N_BLOCK * NUM_RANK_BLOCK,
                                     dpct::get_in_order_queue());

    dpct::get_in_order_queue().memcpy(d_A, h_A,
                                      M_BLOCK * K_BLOCK * sizeof(sycl::half));
    dpct::get_in_order_queue()
        .memcpy(d_B, h_B, K_BLOCK * N_BLOCK * sizeof(sycl::half))
        .wait();
    dpct::get_in_order_queue()
        .memset(d_C, 0, M_BLOCK * N_BLOCK * NUM_RANK_BLOCK * sizeof(float))
        .wait();

    dpct::event_ptr start, stop;
    start = new sycl::event();
    stop = new sycl::event();

    std::cout << "Launching kernel with " << BLOCKS_PER_GRID << " blocks and "
              << THREADS_PER_BLOCK << " threads per block and " << shmem_size_block_gemm * sizeof(sycl::half) << " bytes of shared memory" << std::endl;

    std::cout << "Local Memory Size: "
              << dpct::get_in_order_queue().get_device().get_info<sycl::info::device::local_mem_size>()
              << std::endl;

    dpct::sync_barrier(start);

    {
        dpct::has_capability_or_fail(dpct::get_in_order_queue().get_device(),
                                     {sycl::aspect::fp16});

        dpct::get_in_order_queue().submit([&](sycl::handler &cgh)
                                          {
            sycl::local_accessor<sycl::half, 1> dpct_local_acc_ct1(
                sycl::range<1>(shmem_size_block_gemm), cgh);

            const sycl::half ct3 = 1.0;
            const sycl::half ct4 = 0.0;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, BLOCKS_PER_GRID) *
                                      sycl::range<3>(1, 1, THREADS_PER_BLOCK),
                                  sycl::range<3>(1, 1, THREADS_PER_BLOCK)),
                [=](sycl::nd_item<3> item_ct1) 
                [[sycl::reqd_sub_group_size(WARP_SIZE)]]
                {
                    block_gemm_3d_half_sycl(
                        d_A, d_B, d_C, ct3, ct4, item_ct1,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get(),
                            d_warpsize
                        );
                }); });
    }
    dpct::sync_barrier(stop);

    dpct::get_current_device().queues_wait_and_throw();

    dpct::err0 err = 0;

    if (err != 0)
    {
        std::cerr << "CUDA Error: " << std::endl;
        return -1;
    }

    double milliseconds =
        (stop->get_profiling_info<sycl::info::event_profiling::command_end>() -
         start->get_profiling_info<
             sycl::info::event_profiling::command_start>()) /
        1000000.0f;

    long long numOpsPerMatrix = 2LL * M_BLOCK * N_BLOCK * K_BLOCK;
    long long totalOps = numOpsPerMatrix * BLOCKS_PER_GRID * NUM_ITER;
    double gflops = static_cast<double>(totalOps) / (milliseconds * 1e6);
    double tflops = gflops / 1000.0f;

    std::cout << "Execution time: " << milliseconds << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS (" << tflops << " TFLOPS)" << std::endl;

    std::cout << "[hemeng_log],3d," << M_BLOCK << "," << N_BLOCK << "," << K_BLOCK << "," << NUM_RANK_BLOCK << "," << milliseconds << "," << tflops << "," << THREADS_PER_BLOCK << std::endl;

    float *h_C =
        (float *)malloc(M_BLOCK * N_BLOCK * NUM_RANK_BLOCK * sizeof(float));
    dpct::get_in_order_queue()
        .memcpy(h_C, d_C, M_BLOCK * N_BLOCK * NUM_RANK_BLOCK * sizeof(float))
        .wait();

    int warpSize = 0;
    dpct::get_in_order_queue().memcpy(&warpSize, d_warpsize, sizeof(int)).wait();
    std::cout << "warpSize: " << warpSize << std::endl;

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

    double *h_C_ref =
        (double *)malloc(M_BLOCK * N_BLOCK * sizeof(double));

    for (int i = 0; i < M_BLOCK; i++)
    {
        for (int j = 0; j < M_BLOCK; j++)
        {
            h_C_ref[i * N_BLOCK + j] = 0;
            for (int k = 0; k < K_BLOCK; k++)
            {
                h_C_ref[i * N_BLOCK + j] = (double)h_C_ref[i * N_BLOCK + j] + (double)h_A[i * K_BLOCK + k] * (double)h_B[k * K_BLOCK + j];
            }
        }
    }

    int error = 0;
    for (int i = 0; i < M_BLOCK * N_BLOCK; i++)
    {
        if (fabs((double)h_C[i] / NUM_ITER - (double)h_C_ref[i]) > 1e-6)
        {
            std::cout << "Error: " << i << " " << (double)h_C[i] / NUM_ITER << " " << (double)h_C_ref[i] << std::endl;
            error = 1;
            break;
        }
    }

    if (!error)
    {
        std::cout << "Validation successful!" << std::endl;
    }

    dpct::destroy_event(start);
    dpct::destroy_event(stop);

    return 0;
}