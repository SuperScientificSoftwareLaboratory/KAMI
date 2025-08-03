/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Unit testbed for kernel-level GEMM
*/

#pragma once

#include "cutlass_unit_test.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/array.h"
#include "cutlass/core_io.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

namespace test
{
  namespace gemm
  {
    namespace threadblock
    {

      template <typename Mma>
      __global__ void kernel_multistage_mma(cutlass::gemm::GemmCoord problem_size,
                                            typename Mma::IteratorA::Params params_A,
                                            typename Mma::IteratorA::TensorRef ref_A,
                                            typename Mma::IteratorB::Params params_B,
                                            typename Mma::IteratorB::TensorRef ref_B,
                                            typename Mma::ElementC *ptr_C,
                                            typename Mma::LayoutC::Stride::Index ldc)
      {

        extern __shared__ int GemmSharedStorageBase[];

        typename Mma::SharedStorage *shared_storage =
            reinterpret_cast<typename Mma::SharedStorage *>(GemmSharedStorageBase);

        cutlass::gemm::GemmCoord tb_tile_offset = {0, 0,
                                                   0};
        cutlass::MatrixCoord tb_offset_A{tb_tile_offset.m() * Mma::Shape::kM,
                                         tb_tile_offset.k()};

        cutlass::MatrixCoord tb_offset_B{tb_tile_offset.k(),
                                         tb_tile_offset.n() * Mma::Shape::kN};

        int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;

        typename Mma::IteratorA iterator_A(params_A, ref_A.data(),
                                           {problem_size.m(), problem_size.k()},
                                           tb_thread_id, tb_offset_A);

        typename Mma::IteratorB iterator_B(params_B, ref_B.data(),
                                           {problem_size.k(), problem_size.n()},
                                           tb_thread_id, tb_offset_B);

        int warp_id = __shfl_sync(0xffffffff, threadIdx.y, 0);

        typename Mma::FragmentC accum;

        for (int i = 0; i < 1000; i++)
        {
          accum.clear();

          Mma mma(*shared_storage, tb_thread_id, warp_id, threadIdx.x);

          int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

          mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);
        }

        typename Mma::Operator::IteratorC iterator_C({ptr_C, ldc}, threadIdx.x);

        iterator_C.add_tile_offset(
            {(tb_tile_offset.m() * Mma::WarpCount::kM) +
                 (warp_id % Mma::WarpCount::kM),
             (tb_tile_offset.n() * Mma::WarpCount::kN) +
                 (warp_id / Mma::WarpCount::kM)});
        iterator_C.store(accum);
      }

      template <

          typename MmaCore_>
      struct Testbed
      {

        using MmaCore = MmaCore_;
        using ThreadblockShape = typename MmaCore::Shape;
        using WarpShape = typename MmaCore::WarpShape;
        using InstructionShape = typename MmaCore::InstructionShape;
        using ElementA = typename MmaCore::ElementA;
        using LayoutA = typename MmaCore::LayoutA;
        using ElementB = typename MmaCore::ElementB;
        using LayoutB = typename MmaCore::LayoutB;
        using ElementC = typename MmaCore::ElementC;
        using LayoutC = typename MmaCore::LayoutC;
        using ThreadMapA = typename MmaCore::IteratorThreadMapA;
        using ThreadMapB = typename MmaCore::IteratorThreadMapB;
        using AccessTypeA = cutlass::Array<ElementA, ThreadMapA::kElementsPerAccess>;
        using AccessTypeB = cutlass::Array<ElementB, ThreadMapB::kElementsPerAccess>;
        static int const Stages = MmaCore::kStages;
        static cutlass::arch::CacheOperation::Kind const CacheOpA =
            MmaCore::kCacheOpA;
        static cutlass::arch::CacheOperation::Kind const CacheOpB =
            MmaCore::kCacheOpB;

        using IteratorA =
            cutlass::transform::threadblock::PredicatedTileAccessIterator<
                cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
                ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

        using IteratorB =
            cutlass::transform::threadblock::PredicatedTileAccessIterator<
                cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
                ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;

        using Mma = cutlass::gemm::threadblock::MmaMultistage<
            typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
            CacheOpA, IteratorB, typename MmaCore::SmemIteratorB, CacheOpB, ElementC,
            LayoutC, typename MmaCore::MmaPolicy, Stages>;

        cutlass::HostTensor<ElementA, LayoutA> matrix_A;
        cutlass::HostTensor<ElementB, LayoutB> matrix_B;
        cutlass::HostTensor<ElementC, LayoutC> matrix_C_computed;
        cutlass::HostTensor<ElementC, LayoutC> matrix_C_reference;

        cutlass::gemm::GemmCoord problem_size;
        float alpha, beta;

        Testbed(int m, int n, int k, float alpha_ = float(1), float beta_ = float(0))
            : problem_size(m, n, k), alpha(alpha_), beta(beta_)
        {
          matrix_A.reset(cutlass::make_Coord(m, k));
          matrix_B.reset(cutlass::make_Coord(k, n));
          matrix_C_computed.reset(cutlass::make_Coord(m, n));
          matrix_C_reference.reset(cutlass::make_Coord(m, n), false);
        }

        bool sufficient() const
        {

          cudaDeviceProp properties;
          int device_idx;
          cudaError_t result = cudaGetDevice(&device_idx);

          if (result != cudaSuccess)
          {
            throw std::runtime_error("cudaGetDevice() API call failed.");
          }

          result = cudaGetDeviceProperties(&properties, device_idx);

          if (result != cudaSuccess)
          {
            throw std::runtime_error("cudaGetDeviceProperties() failed");
          }

          return true;
        }

        bool run(
            dim3 grid, dim3 block,
            cutlass::Distribution::Kind init_A = cutlass::Distribution::Uniform,
            cutlass::Distribution::Kind init_B = cutlass::Distribution::Uniform)
        {
          std::cout << "===== GEMM Configuration =====" << std::endl;
          std::cout << "Problem Size: M = " << problem_size.m()
                    << ", N = " << problem_size.n()
                    << ", K = " << problem_size.k() << std::endl;
          std::cout << "ThreadblockShape: "
                    << ThreadblockShape::kM << " x "
                    << ThreadblockShape::kN << " x "
                    << ThreadblockShape::kK << std::endl;
          std::cout << "WarpShape:        "
                    << WarpShape::kM << " x "
                    << WarpShape::kN << " x "
                    << WarpShape::kK << std::endl;
          std::cout << "InstructionShape: "
                    << InstructionShape::kM << " x "
                    << InstructionShape::kN << " x "
                    << InstructionShape::kK << std::endl;
          std::cout << "==============================" << std::endl;
          if (!sufficient())
          {
            return true;
          }

          if (init_A == cutlass::Distribution::Uniform)
          {
            cutlass::reference::host::TensorFill(matrix_A.host_view(), ElementA(1));
          }
          else if (init_A == cutlass::Distribution::Sequential)
          {
            cutlass::reference::host::BlockFillSequential(matrix_A.host_data(),
                                                          matrix_A.capacity());
          }
          else if (init_A == cutlass::Distribution::Identity)
          {
            cutlass::reference::host::TensorFillIdentity(matrix_A.host_view());
          }
          else
          {
            return false;
          }

          if (init_B == cutlass::Distribution::Uniform)
          {
            cutlass::reference::host::TensorFill(matrix_B.host_view(), ElementB(1));
          }
          else if (init_B == cutlass::Distribution::Sequential)
          {
            cutlass::reference::host::BlockFillSequential(matrix_B.host_data(),
                                                          matrix_B.capacity());
          }
          else if (init_B == cutlass::Distribution::Identity)
          {
            cutlass::reference::host::TensorFillIdentity(matrix_B.host_view());
          }
          else
          {
            return false;
          }
          cutlass::reference::host::TensorFill(matrix_C_computed.host_view());

          cutlass::reference::host::TensorFill(matrix_C_reference.host_view());

          matrix_A.sync_device();
          matrix_B.sync_device();
          matrix_C_computed.sync_device();

          typename IteratorA::Params params_A(matrix_A.layout());
          typename IteratorB::Params params_B(matrix_B.layout());

          cudaError_t result;

          int smem_size = int(sizeof(typename Mma::SharedStorage));
          if (smem_size >= (48 << 10))
          {
            result = cudaFuncSetAttribute(
                test::gemm::threadblock::kernel_multistage_mma<Mma>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);

            if (result != cudaSuccess)
            {
              if (CUTLASS_TEST_UNIT_ENABLE_WARNINGS)
              {
                std::cerr << "Test waived due to insufficient CUDA device." << std::endl;
              }
              return true;
            }

            result = cudaFuncSetAttribute(
                test::gemm::threadblock::kernel_multistage_mma<Mma>,
                cudaFuncAttributePreferredSharedMemoryCarveout, 100);
            if (result != cudaSuccess)
            {
              if (CUTLASS_TEST_UNIT_ENABLE_WARNINGS)
              {
                std::cerr << "Test waived due to insufficient CUDA device." << std::endl;
              }
              return true;
            }
          }
          cudaEvent_t start, stop;
          cudaEventCreate(&start);
          cudaEventCreate(&stop);

          cudaEventRecord(start);
          test::gemm::threadblock::kernel_multistage_mma<Mma>
              <<<grid, block, smem_size, 0>>>(
                  problem_size, params_A, matrix_A.device_ref(), params_B,
                  matrix_B.device_ref(), matrix_C_computed.device_data(),
                  matrix_C_computed.layout().stride(0));
          cudaDeviceSynchronize();
          cudaEventRecord(stop);

          cudaEventSynchronize(stop);
          float milliseconds = 0;
          cudaEventElapsedTime(&milliseconds, start, stop);
          long long numOpsPerMatrix = 2LL * problem_size.m() * problem_size.n() * problem_size.k();
          long long totalOps = numOpsPerMatrix * grid.x * grid.y * grid.z * 1000;
          double gflops = static_cast<double>(totalOps) / (milliseconds * 1e6);
          double tflops = gflops / 1000.0f;
          printf("[hemeng_log],%d,%d,%d,%d,%d,%f,%f\n", problem_size.m(), problem_size.n(), problem_size.k(), block.x * block.y * block.z, grid.x * grid.y * grid.z, gflops, tflops);

          result = cudaDeviceSynchronize();
          EXPECT_EQ(result, cudaSuccess)
              << " kernel error: " << cudaGetErrorString(result);

          matrix_C_computed.sync_host();

          cutlass::reference::host::Gemm<ElementA, LayoutA, ElementB, LayoutB,
                                         ElementC, LayoutC, ElementC, ElementC>
              reference_gemm;

          reference_gemm(
              problem_size, ElementC(alpha), matrix_A.host_view(),
              matrix_B.host_view(), ElementC(beta), matrix_C_reference.host_view());

          bool passed = cutlass::reference::host::TensorEquals(
              matrix_C_computed.host_view(), matrix_C_reference.host_view());

          EXPECT_TRUE(passed);

          if (!passed)
          {
            std::cout
                << __FILE__ << ":" << __LINE__ << "  "
                << "A:\n"
                << matrix_A.host_view() << "\n"
                << "B:\n"
                << matrix_B.host_view() << "\n"
                << "Reference:\n"
                << matrix_C_reference.host_view() << "\n"
                << "Computed:\n"
                << matrix_C_computed.host_view() << "\n";
          }

          EXPECT_GT(cutlass::reference::host::TensorNorm(matrix_C_reference.host_view()), 0);
          EXPECT_GT(cutlass::reference::host::TensorNorm(matrix_C_computed.host_view()), 0);

          return passed;
        }
      };

    }
  }
}
