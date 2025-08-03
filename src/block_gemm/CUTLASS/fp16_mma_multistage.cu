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
    \brief Unit tests for threadblock-level GEMM
*/

#include "mma_multistage_testbed.h"

#ifndef Problem_m
#define Problem_m 32
#define Problem_n 32
#define Problem_k 32
#endif

#ifndef Block_m
#define Block_m 32
#define Block_n 32
#define Block_k 32
#endif

#ifndef Warp_m
#define Warp_m 32
#define Warp_n 32
#define Warp_k 32
#endif

#ifndef Instruction_m
#define Instruction_m 16
#define Instruction_n 8
#define Instruction_k 16
#endif

#ifndef STAGES
#define STAGES 2
#endif

#if defined(CUTLASS_ARCH_MMA_SM80_SUPPORTED)
TEST(gemm_threadblock_congruous, tensor_op)
{
    using ElementA = cutlass::half_t;
    using LayoutA = cutlass::layout::RowMajor;
    using ElementB = cutlass::half_t;
    using LayoutB = cutlass::layout::ColumnMajor;
    using ElementC = cutlass::half_t;
    using LayoutC = cutlass::layout::RowMajor;

    cutlass::gemm::GemmCoord problem_size(Problem_m, Problem_n, Problem_k);

    using ThreadblockShape = cutlass::gemm::GemmShape<Block_m, Block_n, Block_k>;
    using WarpShape = cutlass::gemm::GemmShape<Warp_m,Warp_n,Warp_k>;
    using InstructionShape = cutlass::gemm::GemmShape<Instruction_m, Instruction_n, Instruction_k>;

    cutlass::half_t alpha = cutlass::half_t(1.0f);
    cutlass::half_t beta = cutlass::half_t(0.0f);
    int const Stages = STAGES;
    // Define the MmaCore components
    using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
        ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
        ElementB, LayoutB, ElementC, LayoutC,
        cutlass::arch::OpClassTensorOp, Stages>;    

    dim3 grid(16384, 1);
    dim3 block(32, MmaCore::WarpCount::kM * MmaCore::WarpCount::kN * MmaCore::WarpCount::kK, 1);
    test::gemm::threadblock::Testbed<MmaCore>(problem_size.m(), problem_size.n(),
                                              problem_size.k(), alpha, beta)
        .run(grid, block);
}
#endif
