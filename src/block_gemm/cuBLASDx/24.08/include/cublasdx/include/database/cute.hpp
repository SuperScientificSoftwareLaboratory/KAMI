// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_HPP
#define CUBLASDX_DATABASE_CUTE_HPP

#include "cute_tensor.hpp"
#include "cute_db.hpp"
#include "cute_utils.hpp"
#include "suggested_layouts.hpp"

namespace cublasdx {
    namespace detail {

    template<unsigned int BlockDimRank>
    __device__ __forceinline__ static unsigned int get_thread_idx() {
        if constexpr (BlockDimRank == 3) {
            return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
        } else if constexpr (BlockDimRank == 2) {
            return threadIdx.x + threadIdx.y * blockDim.x;
        } else {
            return threadIdx.x;
        }
    }

        namespace cute_backend {


template<typename TypeA,
         typename TypeB,
         typename TypeC,
         typename Alignment,
         int SizeM,
         int SizeN,
         int SizeK,
         typename Arrangement,
         typename TransposeMode,
         typename SM,
         typename BlockSize,
         bool     HasBlockSize,
         bool     Benchmark = false,
         mma_atom MmaAtom   = mma_atom::universal_fma,
         int      TileX     = 0,
         int      TileY     = 0>
struct execution {
    static constexpr bool benchmark = Benchmark;
    // Necessary only for pointer API
    using blas_transpose_mode = TransposeMode;
    static constexpr auto tr_mode_a = blas_transpose_mode::a_transpose_mode;
    static constexpr auto tr_mode_b = blas_transpose_mode::b_transpose_mode;
    static constexpr auto tr_mode_c = transpose_mode::non_transposed;

    // Necessary only for pointer API
    using blas_arrangement = Arrangement;
    static constexpr auto arr_a = blas_arrangement::a;
    static constexpr auto arr_b = blas_arrangement::b;
    static constexpr auto arr_c = blas_arrangement::c;

    using blas_alignment = Alignment;
    static constexpr auto align_a = blas_alignment::a;
    using default_a_copy_op = cute::AutoVectorizingCopyWithAssumedAlignment<align_a * 8>;
    static constexpr auto align_b = blas_alignment::b;
    using default_b_copy_op = cute::AutoVectorizingCopyWithAssumedAlignment<align_b * 8>;
    static constexpr auto align_c = blas_alignment::c;
    using default_c_copy_op = cute::AutoVectorizingCopyWithAssumedAlignment<align_c * 8>;

    // Necessary for both APIs
    static constexpr unsigned int m = SizeM;
    static constexpr unsigned int n = SizeN;
    static constexpr unsigned int k = SizeK;

    static constexpr int block_size = HasBlockSize ? (BlockSize::value.x * BlockSize::value.y * BlockSize::value.z) : 128;
    using swizzled_config = cublasdx::detail::layout_database::optimal_config<
        block_size, SM::value,
        TypeA, arr_a == arrangement::col_major, align_a,
        TypeB, arr_b == arrangement::col_major, align_b,
        TypeC, arr_c == arrangement::col_major, align_c,
        m, n, k>;

    using swizzled_a_layout = typename swizzled_config::a_layout;
    using swizzled_b_layout = typename swizzled_config::b_layout;
    using swizzled_c_layout = typename swizzled_config::c_layout;

    // This is necessary for a case where swizzled config is available but the user
    // does not utilize suggested layout and db_entry has higher number of threads
    // than 128
    static constexpr bool forced_128 = (not HasBlockSize and swizzled_config::optimal);
    static constexpr int db_threads = (not HasBlockSize and swizzled_config::optimal) ? 128 : block_size;
    using db_config = typename cute_config<TypeA,
                                       TypeB,
                                       TypeC,
                                       m,
                                       n,
                                       k,
                                       db_threads,
                                       HasBlockSize or forced_128,
                                       SM,
                                       Benchmark,
                                       MmaAtom,
                                       TileX,
                                       TileY>::config;

    using mma_t = COMMONDX_STL_NAMESPACE::conditional_t<(not Benchmark) and swizzled_config::optimal, typename swizzled_config::TiledMma, db_config>;
    static constexpr unsigned int suggested_threads = cute::size(mma_t{});
    static constexpr unsigned int threads = cute::size(mma_t{});

    // This operates on assumption (checked in BLAS.execute()) that tensor sizes agree with operator sizes
    template<typename TA,
             typename ALayout,
             typename Alpha,
             typename TB,
             typename BLayout,
             typename Beta,
             typename TC,
             typename CLayout,
             typename ALoadOp = identity,
             typename BLoadOp = identity,
             typename CLoadOp = identity,
             typename CStoreOp = identity>
    __device__ __forceinline__ static void tensor_gemm(cute::Tensor<TA, ALayout> smem_tensor_a,
                                                       cute::Tensor<TB, BLayout> smem_tensor_b,
                                                       cute::Tensor<TC, CLayout> smem_tensor_c,
                                                       const Alpha               alpha,
                                                       const Beta                beta,
                                                       const ALoadOp&            a_load_op  = identity {},
                                                       const BLoadOp&            b_load_op  = identity {},
                                                       const CLoadOp&            c_load_op  = identity {},
                                                       const CStoreOp&           c_store_op = identity {}) {
        const auto thread_idx = cublasdx::detail::get_thread_idx<BlockSize::rank>();

        constexpr bool is_suggested_execution =
            swizzled_config::optimal and
            COMMONDX_STL_NAMESPACE::is_same_v<ALayout, swizzled_a_layout> and
            COMMONDX_STL_NAMESPACE::is_same_v<BLayout, swizzled_b_layout> and
            COMMONDX_STL_NAMESPACE::is_same_v<CLayout, swizzled_c_layout>;

        using tiled_mma = COMMONDX_STL_NAMESPACE::conditional_t<is_suggested_execution, typename swizzled_config::TiledMma, db_config>;
        using a_copy_op = COMMONDX_STL_NAMESPACE::conditional_t<is_suggested_execution, typename swizzled_config::a_copy_op, default_a_copy_op>;
        using b_copy_op = COMMONDX_STL_NAMESPACE::conditional_t<is_suggested_execution, typename swizzled_config::b_copy_op, default_b_copy_op>;
        using c_copy_op = COMMONDX_STL_NAMESPACE::conditional_t<is_suggested_execution, typename swizzled_config::c_copy_op, default_c_copy_op>;

        if (thread_idx < cute::size(tiled_mma{})) {
            cute::cooperative_gemm<a_copy_op, b_copy_op, c_copy_op>(
                                   thread_idx,
                                   tiled_mma {},
                                   alpha,
                                   smem_tensor_a,
                                   swap_tensor_modes(smem_tensor_b),
                                   beta,
                                   smem_tensor_c,
                                   a_load_op,
                                   b_load_op,
                                   c_load_op,
                                   c_store_op);
        }
    }
};

template<typename TypeA,
         typename TypeB,
         typename TypeC,
         typename Alignment,
         int SizeM,
         int SizeN,
         int SizeK,
         typename Arrangement,
         typename TransposeMode,
         typename SM>
struct execution_suggestions {
private:
    using execution_type = cute_backend::execution<TypeA,
                                                   TypeB,
                                                   TypeC,
                                                   Alignment,
                                                   SizeM,
                                                   SizeN,
                                                   SizeK,
                                                   Arrangement,
                                                   TransposeMode,
                                                   SM,
                                                   BlockDim<256> /* dummy */,
                                                   false>;

public:
    using block_dim = BlockDim<execution_type::suggested_threads, 1, 1>;
};

        } // namespace cute_backend
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_CUTE_HPP
