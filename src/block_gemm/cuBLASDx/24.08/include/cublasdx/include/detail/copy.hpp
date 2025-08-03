// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_COPY_HPP
#define CUBLASDX_DETAIL_COPY_HPP

#include "../database/cute.hpp"

namespace cublasdx {
    __forceinline__ __device__
    void copy_wait() {
        cute::cp_async_fence();
        cute::cp_async_wait<0>();
        __syncthreads();
    }

    template<uint32_t NumThreads,
             uint32_t AlignmentInBytes,
             class SrcEngine,
             class SrcLayout,
             class DstEngine,
             class DstLayout>
    __forceinline__ __device__
    void copy(const unsigned int                            tid,
              const cublasdx::tensor<SrcEngine, SrcLayout>& src,
              cublasdx::tensor<DstEngine, DstLayout>&       dst) {
        using src_shape = decltype(src.shape());
        using dst_shape = decltype(dst.shape());
        static_assert(cute::is_static_v<src_shape> and cute::is_static_v<dst_shape>,
            "cublasdx::copy requires static tensor layouts");
        if(tid < NumThreads) {
            // Below WAR for a case where:
            // Partitioning of the data among threads is unaligned with the possible swizzle.
            // Can be loosened up in specific cases but not in a general swizzle case
            if constexpr(cute::is_constant<48, decltype(cute::size<0>(SrcLayout{}))>::value and
                         cute::is_constant<48, decltype(cute::size<1>(SrcLayout{}))>::value) {
                return cute::naive_cooperative_copy<NumThreads>(tid, src, dst);
            } else {
                return cute::cooperative_copy<NumThreads, 8 * AlignmentInBytes>(tid, src, dst);
            }
        }
    }

    template<uint32_t NumThreads,
             class SrcEngine,
             class SrcLayout,
             class DstEngine,
             class DstLayout>
    __forceinline__ __device__
    void copy(const unsigned int                            tid,
              const cublasdx::tensor<SrcEngine, SrcLayout>& src,
              cublasdx::tensor<DstEngine, DstLayout>&       dst) {
        using src_shape = decltype(src.shape());
        using dst_shape = decltype(dst.shape());
        static_assert(cute::is_static_v<src_shape> and cute::is_static_v<dst_shape>,
            "cublasdx::copy requires static tensor layouts");
        // This takes only max_vec_bits of source tensor under consideration, but
        // if the datatypes of src and dst are different, the vectorization is
        // turned off and max_vec_bits value does not matter
        constexpr unsigned int max_vec_bits = cute::sizeof_bits_v<typename SrcEngine::value_type>;
        if(tid < NumThreads) {
            if constexpr(cute::is_constant<48, decltype(cute::size<0>(SrcLayout{}))>::value and
                         cute::is_constant<48, decltype(cute::size<1>(SrcLayout{}))>::value) {
                return cute::naive_cooperative_copy<NumThreads>(tid, src, dst);
            } else {
                return cute::cooperative_copy<NumThreads, max_vec_bits>(tid, src, dst);
            }
        }
    }

    // This overload uses as many threads as defined in BLAS::block_dim
    template<class BLAS, uint32_t AlignmentInBytes, class SrcEngine, class SrcLayout, class DstEngine, class DstLayout>
    __forceinline__ __device__
    void copy(const cublasdx::tensor<SrcEngine, SrcLayout>& src,
              cublasdx::tensor<DstEngine, DstLayout>&       dst) {
        using src_shape = decltype(src.shape());
        using dst_shape = decltype(dst.shape());
        static_assert(cute::is_static_v<src_shape> and cute::is_static_v<dst_shape>,
            "cublasdx::copy requires static tensor layouts");
        constexpr unsigned int num_threads = BLAS::block_dim.x * BLAS::block_dim.y * BLAS::block_dim.z;
        constexpr unsigned int block_dim_rank = (BLAS::block_dim.z > 1) ? 3 : ((BLAS::block_dim.y > 1) ? 2 : 1);
        unsigned int tid = detail::get_thread_idx<block_dim_rank>();
        copy<num_threads, AlignmentInBytes>(tid, src, dst);
    }
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_COPY_HPP
