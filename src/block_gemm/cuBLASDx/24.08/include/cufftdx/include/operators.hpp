// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_OPERATORS_HPP
#define CUFFTDX_OPERATORS_HPP

#include "operators/fft_operator.hpp"

#include "operators/block_operators.hpp"
#include "operators/direction.hpp"
#include "operators/precision.hpp"
#include "operators/size.hpp"
#include "operators/type.hpp"

#include "operators/real_fft_options.hpp"

#include "commondx/operators/block_dim.hpp"
#include "commondx/operators/sm.hpp"
#include "commondx/operators/execution_operators.hpp"

namespace cufftdx {
    // Import selected operators from commonDx
    struct Thread: public commondx::Thread {
    };
    struct Block: public commondx::Block {
    };
    template<unsigned int X, unsigned int Y = 1, unsigned int Z = 1>
    struct BlockDim: public commondx::BlockDim<X, Y, Z> {
    };
    template<unsigned int Architecture>
    struct SM: public commondx::SM<Architecture> {
    };
} // namespace cufftdx

namespace commondx::detail {
    template<>
    struct is_operator<cufftdx::fft_operator, cufftdx::fft_operator::thread, cufftdx::Thread>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<>
    struct get_operator_type<cufftdx::Thread> {
        static constexpr cufftdx::fft_operator value = cufftdx::fft_operator::thread;
    };

    template<>
    struct is_operator<cufftdx::fft_operator, cufftdx::fft_operator::block, cufftdx::Block>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<>
    struct get_operator_type<cufftdx::Block> {
        static constexpr cufftdx::fft_operator value = cufftdx::fft_operator::block;
    };

    template<unsigned int Architecture>
    struct is_operator<cufftdx::fft_operator, cufftdx::fft_operator::sm, cufftdx::SM<Architecture>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<unsigned int Architecture>
    struct get_operator_type<cufftdx::SM<Architecture>> {
        static constexpr cufftdx::fft_operator value = cufftdx::fft_operator::sm;
    };

    template<unsigned int X, unsigned int Y, unsigned int Z>
    struct is_operator<cufftdx::fft_operator, cufftdx::fft_operator::block_dim, cufftdx::BlockDim<X, Y, Z>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<unsigned int X, unsigned int Y, unsigned int Z>
    struct get_operator_type<cufftdx::BlockDim<X, Y, Z>> {
        static constexpr cufftdx::fft_operator value = cufftdx::fft_operator::block_dim;
    };
} // namespace commondx::detail

#endif // CUFFTDX_OPERATORS_HPP
