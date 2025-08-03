// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_HPP
#define CUBLASDX_OPERATORS_HPP

#include "operators/operator_type.hpp"

#include "operators/alignment.hpp"
#include "operators/arrangement.hpp"
#include "operators/precision.hpp"
#include "operators/size.hpp"
#include "operators/type.hpp"
#include "operators/function.hpp"
#include "operators/ld.hpp"
#include "operators/transpose_mode.hpp"
// #include "operators/trsm_operators.hpp"
// #include "operators/fill_mode.hpp"

#include "commondx/operators/block_dim.hpp"
#include "commondx/operators/sm.hpp"
#include "commondx/operators/execution_operators.hpp"

namespace cublasdx {
    // Import selected operators from commonDx
    struct Block: public commondx::Block {};
    template<unsigned int X, unsigned int Y = 1, unsigned int Z = 1>
    struct BlockDim: public commondx::BlockDim<X, Y, Z> {};
    template<unsigned int Architecture>
    struct SM: public commondx::SM<Architecture> {};
} // namespace cublasdx

// Register operators
namespace commondx::detail {
    template<>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::block, cublasdx::Block>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<>
    struct get_operator_type<cublasdx::Block> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::block;
    };

    template<unsigned int Architecture>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::sm, cublasdx::SM<Architecture>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<unsigned int Architecture>
    struct get_operator_type<cublasdx::SM<Architecture>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::sm;
    };

    template<unsigned int X, unsigned int Y, unsigned int Z>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::block_dim, cublasdx::BlockDim<X, Y, Z>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<unsigned int X, unsigned int Y, unsigned int Z>
    struct get_operator_type<cublasdx::BlockDim<X, Y, Z>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::block_dim;
    };
} // namespace commondx::detail

#endif // CUBLASDX_OPERATORS_HPP
