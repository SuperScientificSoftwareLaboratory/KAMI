// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef COMMONDX_OPERATORS_BLOCK_DIM_HPP
#define COMMONDX_OPERATORS_BLOCK_DIM_HPP

#include "../detail/expressions.hpp"

// Namespace wrapper
#include "../detail/namespace_wrapper_open.hpp"

namespace commondx {
    template<unsigned int X, unsigned int Y = 1, unsigned int Z = 1>
    struct BlockDim: public detail::block_operator_expression  {
        static constexpr unsigned int x     = X;
        static constexpr unsigned int y     = Y;
        static constexpr unsigned int z     = Z;
        static constexpr dim3         value = dim3(x, y, z);

        static constexpr unsigned int flat_size = x * y * z;
        static constexpr unsigned int rank      = (z != 1) ? 3 : ((y != 1) ? 2 : 1);
    };

    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr unsigned int BlockDim<X, Y, Z>::x;
    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr unsigned int BlockDim<X, Y, Z>::y;
    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr unsigned int BlockDim<X, Y, Z>::z;

    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr dim3 BlockDim<X, Y, Z>::value;

    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr unsigned int BlockDim<X, Y, Z>::flat_size;

    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr unsigned int BlockDim<X, Y, Z>::rank;
} // namespace commondx

// Namespace wrapper
#include "../detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_OPERATORS_BLOCK_DIM_HPP
