// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef COMMONDX_OPERATORS_SIZE_HPP
#define COMMONDX_OPERATORS_SIZE_HPP

#include "../detail/expressions.hpp"

#include "../traits/detail/is_operator_fd.hpp"
#include "../traits/detail/get_operator_fd.hpp"

// Namespace wrapper
#include "../detail/namespace_wrapper_open.hpp"

namespace commondx {
    template<unsigned int X, unsigned int Y = 1, unsigned int Z = 1>
    struct Size3D: public detail::operator_expression {
        static_assert(X > 0, "First dimension must be greater than 0");
        static_assert(Y > 0, "Second dimension size must be greater than 0");
        static_assert(Z > 0, "Third dimension size must be greater than 0");

        static constexpr unsigned int x = X;
        static constexpr unsigned int y = Y;
        static constexpr unsigned int z = Z;

        static constexpr unsigned int flat_size = x * y * z;
        static constexpr unsigned int rank      = (x != 1) + (y != 1) + (z != 1);
    };

    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr unsigned int Size3D<X, Y, Z>::x;

    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr unsigned int Size3D<X, Y, Z>::y;

    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr unsigned int Size3D<X, Y, Z>::z;

    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr unsigned int Size3D<X, Y, Z>::flat_size;

    template<unsigned int X, unsigned int Y, unsigned int Z>
    constexpr unsigned int Size3D<X, Y, Z>::rank;
} // namespace commondx

// Namespace wrapper
#include "../detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_OPERATORS_SIZE_HPP
