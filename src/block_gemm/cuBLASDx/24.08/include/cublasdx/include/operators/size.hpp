// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_SIZE_HPP
#define CUBLASDX_OPERATORS_SIZE_HPP

#include "commondx/operators/size3d.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cublasdx {
    template<unsigned int M, unsigned int N, unsigned int K = 1>
    struct Size: public commondx::Size3D<M, N, K> {
        static_assert(M > 0, "First dimension must be greater than 0");
        static_assert(N > 0, "Second dimension size must be greater than 0");
        static_assert(K > 0, "Third dimension size must be greater than 0");

        static constexpr unsigned int m = M;
        static constexpr unsigned int n = N;
        static constexpr unsigned int k = K;
    };

    template<unsigned int M, unsigned int N, unsigned int K>
    constexpr unsigned int Size<M, N, K>::m;

    template<unsigned int M, unsigned int N, unsigned int K>
    constexpr unsigned int Size<M, N, K>::n;

    template<unsigned int M, unsigned int N, unsigned int K>
    constexpr unsigned int Size<M, N, K>::k;
} // namespace cublasdx


namespace commondx::detail {
    template<unsigned int M, unsigned int N, unsigned int K>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::size, cublasdx::Size<M, N, K>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<unsigned int M, unsigned int N, unsigned int K>
    struct get_operator_type<cublasdx::Size<M, N, K>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::size;
    };
} // namespace commondx::detail

#endif // CUBLASDX_OPERATORS_SIZE_HPP
