// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_TRANSPOSE_MODE_HPP
#define CUBLASDX_OPERATORS_TRANSPOSE_MODE_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cublasdx {
    enum class transpose_mode
    {
        non_transposed,
        transposed,
        conj_transposed,
    };

    inline constexpr auto N = transpose_mode::non_transposed;
    inline constexpr auto T = transpose_mode::transposed;
    inline constexpr auto C = transpose_mode::conj_transposed;

    template<transpose_mode A, transpose_mode B = N>
    struct [[deprecated]] TransposeMode: commondx::detail::operator_expression {
        static constexpr transpose_mode a_transpose_mode = A;
        static constexpr transpose_mode b_transpose_mode = B;
    };

    template<transpose_mode A, transpose_mode B>
    constexpr transpose_mode TransposeMode<A, B>::a_transpose_mode;

    template<transpose_mode A, transpose_mode B>
    constexpr transpose_mode TransposeMode<A, B>::b_transpose_mode;

    namespace detail {
        using default_blas_transpose_mode_operator = TransposeMode<T, N>;
    } // namespace detail
} // namespace cublasdx


namespace commondx::detail {
    template<cublasdx::transpose_mode A, cublasdx::transpose_mode B>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::transpose_mode, cublasdx::TransposeMode<A, B>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<cublasdx::transpose_mode A, cublasdx::transpose_mode B>
    struct get_operator_type<cublasdx::TransposeMode<A, B>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::transpose_mode;
    };
} // namespace commondx::detail

#endif // CUBLASDX_OPERATORS_TRANSPOSE_MODE_HPP
