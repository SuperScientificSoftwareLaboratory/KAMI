// Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_OPERATORS_TYPE_HPP
#define CUFFTDX_OPERATORS_TYPE_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cufftdx {
    enum class fft_type
    {
        c2c,
        c2r,
        r2c
    };

    template<fft_type Value>
    struct Type: public commondx::detail::constant_operator_expression<fft_type, Value> {};
} // namespace cufftdx

namespace commondx::detail {
    template<cufftdx::fft_type Value>
    struct is_operator<cufftdx::fft_operator, cufftdx::fft_operator::type, cufftdx::Type<Value>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<cufftdx::fft_type Value>
    struct get_operator_type<cufftdx::Type<Value>> {
        static constexpr cufftdx::fft_operator value = cufftdx::fft_operator::type;
    };
} // namespace commondx::detail

#endif // CUFFTDX_OPERATORS_TYPE_HPP
