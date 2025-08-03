// Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_OPERATORS_DIRECTION_HPP
#define CUFFTDX_OPERATORS_DIRECTION_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cufftdx {
    enum class fft_direction
    {
        forward,
        inverse
    };

    template<fft_direction Value>
    struct Direction: public commondx::detail::constant_operator_expression<fft_direction, Value> {};
} // namespace cufftdx

namespace commondx::detail {
    template<cufftdx::fft_direction Value>
    struct is_operator<cufftdx::fft_operator, cufftdx::fft_operator::direction, cufftdx::Direction<Value>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<cufftdx::fft_direction Value>
    struct get_operator_type<cufftdx::Direction<Value>> {
        static constexpr cufftdx::fft_operator value = cufftdx::fft_operator::direction;
    };
} // namespace commondx::detail

#endif // CUFFTDX_OPERATORS_DIRECTION_HPP
