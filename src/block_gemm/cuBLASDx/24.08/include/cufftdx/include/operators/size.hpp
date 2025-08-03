// Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_OPERATORS_SIZE_HPP
#define CUFFTDX_OPERATORS_SIZE_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cufftdx {
    template<unsigned int Value>
    struct Size: public commondx::detail::constant_operator_expression<unsigned int, Value> {
        static_assert(Value > 1, "FFT size must be greater than 1");
    };
} // namespace cufftdx

namespace commondx::detail {
    template<unsigned int Value>
    struct is_operator<cufftdx::fft_operator, cufftdx::fft_operator::size, cufftdx::Size<Value>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<unsigned int Value>
    struct get_operator_type<cufftdx::Size<Value>> {
        static constexpr cufftdx::fft_operator value = cufftdx::fft_operator::size;
    };
} // namespace commondx::detail

#endif // CUFFTDX_OPERATORS_SIZE_HPP
