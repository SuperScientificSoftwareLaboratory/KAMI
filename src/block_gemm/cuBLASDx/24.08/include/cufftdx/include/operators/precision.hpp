// Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_OPERATORS_PRECISION_HPP
#define CUFFTDX_OPERATORS_PRECISION_HPP

#include "commondx/operators/precision.hpp"

namespace cufftdx {
    template<class T>
    struct Precision: public commondx::PrecisionBase<T, __half, float, double> {
    };

    namespace detail {
        using default_fft_precision_operator = Precision<float>;
    } // namespace detail
} // namespace cufftdx

namespace commondx::detail {
    template<class T>
    struct is_operator<cufftdx::fft_operator, cufftdx::fft_operator::precision, cufftdx::Precision<T>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<class T>
    struct get_operator_type<cufftdx::Precision<T>> {
        static constexpr cufftdx::fft_operator value = cufftdx::fft_operator::precision;
    };
} // namespace commondx::detail

#endif // CUFFTDX_OPERATORS_TYPE_HPP
