// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_OPERATORS_REAL_FFT_OPTIONS_HPP
#define CUFFTDX_OPERATORS_REAL_FFT_OPTIONS_HPP

#include "fft_operator.hpp"
#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cufftdx {
    enum class complex_layout
    {
        natural,
        packed,
        full
    };

    enum class real_mode
    {
        normal,
        folded,
        batch_optimized
    };

    template<complex_layout RealLayout, real_mode RealMode = real_mode::normal>
    struct RealFFTOptions: public commondx::detail::operator_expression {
        static_assert(RealMode != real_mode::batch_optimized, "Batch optimization is not implemented yet!");

        static constexpr complex_layout layout = RealLayout;
        static constexpr real_mode   mode   = RealMode;
    };

    template<complex_layout RealLayout, real_mode RealMode>
    constexpr complex_layout RealFFTOptions<RealLayout, RealMode>::layout;

    template<complex_layout RealLayout, real_mode RealMode>
    constexpr real_mode RealFFTOptions<RealLayout, RealMode>::mode;
} // namespace cufftdx

namespace commondx::detail {
    template<cufftdx::complex_layout RealLayout, cufftdx::real_mode RealMode>
    struct is_operator<cufftdx::fft_operator, cufftdx::fft_operator::real_fft_options, cufftdx::RealFFTOptions<RealLayout, RealMode>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<cufftdx::complex_layout RealLayout, cufftdx::real_mode RealMode>
    struct get_operator_type<cufftdx::RealFFTOptions<RealLayout, RealMode>> {
        static constexpr cufftdx::fft_operator value = cufftdx::fft_operator::real_fft_options;
    };
} // namespace commondx::detail

#endif // CUFFTDX_OPERATORS_HERMITIAN_OPTIONS_HPP
