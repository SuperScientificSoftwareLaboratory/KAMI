// Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_OPERATORS_BLOCK_OPERATORS_HPP
#define CUFFTDX_OPERATORS_BLOCK_OPERATORS_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cufftdx {
    template<unsigned int Value>
    struct FFTsPerBlock: public commondx::detail::constant_block_operator_expression<unsigned int, Value> {
    };

    template<unsigned int Value>
    struct ElementsPerThread: public commondx::detail::constant_block_operator_expression<unsigned int, Value> {
    };
} // namespace cufftdx

namespace commondx::detail {
    template<unsigned int Value>
    struct is_operator<cufftdx::fft_operator, cufftdx::fft_operator::ffts_per_block, cufftdx::FFTsPerBlock<Value>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<unsigned int Value>
    struct get_operator_type<cufftdx::FFTsPerBlock<Value>> {
        static constexpr cufftdx::fft_operator value = cufftdx::fft_operator::ffts_per_block;
    };

    template<unsigned int Value>
    struct is_operator<cufftdx::fft_operator, cufftdx::fft_operator::elements_per_thread, cufftdx::ElementsPerThread<Value>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<unsigned int Value>
    struct get_operator_type<cufftdx::ElementsPerThread<Value>> {
        static constexpr cufftdx::fft_operator value = cufftdx::fft_operator::elements_per_thread;
    };
} // namespace commondx::detail

#endif // CUFFTDX_OPERATORS_BLOCK_OPERATORS_HPP
