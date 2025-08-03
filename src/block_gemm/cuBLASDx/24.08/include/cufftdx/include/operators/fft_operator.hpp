// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_OPERATORS_OPERATOR_TYPE_HPP
#define CUFFTDX_OPERATORS_OPERATOR_TYPE_HPP

#include "commondx/detail/expressions.hpp"

namespace cufftdx {
    enum class fft_operator
    {
        direction,
        precision,
        size,
        sm,
        type,
        // options
        real_fft_options,
        // execution
        thread,
        block,
        // block-only
        elements_per_thread,
        ffts_per_block,
        block_dim,
    };
} // namespace cufftdx

#endif // CUFFTDX_OPERATORS_OPERATOR_TYPE_HPP
