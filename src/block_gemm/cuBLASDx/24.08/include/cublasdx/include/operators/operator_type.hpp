// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_OPERATOR_TYPE_HPP
#define CUBLASDX_OPERATORS_OPERATOR_TYPE_HPP

#include "commondx/detail/expressions.hpp"

namespace cublasdx {
    enum class operator_type
    {
        size,
        precision,
        type,
        function,
        fill_mode,
        transpose_mode,
        diagonal,
        side,
        sm,
        alignment,
        arrangement,
        ld,
        // execution
        block,
        thread,
        // block only
        block_dim
    };
} // namespace cublasdx

#endif // CUBLASDX_OPERATORS_OPERATOR_TYPE_HPP
