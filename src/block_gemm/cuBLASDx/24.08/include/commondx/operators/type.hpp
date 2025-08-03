// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef COMMONDX_OPERATORS_TYPE_HPP
#define COMMONDX_OPERATORS_TYPE_HPP

#include "../detail/expressions.hpp"

// Namespace wrapper
#include "../detail/namespace_wrapper_open.hpp"

namespace commondx {
    enum class data_type
    {
        real,
        complex
    };

    template<data_type Value>
    struct DataType: public detail::constant_operator_expression<data_type, Value> {};
} // namespace commondx

// Namespace wrapper
#include "../detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_OPERATORS_TYPE_HPP
