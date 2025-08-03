// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef COMMONDX_DETAIL_EXPRESSIONS_HPP
#define COMMONDX_DETAIL_EXPRESSIONS_HPP

#include "../detail/stl/type_traits.hpp"

// Namespace wrapper
#include "../detail/namespace_wrapper_open.hpp"

namespace commondx {
    namespace detail {
        struct expression {};
        struct operator_expression: expression {};
        struct block_operator_expression: operator_expression {};

        struct description_expression: expression {};
        struct execution_description_expression: description_expression {};

        template<class ValueType, ValueType Value>
        struct constant_operator_expression:
            public operator_expression,
            public COMMONDX_STL_NAMESPACE::integral_constant<ValueType, Value> {};

        template<class ValueType, ValueType Value>
        struct constant_block_operator_expression:
            public block_operator_expression,
            public COMMONDX_STL_NAMESPACE::integral_constant<ValueType, Value> {};
    } // namespace detail
} // namespace commondx

// Namespace wrapper
#include "../detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_DETAIL_EXPRESSIONS_HPP
