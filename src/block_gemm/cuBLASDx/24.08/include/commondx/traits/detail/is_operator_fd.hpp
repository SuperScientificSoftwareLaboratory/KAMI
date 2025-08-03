// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef COMMONDX_TRAITS_DETAIL_IS_OPERATOR_HPP
#define COMMONDX_TRAITS_DETAIL_IS_OPERATOR_HPP

// Namespace wrapper
#include "../../detail/namespace_wrapper_open.hpp"

namespace commondx {
    namespace detail {
        template<class OperatorTypeClass, OperatorTypeClass OperatorType, class T>
        struct is_operator;
    } // namespace detail
} // namespace commondx

// Namespace wrapper
#include "../../detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_TRAITS_DETAIL_IS_OPERATOR_HPP
