// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_FILL_MODE_HPP
#define CUBLASDX_OPERATORS_FILL_MODE_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cublasdx {
    enum class fill_mode
    {
        upper,
        lower,
    };

    template<fill_mode Value>
    struct FillMode: public commondx::detail::constant_operator_expression<fill_mode, Value> {
    };
} // namespace cublasdx

namespace commondx::detail {
    template<cublasdx::fill_mode Value>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::fill_mode, cublasdx::FillMode<Value>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<cublasdx::fill_mode Value>
    struct get_operator_type<cublasdx::FillMode<Value>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::fill_mode;
    };
} // namespace commondx::detail

#endif // CUBLASDX_OPERATORS_FILL_MODE_HPP
