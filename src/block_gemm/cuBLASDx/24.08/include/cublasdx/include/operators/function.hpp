// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_FUNCTION_HPP
#define CUBLASDX_OPERATORS_FUNCTION_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cublasdx {
    enum class function
    {
        MM,
        TRSM
    };

    template<function Value>
    struct Function: public commondx::detail::constant_operator_expression<function, Value> {
        static_assert(Value != function::TRSM, "TRSM is not supported yet");
    };
} // namespace cublasdx

namespace commondx::detail {
    template<cublasdx::function Value>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::function, cublasdx::Function<Value>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<cublasdx::function Value>
    struct get_operator_type<cublasdx::Function<Value>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::function;
    };
} // namespace commondx::detail

#endif // CUBLASDX_OPERATORS_FUNCTION_HPP
