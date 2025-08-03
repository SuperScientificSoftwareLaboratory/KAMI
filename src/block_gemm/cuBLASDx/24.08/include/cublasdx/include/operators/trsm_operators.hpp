// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_TRSM_OPERATORS_HPP
#define CUBLASDX_OPERATORS_TRSM_OPERATORS_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cublasdx {
    enum class diagonal
    {
        non_unit,
        unit
    };

    template<diagonal Value>
    struct Diagonal: public commondx::detail::constant_operator_expression<diagonal, Value> {
    };

    enum class side
    {
        left,
        right
    };

    template<side Value>
    struct Side: public commondx::detail::constant_operator_expression<side, Value> {
    };

} // namespace cublasdx

namespace commondx::detail {
    template<cublasdx::diagonal Value>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::diagonal, cublasdx::Diagonal<Value>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<cublasdx::diagonal Value>
    struct get_operator_type<cublasdx::Diagonal<Value>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::diagonal;
    };

    template<cublasdx::side Value>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::side, cublasdx::Side<Value>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<cublasdx::side Value>
    struct get_operator_type<cublasdx::Side<Value>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::side;
    };
} // namespace commondx::detail

#endif // CUBLASDX_OPERATORS_TRSM_OPERATORS_HPP
