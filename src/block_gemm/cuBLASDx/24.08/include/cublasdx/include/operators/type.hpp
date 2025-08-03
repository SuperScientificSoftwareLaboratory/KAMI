// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_TYPE_HPP
#define CUBLASDX_OPERATORS_TYPE_HPP

#include "commondx/operators/type.hpp"

namespace cublasdx {
    using type = commondx::data_type;

    template<type Value>
    using Type = commondx::DataType<Value>;

    namespace detail {
        using default_blas_type_operator = Type<type::real>;
    }
} // namespace cublasdx


namespace commondx::detail {
    template<cublasdx::type Value>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::type, cublasdx::Type<Value>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<cublasdx::type Value>
    struct get_operator_type<cublasdx::Type<Value>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::type;
    };
} // namespace commondx::detail

#endif // CUBLASDX_OPERATORS_TYPE_HPP
