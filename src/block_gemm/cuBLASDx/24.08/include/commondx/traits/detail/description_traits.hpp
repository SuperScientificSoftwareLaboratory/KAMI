// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef COMMONDX_TRAITS_DETAIL_DESCRIPTION_TRAITS_HPP
#define COMMONDX_TRAITS_DETAIL_DESCRIPTION_TRAITS_HPP

#include "../../detail/stl/type_traits.hpp"

#include "../../detail/expressions.hpp"

#include "is_operator_fd.hpp"
#include "get.hpp"
#include "get_operator_fd.hpp"

// Namespace wrapper
#include "../../detail/namespace_wrapper_open.hpp"

namespace commondx {
    namespace detail {
        /// is_expression

        template<class T>
        struct is_expression: public COMMONDX_STL_NAMESPACE::is_base_of<expression, T> {};

        template<class T, class U>
        struct are_expressions:
            public COMMONDX_STL_NAMESPACE::integral_constant<bool, is_expression<T>::value && is_expression<U>::value> {};

        /// is_operator_expression

        template<class T>
        struct is_operator_expression: public COMMONDX_STL_NAMESPACE::is_base_of<operator_expression, T> {};

        template<class T, class U>
        struct are_operator_expressions:
            public COMMONDX_STL_NAMESPACE::integral_constant<bool, is_operator_expression<T>::value && is_operator_expression<U>::value> {
        };

        /// is_description_expression

        template<class T>
        struct is_description_expression: public COMMONDX_STL_NAMESPACE::is_base_of<description_expression, T> {};

        /// is_operator definition

        template<class OperatorTypeClass, OperatorTypeClass OperatorType, class T>
        struct is_operator: COMMONDX_STL_NAMESPACE::false_type {};

        /// has_n_of

        namespace __has_n_of {
            template<unsigned int Counter, class OperatorTypeClass, OperatorTypeClass OperatorType, class Head, class... Types>
            struct counter_helper {
                static constexpr unsigned int value = is_operator<OperatorTypeClass, OperatorType, Head>::value
                                                          ? counter_helper<(Counter + 1), OperatorTypeClass, OperatorType, Types...>::value
                                                          : counter_helper<Counter, OperatorTypeClass, OperatorType, Types...>::value;
            };

            template<unsigned int Counter, class OperatorTypeClass, OperatorTypeClass OperatorType, class Head>
            struct counter_helper<Counter, OperatorTypeClass, OperatorType, Head> {
                static constexpr unsigned int value = is_operator<OperatorTypeClass, OperatorType, Head>::value ? Counter + 1 : Counter;
            };

            template<class OperatorTypeClass, OperatorTypeClass OperatorType, class Operator>
            struct counter: COMMONDX_STL_NAMESPACE::integral_constant<unsigned int, is_operator<OperatorTypeClass, OperatorType, Operator>::value> {};

            template<class OperatorTypeClass, OperatorTypeClass OperatorType, template<class...> class Description, class... Types>
            struct counter<OperatorTypeClass, OperatorType, Description<Types...>>:
                COMMONDX_STL_NAMESPACE::integral_constant<unsigned int, counter_helper<0, OperatorTypeClass,OperatorType, Types...>::value> {};
        } // namespace __has_n_of

        template<unsigned int N, class OperatorTypeClass, OperatorTypeClass OperatorType, class Description>
        struct has_n_of: COMMONDX_STL_NAMESPACE::integral_constant<bool, __has_n_of::counter<OperatorTypeClass, OperatorType, Description>::value == N> {};

        template<class OperatorTypeClass, OperatorTypeClass OperatorType, class Description>
        struct has_at_most_one_of:
            COMMONDX_STL_NAMESPACE::integral_constant<bool, (__has_n_of::counter<OperatorTypeClass, OperatorType, Description>::value <= 1)> {};

        /// has_operator

        template<class OperatorTypeClass, OperatorTypeClass OperatorType, class Description>
        struct has_operator:
            COMMONDX_STL_NAMESPACE::integral_constant<bool, (__has_n_of::counter<OperatorTypeClass, OperatorType, Description>::value > 0)> {};
    } // namespace detail
} // namespace commondx

// Namespace wrapper
#include "../../detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_TRAITS_DETAIL_DESCRIPTION_TRAITS_HPP
