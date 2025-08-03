// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_TRAITS_DETAIL_DESCRIPTION_TRAITS_HPP
#define CUBLASDX_TRAITS_DETAIL_DESCRIPTION_TRAITS_HPP

#include "commondx/detail/stl/type_traits.hpp"

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/description_traits.hpp"

namespace cublasdx {
    namespace detail {
        template<operator_type OperatorType, class Description>
        using get_t = commondx::detail::get_t<operator_type, OperatorType, Description>;
        template<operator_type OperatorType, class Description, class Default = void>
        using get_or_default_t = commondx::detail::get_or_default_t<operator_type, OperatorType, Description, Default>;
        template<operator_type OperatorType, class Description>
        using has_operator = commondx::detail::has_operator<operator_type, OperatorType, Description>;
        template<operator_type OperatorType, class Description>
        using has_at_most_one_of = commondx::detail::has_at_most_one_of<operator_type, OperatorType, Description>;

        // is_complete_description

        namespace __is_complete_description {
            template<class Function, class Diagonal, class Side>
            struct complete_trsm_helper : COMMONDX_STL_NAMESPACE::true_type {};

            template<class Diagonal>
            struct complete_trsm_helper<Function<function::TRSM>, Diagonal, void> : COMMONDX_STL_NAMESPACE::false_type {};

            template<class Side>
            struct complete_trsm_helper<Function<function::TRSM>, void, Side> : COMMONDX_STL_NAMESPACE::false_type {};

            template<class Description, class Enable = void>
            struct helper: COMMONDX_STL_NAMESPACE::false_type {};

            template<template<class...> class Description, class... Types>
            struct helper<Description<Types...>, typename COMMONDX_STL_NAMESPACE::enable_if<commondx::detail::is_description_expression<Description<Types...>>::value>::type> {
                using description_type = Description<Types...>;

                // SM
                using this_blas_sm = get_t<operator_type::sm, description_type>;
                // Size
                using this_blas_size = get_t<operator_type::size, description_type>;
                // Precision
                using this_blas_precision = get_or_default_t<operator_type::precision, description_type, default_blas_precision_operator>;
                // Type
                using this_blas_type = get_or_default_t<operator_type::type, description_type, default_blas_type_operator>;
                // Function
                using this_blas_function = get_t<operator_type::function, description_type>;

                static constexpr bool value =
                    !(COMMONDX_STL_NAMESPACE::is_void<this_blas_size>::value || COMMONDX_STL_NAMESPACE::is_void<this_blas_precision>::value ||
                      COMMONDX_STL_NAMESPACE::is_void<this_blas_type>::value || COMMONDX_STL_NAMESPACE::is_void<this_blas_function>::value ||
                      COMMONDX_STL_NAMESPACE::is_void<this_blas_sm>::value);
            };
        } // namespace __is_complete_description

        template<class Description>
        struct is_complete_description: COMMONDX_STL_NAMESPACE::integral_constant<bool, __is_complete_description::helper<Description>::value> {};
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_TRAITS_DETAIL_DESCRIPTION_TRAITS_HPP
