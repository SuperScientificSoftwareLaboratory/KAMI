// Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_TRAITS_DETAIL_DESCRIPTION_TRAITS_HPP
#define CUFFTDX_TRAITS_DETAIL_DESCRIPTION_TRAITS_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#else
#    include <type_traits>
#endif

#include "../../operators.hpp"

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/description_traits.hpp"
#include "commondx/traits/dx_traits.hpp"

namespace cufftdx {
    namespace detail {
        template<fft_operator OperatorType, class Description>
        using get = commondx::detail::get<fft_operator, OperatorType, Description>;
        template<fft_operator OperatorType, class Description>
        using get_t = commondx::detail::get_t<fft_operator, OperatorType, Description>;
        template<fft_operator OperatorType, class Description, class Default = void>
        using get_or_default_t = commondx::detail::get_or_default_t<fft_operator, OperatorType, Description, Default>;
        template<fft_operator OperatorType, class Description>
        using has_operator = commondx::detail::has_operator<fft_operator, OperatorType, Description>;
        template<fft_operator OperatorType, class Description>
        using has_at_most_one_of = commondx::detail::has_at_most_one_of<fft_operator, OperatorType, Description>;
        template<unsigned int N, fft_operator OperatorType, class Description>
        using has_n_of = commondx::detail::has_n_of<N, fft_operator, OperatorType, Description>;

        template<class Description>
        using is_expression = commondx::is_dx_expression<Description>;

        template<class T, class U>
        using are_expressions = commondx::detail::are_expressions<T, U>;

        template<class Description>
        using is_execution_expression = commondx::is_dx_execution_expression<fft_operator, Description>;

        /// has_block_operator
        namespace has_block_operator_impl {
            template<unsigned int Counter, class Head, class... Types>
            struct counter_helper {
                static constexpr unsigned int value = CUFFTDX_STD::is_base_of<commondx::detail::block_operator_expression, Head>::value
                                                          ? counter_helper<(Counter + 1), Types...>::value
                                                          : counter_helper<Counter, Types...>::value;
            };

            template<unsigned int Counter, class Head>
            struct counter_helper<Counter, Head> {
                static constexpr unsigned int value =
                    CUFFTDX_STD::is_base_of<commondx::detail::block_operator_expression, Head>::value ? Counter + 1 : Counter;
            };

            template<class Operator>
            struct counter:
                CUFFTDX_STD::integral_constant<unsigned int, CUFFTDX_STD::is_base_of<commondx::detail::block_operator_expression, Operator>::value> {};

            template<template<class...> class Description, class... Types>
            struct counter<Description<Types...>>:
                CUFFTDX_STD::integral_constant<unsigned int, counter_helper<0, Types...>::value> {};
        } // namespace has_block_operator_impl

        template<class Description>
        struct has_any_block_operator:
            CUFFTDX_STD::integral_constant<bool, (has_block_operator_impl::counter<Description>::value > 0)> {};

        /// deduce_direction_type

        template<class T>
        struct deduce_direction_type {
            using type = void;
        };

        template<>
        struct deduce_direction_type<Type<fft_type::c2r>> {
            using type = Direction<fft_direction::inverse>;
        };

        template<>
        struct deduce_direction_type<Type<fft_type::r2c>> {
            using type = Direction<fft_direction::forward>;
        };

        template<class T>
        using deduce_direction_type_t = typename deduce_direction_type<T>::type;

        // is_complete_description

        namespace is_complete_description_impl {
            template<class Description, class Enable = void>
            struct helper: CUFFTDX_STD::false_type {};

            template<template<class...> class Description, class... Types>
            struct helper<Description<Types...>,
                          CUFFTDX_STD::enable_if_t<commondx::detail::is_description_expression<Description<Types...>>::value>> {
                using description_type = Description<Types...>;

                // Extract and/or deduce description types

                // Size
                using this_fft_size = get_t<fft_operator::size, description_type>;
                // Type (C2C, C2R, R2C)
                using default_fft_type = Type<fft_type::c2c>;
                using this_fft_type    = get_or_default_t<fft_operator::type, description_type, default_fft_type>;
                // Direction
                using deduced_fft_direction = deduce_direction_type_t<this_fft_type>;
                using this_fft_direction =
                    get_or_default_t<fft_operator::direction, description_type, deduced_fft_direction>;
                // SM
                using this_fft_sm = get_t<fft_operator::sm, description_type>;
                // Thread FFT
                static constexpr bool is_thread_execution = has_operator<fft_operator::thread, description_type>::value;

                static constexpr bool value =
                    !(CUFFTDX_STD::is_void<this_fft_size>::value || CUFFTDX_STD::is_void<this_fft_type>::value ||
                      CUFFTDX_STD::is_void<this_fft_direction>::value ||
                      // If we not that FFT is a thread FFT, then we don't require SM for completness
                      (CUFFTDX_STD::is_void<this_fft_sm>::value && !is_thread_execution));
            };
        } // namespace is_complete_description_impl

        template<class Description>
        struct is_complete_description:
            CUFFTDX_STD::integral_constant<bool, is_complete_description_impl::helper<Description>::value> {};
    } // namespace detail
} // namespace cufftdx

#endif // CUFFTDX_TRAITS_DETAIL_DESCRIPTION_TRAITS_HPP
