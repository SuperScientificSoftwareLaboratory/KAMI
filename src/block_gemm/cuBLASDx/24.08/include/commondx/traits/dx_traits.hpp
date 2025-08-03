// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef COMMONDX_TRAITS_DX_TRAITS_HPP
#define COMMONDX_TRAITS_DX_TRAITS_HPP

#include "../operators/block_dim.hpp"
#include "../operators/execution_operators.hpp"
#include "../operators/precision.hpp"
#include "../operators/sm.hpp"
#include "../operators/type.hpp"

#include "detail/get.hpp"
#include "detail/description_traits.hpp"

#include "../detail/stl/type_traits.hpp"

// Namespace wrapper
#include "../detail/namespace_wrapper_open.hpp"

namespace commondx {
    template<class OperatorTypeClass, class Description, class Default>
    struct precision_of {
        using type = typename detail::get_or_default_t<OperatorTypeClass, OperatorTypeClass::precision, Description, Default>::type;
    };

    template<class OperatorTypeClass, class Description, class Default = DataType<data_type::real>>
    struct data_type_of {
        using value_type                  = data_type;
        static constexpr value_type value = detail::get_or_default_t<OperatorTypeClass, OperatorTypeClass::type, Description, Default>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class OperatorTypeClass, class Description, class Default>
    constexpr data_type data_type_of<OperatorTypeClass, Description, Default>::value;

    template<class OperatorTypeClass, class Description>
    inline constexpr data_type data_type_of_v = data_type_of<OperatorTypeClass, Description>::value;

    template<class OperatorTypeClass, class Description>
    struct sm_of {
    private:
        static constexpr bool has_sm = detail::has_operator<OperatorTypeClass, OperatorTypeClass::sm, Description>::value;
        static_assert(has_sm, "Description does not have CUDA architecture defined");

    public:
        using value_type                  = unsigned int;
        static constexpr value_type value = detail::get_t<OperatorTypeClass, OperatorTypeClass::sm, Description>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class OperatorTypeClass, class Description>
    constexpr unsigned int sm_of<OperatorTypeClass, Description>::value;

    template<class OperatorTypeClass, class Description>
    inline constexpr unsigned int sm_of_v = sm_of<OperatorTypeClass, Description>::value;

    template<class OperatorTypeClass, class Description>
    struct block_dim_of {
    private:
        static constexpr bool has_block_dim = detail::has_operator<OperatorTypeClass, OperatorTypeClass::block_dim, Description>::value;
        static_assert(has_block_dim, "Description does not have block dimensions");

    public:
        using value_type                  = dim3;
        static constexpr value_type value = detail::get_t<OperatorTypeClass, OperatorTypeClass::block_dim, Description>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class OperatorTypeClass, class Description>
    constexpr dim3 block_dim_of<OperatorTypeClass, Description>::value;

    template<class OperatorTypeClass, class Description>
    inline constexpr dim3 block_dim_of_v = block_dim_of<OperatorTypeClass, Description>::value;

    template<class Description>
    struct is_dx_expression {
        using value_type                  = bool;
        static constexpr value_type value = detail::is_expression<Description>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    constexpr bool is_dx_expression<Description>::value;

    template<class OperatorTypeClass, class Description>
    struct is_dx_execution_expression {
        static constexpr auto block        = detail::has_operator<OperatorTypeClass, OperatorTypeClass::block, Description>::value;
        static constexpr auto is_block_op  = COMMONDX_STL_NAMESPACE::is_same<Description, Block>::value;
        static constexpr auto thread       = detail::has_operator<OperatorTypeClass, OperatorTypeClass::thread, Description>::value;
        static constexpr auto is_thread_op = COMMONDX_STL_NAMESPACE::is_same<Description, Thread>::value;

    public:
        using value_type                  = bool;
        static constexpr value_type value = is_dx_expression<Description>::value && ((block && !is_block_op) || (thread && !is_thread_op));
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class OperatorTypeClass, class Description>
    constexpr bool is_dx_execution_expression<OperatorTypeClass, Description>::value;

    template<class Description, template<class> class IsCompleteDescriptionCheck>
    struct is_complete_dx_expression {
        using value_type                  = bool;
        static constexpr value_type value = is_dx_expression<Description>::value && IsCompleteDescriptionCheck<Description>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description, template<class> class IsCompleteDescriptionCheck>
    constexpr bool is_complete_dx_expression<Description, IsCompleteDescriptionCheck>::value;

    template<class OperatorTypeClass, class Description, template<class> class IsCompleteDescriptionCheck>
    struct is_complete_dx_execution_expression {
        using value_type = bool;
        static constexpr value_type value =
            is_dx_execution_expression<OperatorTypeClass, Description>::value && is_complete_dx_expression<Description, IsCompleteDescriptionCheck>::value;
        constexpr operator value_type() const noexcept { return value; }
    };

    template<class OperatorTypeClass, class Description, template<class> class IsCompleteDescriptionCheck>
    constexpr bool is_complete_dx_execution_expression<OperatorTypeClass, Description, IsCompleteDescriptionCheck>::value;

    namespace detail {
        // Concatenates OperatorType to the description (faster than using decltype and adding operators)
        template<class OperatorType, class Description>
        struct concatenate_description;

        template<class OperatorType, template<class...> class Description, class... Operators>
        struct concatenate_description<OperatorType, Description<Operators...>> {
            using type = Description<OperatorType, Operators...>;
        };

        template<class OperatorType, class Description>
        using concatenate_description_t = typename concatenate_description<OperatorType, Description>::type;

        // Removes given OperatorType from a Description
        template<class Description, class OperatorTypeClass, OperatorTypeClass OperatorType>
        struct filter {
            using type = void;
        };

        template<class Description, class OperatorTypeClass, OperatorTypeClass OperatorType>
        using filter_t = typename filter<Description, OperatorTypeClass, OperatorType>::type;

        template<template<class...> class Description, class OperatorTypeClass, OperatorTypeClass OperatorType>
        struct filter<Description<>, OperatorTypeClass, OperatorType> {
            using type = Description<>;
        };

        template<template<class...> class Description, class OperatorTypeClass, OperatorTypeClass OperatorType, class Head, class... Tail>
        struct filter<Description<Head, Tail...>,  OperatorTypeClass, OperatorType> {
            using type =
                typename COMMONDX_STL_NAMESPACE::conditional<is_operator<OperatorTypeClass, OperatorType, Head>::value,
                                          filter_t<Description<Tail...>, OperatorTypeClass, OperatorType>,
                                          concatenate_description_t<Head, typename filter<Description<Tail...>, OperatorTypeClass, OperatorType>::type> //
                                          >::type;
        };

        template<template<class...> class DescriptionType, class Description>
        struct convert_to_dx_description {
            using type = void;
        };

        template<template<class...> class DescriptionType, template<class...> class Description, class... Types>
        struct convert_to_dx_description<DescriptionType, Description<Types...>> {
            using type = DescriptionType<Types...>;
        };
    } // namespace detail

    template<template<class...> class DescriptionType, class Description, class OperatorTypeClass>
    struct extract_dx_description {
    private:
        // Converts execution description to simple description, filter_t will remove Thread and Block operators
        using dx_description_type = typename detail::convert_to_dx_description<DescriptionType, Description>::type;

    public:
        static_assert(is_dx_expression<Description>::value, "Description type is not a Dx description");
        using type = typename COMMONDX_STL_NAMESPACE::conditional<
            detail::is_operator_expression<Description>::value,
            Description, // For single operator or if Description just return Description
            detail::filter_t<detail::filter_t<dx_description_type, OperatorTypeClass, OperatorTypeClass::block>, OperatorTypeClass, OperatorTypeClass::thread> //
            >::type;
    };

    template<template<class...> class DescriptionType, class Description, class OperatorTypeClass>
    using extract_dx_description_t = typename extract_dx_description<DescriptionType, Description, OperatorTypeClass>::type;
} // namespace commondx

// Namespace wrapper
#include "../detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_TRAITS_DX_TRAITS_HPP
