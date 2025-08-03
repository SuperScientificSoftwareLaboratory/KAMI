// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef COMMONDX_TRAITS_DETAIL_GET_HPP
#define COMMONDX_TRAITS_DETAIL_GET_HPP

#include "../../detail/stl/type_traits.hpp"

#include "is_operator_fd.hpp"

// Namespace wrapper
#include "../../detail/namespace_wrapper_open.hpp"

namespace commondx {
    namespace detail {
        namespace __get {
            template<class OperatorTypeClass, OperatorTypeClass OperatorType, class T>
            struct helper {
                using type = typename COMMONDX_STL_NAMESPACE::
                    conditional<is_operator<OperatorTypeClass, OperatorType, T>::value, T, void>::type;
            };

            // clang-format off
            template<class OperatorTypeClass, OperatorTypeClass OperatorType,
                     template<class...> class DescriptionType,
                     class TypeHead,
                     class... TailTypes>
            struct helper<OperatorTypeClass, OperatorType, DescriptionType<TypeHead, TailTypes...>> {
                using type = typename COMMONDX_STL_NAMESPACE::conditional<
                    is_operator<OperatorTypeClass, OperatorType, TypeHead>::value,
                    TypeHead,
                    typename helper<OperatorTypeClass, OperatorType, DescriptionType<TailTypes...>>::type>::type;
            };
            // clang-format on
        } // namespace __get

        /// get

        template<class OperatorTypeClass, OperatorTypeClass OperatorType, class Description>
        struct get {
            using type = typename __get::helper<OperatorTypeClass, OperatorType, Description>::type;
        };

        template<class OperatorTypeClass, OperatorTypeClass OperatorType, class Description>
        using get_t = typename get<OperatorTypeClass, OperatorType, Description>::type;

        /// get_or_default

        template<class OperatorTypeClass, OperatorTypeClass OperatorType, class Description, class Default = void>
        struct get_or_default {
        private:
            using get_type = get_t<OperatorTypeClass, OperatorType, Description>;

        public:
            using type = typename COMMONDX_STL_NAMESPACE::
                conditional<COMMONDX_STL_NAMESPACE::is_void<get_type>::value, Default, get_type>::type;
        };

        template<class OperatorTypeClass, OperatorTypeClass OperatorType, class Description, class Default = void>
        using get_or_default_t = typename get_or_default<OperatorTypeClass, OperatorType, Description, Default>::type;
    } // namespace detail
} // namespace commondx

// Namespace wrapper
#include "../../detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_TRAITS_DETAIL_GET_HPP
