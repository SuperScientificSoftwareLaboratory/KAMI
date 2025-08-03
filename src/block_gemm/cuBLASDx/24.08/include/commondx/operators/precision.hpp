// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef COMMONDX_OPERATORS_PRECISION_HPP
#define COMMONDX_OPERATORS_PRECISION_HPP

#include "../detail/stl/type_traits.hpp"

#include <cuda_fp16.h>

#include "../detail/expressions.hpp"

// Namespace wrapper
#include "../detail/namespace_wrapper_open.hpp"

namespace commondx {
    namespace detail {
        template<class T, class... SupportedTypes>
        struct is_supported_type:
            COMMONDX_STL_NAMESPACE::integral_constant<bool,
                                      (COMMONDX_STL_NAMESPACE::is_same<SupportedTypes, typename COMMONDX_STL_NAMESPACE::remove_cv<T>::type>::value ||
                                       ...)> {};
    } // namespace detail

    template<class T, class... SupportedPrecisions>
    struct PrecisionBase: detail::operator_expression {
        using type = typename COMMONDX_STL_NAMESPACE::remove_cv<T>::type;
        static_assert(detail::is_supported_type<type, SupportedPrecisions...>::value, "Unsupported precision type.");
    };
} // namespace commondx

// Namespace wrapper
#include "../detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_OPERATORS_TYPE_HPP
