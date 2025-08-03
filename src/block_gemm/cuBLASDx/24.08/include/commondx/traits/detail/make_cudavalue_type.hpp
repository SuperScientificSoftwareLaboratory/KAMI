// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef COMMONDX_TRAITS_DETAIL_MAKE_CUDAVALUE_TYPE_HPP
#define COMMONDX_TRAITS_DETAIL_MAKE_CUDAVALUE_TYPE_HPP

#include "../../detail/stl/type_traits.hpp"

#include "../../operators/precision.hpp"
#include "../../operators/type.hpp"

#include <cuComplex.h>

// Namespace wrapper
#include "../../detail/namespace_wrapper_open.hpp"

namespace commondx {
    namespace detail {
        template<data_type DataType, class Precision>
        struct make_cudavalue_type {
            static_assert(is_supported_type<Precision, __half, float, double>::value, "Precision must be __half, double, or float");
            using type = void;
        };

        template<class Precision>
        struct make_cudavalue_type<data_type::real, Precision> {
            using type = Precision;
        };

        template<>
        struct make_cudavalue_type<data_type::complex, __half> {
            using type = __half2;
        };

        template<>
        struct make_cudavalue_type<data_type::complex, float> {
            using type = cuComplex;
        };

        template<>
        struct make_cudavalue_type<data_type::complex, double> {
            using type = cuDoubleComplex;
        };

        template<data_type DataType, class Precision>
        using make_cudavalue_type_t = typename make_cudavalue_type<DataType, Precision>::type;
    } // namespace detail
} // namespace commondx

// Namespace wrapper
#include "../../detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_TRAITS_DETAIL_MAKE_CUDAVALUE_TYPE_HPP
