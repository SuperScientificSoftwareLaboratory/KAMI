// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_TYPES_HPP
#define CUBLASDX_TYPES_HPP

#include <cutlass/numeric_types.h>
#include <cutlass/complex.h>

#ifndef COMMONDX_DETAIL_ENABLE_CUTLASS_TYPES
#define COMMONDX_DETAIL_ENABLE_CUTLASS_TYPES
#endif
#include "commondx/complex_types.hpp"

namespace cublasdx {
    namespace detail {
        template <typename T>
        struct convert_to_cutlass_type {
            using type = T;
        };

        template<>
        struct convert_to_cutlass_type<__half> {
            using type = cutlass::half_t;
            static_assert((alignof(__half) >= alignof(type)), "cutlass type has stricter alignment requirement.");
        };

        template<>
        struct convert_to_cutlass_type<__nv_bfloat16> {
            using type = cutlass::bfloat16_t;
            static_assert((alignof(__nv_bfloat16) >= alignof(type)), "cutlass type has stricter alignment requirement.");
        };

    #ifdef COMMONDX_DETAIL_CUDA_FP8_ENABLED
        template<>
        struct convert_to_cutlass_type<__nv_fp8_e5m2> {
            using type = cutlass::float_e5m2_t;
            static_assert((alignof(__nv_fp8_e5m2) >= alignof(type)), "cutlass type has stricter alignment requirement.");
        };

        template<>
        struct convert_to_cutlass_type<__nv_fp8_e4m3> {
            using type = cutlass::float_e4m3_t;
            static_assert((alignof(__nv_fp8_e4m3) >= alignof(type)), "cutlass type has stricter alignment requirement.");
        };
    #endif

        template<typename T>
        struct convert_to_cutlass_type<commondx::complex<T>> {
            using type = cutlass::complex<typename convert_to_cutlass_type<T>::type>;
            static_assert((alignof(commondx::complex<T>) >= alignof(type)), "cutlass type has stricter alignment requirement.");
        };

        template<class T>
        using convert_to_cutlass_type_t = typename convert_to_cutlass_type<T>::type;

        template<class CT, class T>
        __forceinline__ __device__ __host__
        COMMONDX_STL_NAMESPACE::enable_if_t<cutlass::is_complex<CT>::value, CT> cast_to_cutlass_type(const T a) {
            using value_type = typename CT::value_type;
            return CT(static_cast<value_type>(a.real()), static_cast<value_type>(a.imag()));
        }

        template<class CT, class T>
        __forceinline__ __device__ __host__
        COMMONDX_STL_NAMESPACE::enable_if_t<!cutlass::is_complex<CT>::value, CT> cast_to_cutlass_type(const T a) {
            return static_cast<CT>(a);
        }

        template<class T, class CT = convert_to_cutlass_type_t<T>>
        __forceinline__ __device__ __host__
        COMMONDX_STL_NAMESPACE::enable_if_t<cutlass::is_complex<CT>::value, T> cast_from_cutlass_type(const CT a) {
            if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<typename CT::value_type, typename T::value_type>) {
                return T(a.real(), a.imag());
            }
            else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<typename CT::value_type, cutlass::half_t>) {
                return T(a.real().to_half(), a.imag().to_half());
            }
            else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<typename CT::value_type, cutlass::bfloat16_t>) {
                return T(__ushort_as_bfloat16(a.real().raw()), __ushort_as_bfloat16(a.imag().raw()));
            }
        #if COMMONDX_DETAIL_CUDA_FP8_ENABLED
            else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<typename CT::value_type, cutlass::float_e5m2_t> ||
                               COMMONDX_STL_NAMESPACE::is_same_v<typename CT::value_type, cutlass::float_e4m3_t>) {
                using value_type = typename T::value_type;
                return T(static_cast<value_type>(float(a.real())), static_cast<value_type>(float(a.imag())));
            }
        #endif
        }

        template<class T, class CT = convert_to_cutlass_type_t<T>>
        __forceinline__ __device__ __host__
        COMMONDX_STL_NAMESPACE::enable_if_t<!cutlass::is_complex<CT>::value, T> cast_from_cutlass_type(CT a) {
            if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<CT, T>) {
                return a;
            }
            else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<CT, cutlass::half_t>) {
                return a.to_half();
            }
            else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<CT, cutlass::bfloat16_t>) {
                return __ushort_as_bfloat16(a.raw());
            }
        #if COMMONDX_DETAIL_CUDA_FP8_ENABLED
            else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<CT, cutlass::float_e5m2_t> ||
                               COMMONDX_STL_NAMESPACE::is_same_v<CT, cutlass::float_e4m3_t>) {
                return static_cast<T>(float(a));
            }
        #endif
        }
    } // namespace detail

    template<class T>
    using complex      = typename commondx::complex<T>;
    using half_t       = typename commondx::half_t;
    using tfloat32_t   = typename commondx::tfloat32_t;
    using bfloat16_t   = typename commondx::bfloat16_t;
    using float_e5m2_t = typename commondx::float_e5m2_t;
    using float_e4m3_t = typename commondx::float_e4m3_t;

} // namespace cublasdx

#endif // CUBLASDX_TYPES_HPP
