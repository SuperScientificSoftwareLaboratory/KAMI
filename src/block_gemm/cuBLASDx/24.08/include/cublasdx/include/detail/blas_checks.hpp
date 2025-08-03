// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_BLAS_CHECK_HPP
#define CUBLASDX_DETAIL_BLAS_CHECK_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/device_info.hpp"

#include "../operators.hpp"
#include "../traits.hpp"

namespace cublasdx {
    namespace detail {

        template<class Precision,
                 cublasdx::type Type,
                 unsigned int   ASize,
                 unsigned int   BSize,
                 unsigned int   CSize,
                 unsigned int   Arch>
        struct is_supported_helper1 {
            static constexpr unsigned int type_multiplier = (Type == cublasdx::type::real) ? 1 : 2;
            static constexpr size_t       required_shared_memory =
                sizeof(typename Precision::a_type) * type_multiplier * (ASize) +
                sizeof(typename Precision::b_type) * type_multiplier * (BSize) +
                sizeof(typename Precision::c_type) * type_multiplier * (CSize);
            static constexpr bool value = (required_shared_memory <= commondx::device_info<Arch>::shared_memory());
        };

        template<class Precision, cublasdx::type Type, class Size, unsigned int Arch>
        struct is_supported_logical_size:
            public COMMONDX_STL_NAMESPACE::bool_constant<is_supported_helper1<Precision,
                                                              Type,
                                                              (Size::m * Size::k),
                                                              (Size::k * Size::n),
                                                              (Size::m * Size::n),
                                                              Arch>::value> {
        };

        // Checks if matrices A, B, C fits into shared memory
        template<class Precision,
                 cublasdx::type Type,
                 unsigned int   ASize,
                 unsigned int   BSize,
                 unsigned int   CSize,
                 unsigned int   Arch>
        struct is_supported_real_size:
            public COMMONDX_STL_NAMESPACE::bool_constant<is_supported_helper1<Precision, Type, ASize, BSize, CSize, Arch>::value> {
        };

        template<class Precision,
                 cublasdx::type Type,
                 arrangement AArr,
                 arrangement BArr,
                 arrangement CArr,
                 class Size,
                 class LD,
                 unsigned int Arch>
        struct is_supported_impl {
            static constexpr auto lda = LD::a;
            static constexpr auto ldb = LD::b;
            static constexpr auto ldc = LD::c;

            static constexpr unsigned int m = Size::m;
            static constexpr unsigned int n = Size::n;
            static constexpr unsigned int k = Size::k;

            static constexpr auto a_size = lda * ((AArr == arrangement::col_major) ? k : m);
            static constexpr auto b_size = ldb * ((BArr == arrangement::col_major) ? n : k);
            static constexpr auto c_size = ldc * ((CArr == arrangement::col_major) ? n : m);

            static constexpr bool value = is_supported_helper1<Precision, Type, a_size, b_size, c_size, Arch>::value;
        };

        template<class Precision,
                 cublasdx::type Type,
                 arrangement AArr,
                 arrangement BArr,
                 arrangement CArr,
                 class Size,
                 unsigned int Arch>
        struct is_supported_impl<Precision, Type, AArr, BArr, CArr, Size, void, Arch>:
            public COMMONDX_STL_NAMESPACE::bool_constant<is_supported_helper1<Precision,
                                                              Type,
                                                              (Size::m * Size::k),
                                                              (Size::k * Size::n),
                                                              (Size::m * Size::n),
                                                              Arch>::value> {
        };
    } // namespace detail

    // Check if a description is supported on a given CUDA architecture
    template<class Description, unsigned int Architecture>
    struct is_supported:
        public COMMONDX_STL_NAMESPACE::bool_constant<
            detail::is_supported_impl<precision_of<Description>,
                                      type_of_v<Description>,
                                      arrangement_of<Description>::a,
                                      arrangement_of<Description>::b,
                                      arrangement_of<Description>::c,
                                      detail::get_t<operator_type::size, Description>,
                                      detail::get_t<operator_type::ld, Description>,
                                      Architecture>::value> {
    };

    template<class Description, unsigned int Architecture>
    inline constexpr bool is_supported_v = is_supported<Description, Architecture>::value;
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_BLAS_CHECK_HPP
