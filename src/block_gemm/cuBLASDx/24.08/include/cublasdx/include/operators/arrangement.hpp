// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_ARRANGEMENT_HPP
#define CUBLASDX_OPERATORS_ARRANGEMENT_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cublasdx {
    enum class arrangement
    {
        col_major,
        row_major,
    };

    inline constexpr auto col_major   = arrangement::col_major;
    inline constexpr auto left_layout = arrangement::col_major;
    inline constexpr auto row_major   = arrangement::row_major;
    inline constexpr auto right_major = arrangement::row_major;

    template<arrangement AOrder, arrangement BOrder = arrangement::col_major, arrangement COrder = arrangement::col_major>
    struct Arrangement: commondx::detail::operator_expression {
        static_assert((AOrder == arrangement::col_major) || (AOrder == arrangement::row_major),
                      "A order has to be col_major or row_major");
        static_assert((BOrder == arrangement::col_major) || (BOrder == arrangement::row_major),
                      "B order has to be col_major or row_major");
        static_assert((COrder == arrangement::col_major) || (COrder == arrangement::row_major),
                      "C order has to be col_major or row_major");

        static constexpr arrangement a = AOrder;
        static constexpr arrangement b = BOrder;
        static constexpr arrangement c = COrder;
    };

    namespace detail {
        using default_blas_arrangement_operator = Arrangement<arrangement::row_major, arrangement::col_major, arrangement::col_major>;
    } // namespace detail
} // namespace cublasdx

namespace commondx::detail {
    template<cublasdx::arrangement AOrder, cublasdx::arrangement BOrder, cublasdx::arrangement COrder>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::arrangement, cublasdx::Arrangement<AOrder, BOrder, COrder>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<cublasdx::arrangement AOrder, cublasdx::arrangement BOrder, cublasdx::arrangement COrder>
    struct get_operator_type<cublasdx::Arrangement<AOrder, BOrder, COrder>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::arrangement;
    };
} // namespace commondx::detail

#endif // CUBLASDX_OPERATORS_ARRANGEMENT_HPP
