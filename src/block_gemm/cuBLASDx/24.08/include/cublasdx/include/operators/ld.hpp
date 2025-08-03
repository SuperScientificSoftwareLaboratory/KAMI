// Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_LD_HPP
#define CUBLASDX_OPERATORS_LD_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cublasdx {
    template<unsigned int LDA, unsigned int LDB, unsigned int LDC>
    struct LeadingDimension: commondx::detail::operator_expression {
        static constexpr unsigned int a = LDA;
        static constexpr unsigned int b = LDB;
        static constexpr unsigned int c = LDC;
    };

    template<unsigned int LDA, unsigned int LDB, unsigned int LDC>
    constexpr unsigned int LeadingDimension<LDA, LDB, LDC>::a;

    template<unsigned int LDA, unsigned int LDB, unsigned int LDC>
    constexpr unsigned int LeadingDimension<LDA, LDB, LDC>::b;

    template<unsigned int LDA, unsigned int LDB, unsigned int LDC>
    constexpr unsigned int LeadingDimension<LDA, LDB, LDC>::c;
} // namespace cublasdx


namespace commondx::detail {
    template<unsigned int LDA, unsigned int LDB, unsigned int LDC>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::ld, cublasdx::LeadingDimension<LDA, LDB, LDC>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<unsigned int LDA, unsigned int LDB, unsigned int LDC>
    struct get_operator_type<cublasdx::LeadingDimension<LDA, LDB, LDC>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::ld;
    };
} // namespace commondx::detail

#endif // CUBLASDX_OPERATORS_LeadingDimension_HPP
