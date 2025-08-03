// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_OPERATORS_ALIGNMENT_HPP
#define CUBLASDX_OPERATORS_ALIGNMENT_HPP

#include "commondx/detail/expressions.hpp"
#include "commondx/traits/detail/is_operator_fd.hpp"
#include "commondx/traits/detail/get_operator_fd.hpp"

namespace cublasdx {
    // [A/B/C]Alignment - alignment in bytes
    template<unsigned int AAlignment, unsigned int BAlignment, unsigned int CAlignment>
    struct Alignment: commondx::detail::operator_expression {
    private:
        static constexpr unsigned int max_alignment = 16;

        template<unsigned int Alignment>
        static constexpr bool is_power_of_two() {
            return !(Alignment == 0) && !(Alignment & (Alignment - 1));
        }

        static_assert(is_power_of_two<AAlignment>(), "AAlignment has to be a power of two");
        static_assert(is_power_of_two<BAlignment>(), "BAlignment has to be a power of two");
        static_assert(is_power_of_two<CAlignment>(), "CAlignment has to be a power of two");

    public:
        static constexpr unsigned int a = (AAlignment > max_alignment) ? max_alignment : AAlignment;
        static constexpr unsigned int b = (BAlignment > max_alignment) ? max_alignment : BAlignment;
        static constexpr unsigned int c = (CAlignment > max_alignment) ? max_alignment : CAlignment;
    };

    template<unsigned int AAlignment, unsigned int BAlignment, unsigned int CAlignment>
    constexpr unsigned int Alignment<AAlignment, BAlignment, CAlignment>::a;

    template<unsigned int AAlignment, unsigned int BAlignment, unsigned int CAlignment>
    constexpr unsigned int Alignment<AAlignment, BAlignment, CAlignment>::b;

    template<unsigned int AAlignment, unsigned int BAlignment, unsigned int CAlignment>
    constexpr unsigned int Alignment<AAlignment, BAlignment, CAlignment>::c;

    using MaxAlignment = Alignment<16, 16, 16>;
} // namespace cublasdx

namespace commondx::detail {
    template<unsigned int A, unsigned int B, unsigned int C>
    struct is_operator<cublasdx::operator_type, cublasdx::operator_type::alignment, cublasdx::Alignment<A, B, C>>:
        COMMONDX_STL_NAMESPACE::true_type {
    };

    template<unsigned int A, unsigned int B, unsigned int C>
    struct get_operator_type<cublasdx::Alignment<A, B, C>> {
        static constexpr cublasdx::operator_type value = cublasdx::operator_type::alignment;
    };
} // namespace commondx::detail

#endif // CUBLASDX_OPERATORS_ALIGNMENT_HPP
