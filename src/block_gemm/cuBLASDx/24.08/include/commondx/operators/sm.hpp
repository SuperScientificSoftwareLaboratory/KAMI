// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef COMMONDX_OPERATORS_SM_HPP
#define COMMONDX_OPERATORS_SM_HPP

#include "../detail/expressions.hpp"

#include "../traits/detail/is_operator_fd.hpp"
#include "../traits/detail/get_operator_fd.hpp"

// Namespace wrapper
#include "../detail/namespace_wrapper_open.hpp"

namespace commondx {
    template<unsigned int Architecture>
    struct SM;

    template<>
    struct SM<700>: public detail::constant_operator_expression<unsigned int, 700> {};

    template<>
    struct SM<720>: public detail::constant_operator_expression<unsigned int, 720> {};

    template<>
    struct SM<750>: public detail::constant_operator_expression<unsigned int, 750> {};

    template<>
    struct SM<800>: public detail::constant_operator_expression<unsigned int, 800> {};

    template<>
    struct SM<860>: public detail::constant_operator_expression<unsigned int, 860> {};

    template<>
    struct SM<870>: public detail::constant_operator_expression<unsigned int, 870> {};

    template<>
    struct SM<890>: public detail::constant_operator_expression<unsigned int, 890> {};

    template<>
    struct SM<900>: public detail::constant_operator_expression<unsigned int, 900> {};
} // namespace commondx

// Namespace wrapper
#include "../detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_OPERATORS_SM_HPP
