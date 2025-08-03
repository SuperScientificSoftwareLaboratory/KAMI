// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef COMMONDX_TYPE_LIST_HPP
#define COMMONDX_TYPE_LIST_HPP

// Namespace wrapper
#include "detail/namespace_wrapper_open.hpp"

namespace commondx {

    template<int Index, class T>
    struct type_list_element;

    template<class... Elements>
    struct type_list {};

    template<int Index, class Head, class... Tail>
    struct type_list_element<Index, type_list<Head, Tail...>>: type_list_element<Index - 1, type_list<Tail...>> {};

    template<class Head, class... Tail>
    struct type_list_element<0, type_list<Head, Tail...>> {
        using type = Head;
    };

} // namespace commondx

// Namespace wrapper
#include "detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_TYPE_LIST_HPP
