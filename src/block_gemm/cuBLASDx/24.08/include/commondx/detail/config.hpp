// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef COMMONDX_DETAIL_CONFIG_HPP
#define COMMONDX_DETAIL_CONFIG_HPP

// Namespace wrapper
#include "../detail/namespace_wrapper_open.hpp"

#ifdef __CUDACC_RTC__
#    define COMMONDX_DETAIL_USE_CUDA_STL
#endif

#ifdef COMMONDX_DETAIL_USE_CUDA_STL
#    define COMMONDX_STL_NAMESPACE cuda::std
#else
#    define COMMONDX_STL_NAMESPACE std
#endif

// Namespace wrapper
#include "../detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_DETAIL_CONFIG_HPP
