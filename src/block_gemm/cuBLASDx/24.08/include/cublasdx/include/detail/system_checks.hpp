// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_SYSTEM_CHECKS_HPP
#define CUBLASDX_DETAIL_SYSTEM_CHECKS_HPP

// We require target architecture to be Volta+ (only checking on device)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
#   error "cuBLASDx requires GPU architecture sm_70 or higher");
#endif

#ifdef __CUDACC_RTC__

// NVRTC version check
#    ifndef CUBLASDX_IGNORE_DEPRECATED_COMPILER
#        if !(__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 4))
#            error cuBLASDx requires NVRTC from CUDA Toolkit 11.4 or newer
#        endif
#    endif // CUBLASDX_IGNORE_DEPRECATED_COMPILER

// NVRTC compilation checks
#    ifndef CUBLASDX_IGNORE_DEPRECATED_COMPILER
static_assert((__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 4)),
              "cuBLASDx requires CUDA Runtime 11.4 or newer to work with NVRTC");
#    endif // CUBLASDX_IGNORE_DEPRECATED_COMPILER

#else
#    include <cuda.h>

// NVCC compilation

static_assert(CUDART_VERSION >= 11040, "cuBLASDx requires CUDA Runtime 11.4 or newer");
static_assert(CUDA_VERSION >= 11040, "cuBLASDx requires CUDA Toolkit 11.4 or newer");
#ifdef __NVCC__
static_assert((__CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 4)), "cuBLASDx requires NVCC 11.4 or newer");
#endif

// SM89 MMA's require NVCC 12.4
#ifndef CUBLASDX_SUPPORTS_SM89_MMA
#   define CUBLASDX_SUPPORTS_SM89_MMA ((__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4) || __CUDACC_VER_MAJOR__ >= 13)
#   if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890))
#       define CUBLASDX_ARCH_MMA_SM89_ENABLED
#   endif
#endif // CUBLASDX_SUPPORTS_SM89_MMA

#    ifndef CUBLASDX_IGNORE_DEPRECATED_COMPILER

// Test for GCC 7+
#        if defined(__GNUC__) && !defined(__clang__)
#            if (__GNUC__ < 7)
#                error cuBLASDx requires GCC in version 7 or newer
#            endif
#        endif // __GNUC__

// Test for clang 9+
#        ifdef __clang__
#            if (__clang_major__ < 9)
#                error cuBLASDx requires clang in version 9 or newer (experimental support for clang as host compiler)
#            endif
#        endif // __clang__

// MSVC (Visual Studio) is not supported
#        ifdef _MSC_VER
#            error cuBLASDx does not support compilation with MSVC yet
#        endif // _MSC_VER

#    endif // CUBLASDX_IGNORE_DEPRECATED_COMPILER

#endif // __CUDACC_RTC__

// C++ Version
#ifndef CUBLASDX_IGNORE_DEPRECATED_DIALECT
#    if (__cplusplus < 201703L)
#        error cuBLASDx requires C++17 (or newer) enabled
#    endif
#endif // CUBLASDX_IGNORE_DEPRECATED_DIALECT

#endif // CUBLASDX_DETAIL_SYSTEM_CHECKS_HPP
