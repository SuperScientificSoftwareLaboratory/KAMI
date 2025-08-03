// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited._800

#ifndef CUFFTDX_DATABASE_DATABASE_HPP
#define CUFFTDX_DATABASE_DATABASE_HPP

#include "detail/block_fft.hpp"

namespace cufftdx {
    namespace database {
        namespace detail {

            using lut_fp32_type = commondx::complex<float>;
            using lut_fp64_type = commondx::complex<double>;

#ifdef CUFFTDX_DETAIL_USE_EXTERN_LUT
            #include "lut_fp32.h"
            #include "lut_fp64.h"
#else // CUFFTDX_DETAIL_USE_EXTERN_LUT
    #ifndef CUFFTDX_DETAIL_LUT_LINKAGE
    #define CUFFTDX_DETAIL_LUT_LINKAGE static
    #endif
            #include "lut_fp32.hpp.inc"
            #include "lut_fp64.hpp.inc"
    #ifdef CUFFTDX_DETAIL_LUT_LINKAGE
    #undef CUFFTDX_DETAIL_LUT_LINKAGE
    #endif
#endif // CUFFTDX_DETAIL_USE_EXTERN_LUT

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530)
            #include "records/700/database_fp16_fwd.hpp.inc"
            #include "records/700/database_fp16_inv.hpp.inc"
#endif
            #include "records/700/database_fp32_fwd.hpp.inc"
            #include "records/700/database_fp32_inv.hpp.inc"
            #include "records/700/database_fp64_fwd.hpp.inc"
            #include "records/700/database_fp64_inv.hpp.inc"

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530)
            #include "records/750/database_fp16_fwd.hpp.inc"
            #include "records/750/database_fp16_inv.hpp.inc"
#endif
            #include "records/750/database_fp32_fwd.hpp.inc"
            #include "records/750/database_fp32_inv.hpp.inc"
            #include "records/750/database_fp64_fwd.hpp.inc"
            #include "records/750/database_fp64_inv.hpp.inc"

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530)
            #include "records/800/database_fp16_fwd.hpp.inc"
            #include "records/800/database_fp16_inv.hpp.inc"
#endif
            #include "records/800/database_fp32_fwd.hpp.inc"
            #include "records/800/database_fp32_inv.hpp.inc"
            #include "records/800/database_fp64_fwd.hpp.inc"
            #include "records/800/database_fp64_inv.hpp.inc"

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530)
            #include "records/890/database_fp16_fwd.hpp.inc"
            #include "records/890/database_fp16_inv.hpp.inc"
#endif
            #include "records/890/database_fp32_fwd.hpp.inc"
            #include "records/890/database_fp32_inv.hpp.inc"
            #include "records/890/database_fp64_fwd.hpp.inc"
            #include "records/890/database_fp64_inv.hpp.inc"

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530)
            #include "records/900/database_fp16_fwd.hpp.inc"
            #include "records/900/database_fp16_inv.hpp.inc"
#endif
            #include "records/900/database_fp32_fwd.hpp.inc"
            #include "records/900/database_fp32_inv.hpp.inc"
            #include "records/900/database_fp64_fwd.hpp.inc"
            #include "records/900/database_fp64_inv.hpp.inc"

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530)
#ifndef __HALF2_TO_UI
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int *>(&(var)))
#endif
            #include "records/definitions_fp16_fwd.hpp.inc"
            #include "records/definitions_fp16_inv.hpp.inc"
#endif
            #include "records/definitions_fp32_fwd.hpp.inc"
            #include "records/definitions_fp32_inv.hpp.inc"
            #include "records/definitions_fp64_fwd.hpp.inc"
            #include "records/definitions_fp64_inv.hpp.inc"

#ifdef __HALF2_TO_UI
#undef __HALF2_TO_UI
#endif

        } // namespace detail
    }     // namespace database
} // namespace cufftdx

#endif // CUFFTDX_DATABASE_DATABASE_HPP
