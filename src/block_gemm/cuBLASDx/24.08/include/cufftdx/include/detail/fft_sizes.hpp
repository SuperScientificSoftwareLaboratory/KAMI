// Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_DETAIL_FFT_SIZES_HPP
#define CUFFTDX_DETAIL_FFT_SIZES_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#else
#    include <type_traits>
#endif

namespace cufftdx {
    namespace detail {

// SM70
#define CUFFTDX_DETAIL_SM700_FP16_MAX 16384
#define CUFFTDX_DETAIL_SM700_FP32_MAX 16384
#define CUFFTDX_DETAIL_SM700_FP64_MAX 8192
// SM72
#define CUFFTDX_DETAIL_SM720_FP16_MAX 16384
#define CUFFTDX_DETAIL_SM720_FP32_MAX 16384
#define CUFFTDX_DETAIL_SM720_FP64_MAX 8192
// SM75
#define CUFFTDX_DETAIL_SM750_FP16_MAX 16384
#define CUFFTDX_DETAIL_SM750_FP32_MAX 16384
#define CUFFTDX_DETAIL_SM750_FP64_MAX 8192
// SM80
#define CUFFTDX_DETAIL_SM800_FP16_MAX 32768
#define CUFFTDX_DETAIL_SM800_FP32_MAX 32768
#define CUFFTDX_DETAIL_SM800_FP64_MAX 16384
// SM86
#define CUFFTDX_DETAIL_SM860_FP16_MAX 16384
#define CUFFTDX_DETAIL_SM860_FP32_MAX 16384
#define CUFFTDX_DETAIL_SM860_FP64_MAX 8192
// SM87
#define CUFFTDX_DETAIL_SM870_FP16_MAX 32768
#define CUFFTDX_DETAIL_SM870_FP32_MAX 32768
#define CUFFTDX_DETAIL_SM870_FP64_MAX 16384
// SM89
#define CUFFTDX_DETAIL_SM890_FP16_MAX 16384
#define CUFFTDX_DETAIL_SM890_FP32_MAX 16384
#define CUFFTDX_DETAIL_SM890_FP64_MAX 8192
// SM90
#define CUFFTDX_DETAIL_SM900_FP16_MAX 32768
#define CUFFTDX_DETAIL_SM900_FP32_MAX 32768
#define CUFFTDX_DETAIL_SM900_FP64_MAX 16384

// Thread FFT
#define CUFFTDX_DETAIL_THREAD_FP16 64
#define CUFFTDX_DETAIL_THREAD_FP32 64
#define CUFFTDX_DETAIL_THREAD_FP64 40

        template<typename Precision>
        constexpr auto max_thread_size() {
            [[maybe_unused]] constexpr bool is_half   = CUFFTDX_STD::is_same_v<Precision, __half>;
            [[maybe_unused]] constexpr bool is_float  = CUFFTDX_STD::is_same_v<Precision, float>;
            [[maybe_unused]] constexpr bool is_double = CUFFTDX_STD::is_same_v<Precision, double>;

            if constexpr (is_half) {
                return CUFFTDX_DETAIL_THREAD_FP16;
            } else if constexpr (is_float) {
                return CUFFTDX_DETAIL_THREAD_FP32;
            } else if constexpr (is_double) {
                return CUFFTDX_DETAIL_THREAD_FP64;
            }

            return 0;
        }

        template<typename Precision>
        constexpr auto max_block_size(int sm) {
            [[maybe_unused]] constexpr bool is_half   = CUFFTDX_STD::is_same_v<Precision, __half>;
            [[maybe_unused]] constexpr bool is_float  = CUFFTDX_STD::is_same_v<Precision, float>;
            [[maybe_unused]] constexpr bool is_double = CUFFTDX_STD::is_same_v<Precision, double>;

            if constexpr (is_half) {
                switch (sm) {
                    case 700: return CUFFTDX_DETAIL_SM700_FP16_MAX;
                    case 720: return CUFFTDX_DETAIL_SM720_FP16_MAX;
                    case 750: return CUFFTDX_DETAIL_SM750_FP16_MAX;
                    case 800: return CUFFTDX_DETAIL_SM800_FP16_MAX;
                    case 860: return CUFFTDX_DETAIL_SM860_FP16_MAX;
                    case 870: return CUFFTDX_DETAIL_SM870_FP16_MAX;
                    case 890: return CUFFTDX_DETAIL_SM890_FP16_MAX;
                    case 900: return CUFFTDX_DETAIL_SM900_FP16_MAX;
                    default: return 0;
                }
            } else if constexpr (is_float) {
                switch (sm) {
                    case 700: return CUFFTDX_DETAIL_SM700_FP32_MAX;
                    case 720: return CUFFTDX_DETAIL_SM720_FP32_MAX;
                    case 750: return CUFFTDX_DETAIL_SM750_FP32_MAX;
                    case 800: return CUFFTDX_DETAIL_SM800_FP32_MAX;
                    case 860: return CUFFTDX_DETAIL_SM860_FP32_MAX;
                    case 870: return CUFFTDX_DETAIL_SM870_FP32_MAX;
                    case 890: return CUFFTDX_DETAIL_SM890_FP32_MAX;
                    case 900: return CUFFTDX_DETAIL_SM900_FP32_MAX;
                    default: return 0;
                }
            } else if constexpr (is_double) {
                switch (sm) {
                    case 700: return CUFFTDX_DETAIL_SM700_FP64_MAX;
                    case 720: return CUFFTDX_DETAIL_SM720_FP64_MAX;
                    case 750: return CUFFTDX_DETAIL_SM750_FP64_MAX;
                    case 800: return CUFFTDX_DETAIL_SM800_FP64_MAX;
                    case 860: return CUFFTDX_DETAIL_SM860_FP64_MAX;
                    case 870: return CUFFTDX_DETAIL_SM870_FP64_MAX;
                    case 890: return CUFFTDX_DETAIL_SM890_FP64_MAX;
                    case 900: return CUFFTDX_DETAIL_SM900_FP64_MAX;
                    default: return 0;
                }
            }

            return 0;
        }
    }
}

#endif // CUFFTDX_DETAIL_FFT_SIZES_HPP
