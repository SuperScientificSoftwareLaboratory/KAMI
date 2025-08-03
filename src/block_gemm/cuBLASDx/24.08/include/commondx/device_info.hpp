// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef COMMONDX_DEVICE_INFO_HPP
#define COMMONDX_DEVICE_INFO_HPP

#include <cuda_fp16.h>

#include "detail/stl/type_traits.hpp"
#include "complex_types.hpp"

// Namespace wrapper
#include "detail/namespace_wrapper_open.hpp"

namespace commondx {
    template<unsigned int SM>
    struct device_info {
        template<typename DataType>
        inline static constexpr bool is_mma_available() {
            // 7X0
            if constexpr (SM < 800) {
                return COMMONDX_STL_NAMESPACE::is_same_v<DataType, __half>;
            // 8X0
            } else if constexpr (SM < 900) {
                return COMMONDX_STL_NAMESPACE::is_same_v<DataType, __half> ||
                       COMMONDX_STL_NAMESPACE::is_same_v<DataType, double> ||
                       COMMONDX_STL_NAMESPACE::is_same_v<DataType, complex<double>>;
            // 9X0
            } else if constexpr (SM < 1000) {
                return COMMONDX_STL_NAMESPACE::is_same_v<DataType, __half> ||
                       COMMONDX_STL_NAMESPACE::is_same_v<DataType, double> ||
                       COMMONDX_STL_NAMESPACE::is_same_v<DataType, complex<double>>;
            }
            // Next SM
            return false;
        }

        // Source for these chip memory numbers:
        // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability
        inline static constexpr unsigned int shared_memory() {
            if constexpr (SM == 900) {
                return 227 * 1024;
            } else if constexpr (SM == 890) {
                return 99 * 1024; // Assumes same shared memory as 860, TODO: fix
            } else if constexpr (SM == 870) {
                return 163 * 1024;
            } else if constexpr (SM == 860) {
                return 99 * 1024;
            } else if constexpr (SM == 800) {
                return 163 * 1024;
            } else if constexpr (SM == 750) {
                return 64 * 1024;
            } else if constexpr (SM == 720) {
                return 96 * 1024;
            } else if constexpr (SM == 700) {
                return 96 * 1024;
            }
            // default
            return 48 * 1024;
        }
    };
} // namespace commondx

// Namespace wrapper
#include "detail/namespace_wrapper_close.hpp"

#endif // COMMONDX_DEVICE_INFO_HPP
