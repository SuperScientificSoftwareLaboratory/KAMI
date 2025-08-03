// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_TENSOR_CONFIGS_HPP
#define CUBLASDX_DATABASE_CUTE_TENSOR_CONFIGS_HPP

#include "cute_tensor.hpp"

namespace cublasdx {
    namespace detail {
        namespace cute_backend {
            template<class T>
            struct map_precision_type {
                using precision_type = T;
            };
            template<typename T>
            struct map_precision_type<cutlass::complex<T>> {
                using precision_type = T;
            };
            template<typename T>
            using map_precision_type_t = typename map_precision_type<T>::precision_type;

            // This list is currently incomplete,for extensive testing add all cute::MMA types here
            enum class mma_atom
            {
                universal_fma,
                SM70_8x8x4_F16F16F16F16_TN,
                SM70_8x8x4_C16C16C16C16_TN_CUBLASDX,
                SM70_8x8x4_F32F16F16F32_TN,
                SM70_8x8x4_C32C16C16C32_TN_CUBLASDX,
                SM75_16x8x8_F16F16F16F16_TN_CUBLASDX,
                SM75_16x8x8_C16C16C16C16_TN_CUBLASDX,
                SM75_16x8x8_F32F16F16F32_TN_CUBLASDX,
                SM75_16x8x8_C32C16C16C32_TN_CUBLASDX,
                SM80_16x8x16_F16F16F16F16_TN,
                SM80_16x8x16_C16C16C16C16_TN_CUBLASDX,
                SM80_16x8x16_F32F16F16F32_TN,
                SM80_16x8x16_C32C16C16C32_TN_CUBLASDX,
                SM80_8x8x4_F64F64F64F64_TN,
                SM80_8x8x4_C64C64C64C64_TN,
                SM80_16x8x8_F32BF16BF16F32_TN,
                SM80_16x8x8_C32BC16BC16C32_TN_CUBLASDX,
                SM80_16x8x16_F32BF16BF16F32_TN,
                SM80_16x8x16_C32BC16BC16C32_TN_CUBLASDX,
                SM80_16x8x4_F32TF32TF32F32_TN,
                SM80_16x8x4_C32TC32TC32C32_TN_CUBLASDX,
                SM80_16x8x8_F32TF32TF32F32_TN,
                SM80_16x8x8_C32TC32TC32C32_TN_CUBLASDX,
                SM89_16x8x32_F32E4M3E4M3F32_TN_CUBLASDX,
                SM89_16x8x32_C32CE4M3CE4M3C32_TN_CUBLASDX,
                SM89_16x8x32_F32E4M3E5M2F32_TN_CUBLASDX,
                SM89_16x8x32_C32CE4M3CE5M2C32_TN_CUBLASDX,
                SM89_16x8x32_F32E5M2E4M3F32_TN_CUBLASDX,
                SM89_16x8x32_C32CE5M2CE4M3C32_TN_CUBLASDX,
                SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX,
                SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX,
                SM90_16x8x4_F64F64F64F64_TN,
                SM90_16x8x8_F64F64F64F64_TN,
                SM90_16x8x16_F64F64F64F64_TN,
                SM90_16x8x4_C64C64C64C64_TN,
                SM90_16x8x8_C64C64C64C64_TN,
                SM90_16x8x16_C64C64C64C64_TN
            };

            template<typename AType, typename BType, typename CType, unsigned int SM>
            constexpr mma_atom get_default_mma() {
                [[maybe_unused]] constexpr bool is_complex_type = cutlass::is_complex<AType>::value;
                using A_prec_t = map_precision_type_t<AType>;
                using B_prec_t = map_precision_type_t<BType>;
                using C_prec_t = map_precision_type_t<CType>;

                mma_atom ret = mma_atom::universal_fma;
                if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<COMMONDX_STL_NAMESPACE::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                                COMMONDX_STL_NAMESPACE::tuple<half_t, half_t, half_t>>) {
                    // warning: checks on SM should be in the decreasing order
                    if constexpr (SM >= 800) {
                        ret = is_complex_type ? mma_atom::SM80_16x8x16_C16C16C16C16_TN_CUBLASDX : mma_atom::SM80_16x8x16_F16F16F16F16_TN;
                    } else if constexpr (SM >= 750) {
                        ret = is_complex_type ? mma_atom::SM75_16x8x8_C16C16C16C16_TN_CUBLASDX : mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX;
                    } else if constexpr (SM >= 700) {
                        ret = is_complex_type ? mma_atom::SM70_8x8x4_C16C16C16C16_TN_CUBLASDX : mma_atom::SM70_8x8x4_F16F16F16F16_TN;
                    }
                } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<COMMONDX_STL_NAMESPACE::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                                       COMMONDX_STL_NAMESPACE::tuple<half_t, half_t, float>>) {
                    if constexpr (SM >= 800) {
                        ret = is_complex_type ? mma_atom::SM80_16x8x16_C32C16C16C32_TN_CUBLASDX : mma_atom::SM80_16x8x16_F32F16F16F32_TN;
                    } else if constexpr (SM >= 750) {
                        ret = is_complex_type ? mma_atom::SM75_16x8x8_C32C16C16C32_TN_CUBLASDX : mma_atom::SM75_16x8x8_F32F16F16F32_TN_CUBLASDX;
                    } else if constexpr (SM >= 700) {
                        ret = is_complex_type ? mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX : mma_atom::SM70_8x8x4_F32F16F16F32_TN;
                    }
                } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<COMMONDX_STL_NAMESPACE::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                                       COMMONDX_STL_NAMESPACE::tuple<double, double, double>>) {
                    if constexpr (SM >= 900) {
                        ret = is_complex_type ? mma_atom::SM90_16x8x4_C64C64C64C64_TN : mma_atom::SM90_16x8x4_F64F64F64F64_TN;
                    } else if constexpr (SM >= 800) {
                        ret = is_complex_type ? mma_atom::SM80_8x8x4_C64C64C64C64_TN : mma_atom::SM80_8x8x4_F64F64F64F64_TN;
                    }
                } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<COMMONDX_STL_NAMESPACE::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                                       COMMONDX_STL_NAMESPACE::tuple<bfloat16_t, bfloat16_t, float>>) {
                    if constexpr (SM >= 800) {
                        ret = is_complex_type ? mma_atom::SM80_16x8x16_C32BC16BC16C32_TN_CUBLASDX : mma_atom::SM80_16x8x16_F32BF16BF16F32_TN;
                    }
                } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<COMMONDX_STL_NAMESPACE::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                                       COMMONDX_STL_NAMESPACE::tuple<tfloat32_t, tfloat32_t, float>>) {
                    if constexpr (SM >= 800) {
                        ret = is_complex_type ? mma_atom::SM80_16x8x8_C32TC32TC32C32_TN_CUBLASDX : mma_atom::SM80_16x8x8_F32TF32TF32F32_TN;
                    }
                } else if constexpr  (COMMONDX_STL_NAMESPACE::is_same_v<COMMONDX_STL_NAMESPACE::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                                        COMMONDX_STL_NAMESPACE::tuple<float_e4m3_t, float_e4m3_t, float>>) {
                    if constexpr (SM >= 890) {
                    #if CUBLASDX_SUPPORTS_SM89_MMA
                        ret = is_complex_type ? mma_atom::SM89_16x8x32_C32CE4M3CE4M3C32_TN_CUBLASDX : mma_atom::SM89_16x8x32_F32E4M3E4M3F32_TN_CUBLASDX;
                    #endif
                    }
                } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<COMMONDX_STL_NAMESPACE::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                                       COMMONDX_STL_NAMESPACE::tuple<float_e4m3_t, float_e5m2_t, float>>) {
                    if constexpr (SM >= 890) {
                    #if CUBLASDX_SUPPORTS_SM89_MMA
                        ret = is_complex_type ? mma_atom::SM89_16x8x32_C32CE4M3CE5M2C32_TN_CUBLASDX : mma_atom::SM89_16x8x32_F32E4M3E5M2F32_TN_CUBLASDX;
                    #endif
                    }
                } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<COMMONDX_STL_NAMESPACE::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                                       COMMONDX_STL_NAMESPACE::tuple<float_e5m2_t, float_e4m3_t, float>>) {
                    if constexpr (SM >= 890) {
                    #if CUBLASDX_SUPPORTS_SM89_MMA
                        ret = is_complex_type ? mma_atom::SM89_16x8x32_C32CE5M2CE4M3C32_TN_CUBLASDX : mma_atom::SM89_16x8x32_F32E5M2E4M3F32_TN_CUBLASDX;
                    #endif
                    }
                } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<COMMONDX_STL_NAMESPACE::tuple<A_prec_t, B_prec_t, C_prec_t>,
                                                                       COMMONDX_STL_NAMESPACE::tuple<float_e5m2_t, float_e5m2_t, float>>) {
                    if constexpr (SM >= 890) {
                    #if CUBLASDX_SUPPORTS_SM89_MMA
                        ret = is_complex_type ? mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX : mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX;
                    #endif
                    }
                }
                return ret;
            }

            // This function returns instance of CuTe type
            // can be used with decltype to get just the type
            template<typename AType, typename BType, typename CType, mma_atom MmaAtom>
            constexpr auto convert_mma_atom_to_cute() {
                if constexpr (MmaAtom == mma_atom::universal_fma) {
                    return UniversalFMA<AType, BType, CType> {};
                }
                if constexpr (MmaAtom == mma_atom::SM70_8x8x4_F16F16F16F16_TN) {
                    return cute::SM70_8x8x4_F16F16F16F16_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM70_8x8x4_C16C16C16C16_TN_CUBLASDX) {
                    return SM70_8x8x4_C16C16C16C16_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM70_8x8x4_F32F16F16F32_TN) {
                    return cute::SM70_8x8x4_F32F16F16F32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX) {
                    return SM70_8x8x4_C32C16C16C32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX) {
                    return SM75_16x8x8_F16F16F16F16_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM75_16x8x8_C16C16C16C16_TN_CUBLASDX) {
                    return SM75_16x8x8_C16C16C16C16_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM75_16x8x8_F32F16F16F32_TN_CUBLASDX) {
                    return SM75_16x8x8_F32F16F16F32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM75_16x8x8_C32C16C16C32_TN_CUBLASDX) {
                    return SM75_16x8x8_C32C16C16C32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM80_16x8x16_F16F16F16F16_TN) {
                    return cute::SM80_16x8x16_F16F16F16F16_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM80_16x8x16_C16C16C16C16_TN_CUBLASDX) {
                    return SM80_16x8x16_C16C16C16C16_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM80_16x8x16_F32F16F16F32_TN) {
                    return cute::SM80_16x8x16_F32F16F16F32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM80_16x8x16_C32C16C16C32_TN_CUBLASDX) {
                    return SM80_16x8x16_C32C16C16C32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM80_8x8x4_F64F64F64F64_TN) {
                    return cute::SM80_8x8x4_F64F64F64F64_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM80_8x8x4_C64C64C64C64_TN) {
                    return cute::SM80_8x8x4_C64C64C64C64_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM80_16x8x8_F32BF16BF16F32_TN) {
                    return cute::SM80_16x8x8_F32BF16BF16F32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM80_16x8x8_C32BC16BC16C32_TN_CUBLASDX) {
                    return SM80_16x8x8_C32BC16BC16C32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM80_16x8x16_F32BF16BF16F32_TN) {
                    return cute::SM80_16x8x16_F32BF16BF16F32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM80_16x8x16_C32BC16BC16C32_TN_CUBLASDX) {
                    return SM80_16x8x16_C32BC16BC16C32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM80_16x8x4_F32TF32TF32F32_TN) {
                    return cute::SM80_16x8x4_F32TF32TF32F32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM80_16x8x4_C32TC32TC32C32_TN_CUBLASDX) {
                    return SM80_16x8x4_C32TC32TC32C32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM80_16x8x8_F32TF32TF32F32_TN) {
                    return cute::SM80_16x8x8_F32TF32TF32F32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM80_16x8x8_C32TC32TC32C32_TN_CUBLASDX) {
                    return SM80_16x8x8_C32TC32TC32C32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM89_16x8x32_F32E4M3E4M3F32_TN_CUBLASDX) {
                    return SM89_16x8x32_F32E4M3E4M3F32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM89_16x8x32_C32CE4M3CE4M3C32_TN_CUBLASDX) {
                    return SM89_16x8x32_C32CE4M3CE4M3C32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM89_16x8x32_F32E4M3E5M2F32_TN_CUBLASDX) {
                    return SM89_16x8x32_F32E4M3E5M2F32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM89_16x8x32_C32CE4M3CE5M2C32_TN_CUBLASDX) {
                    return SM89_16x8x32_C32CE4M3CE5M2C32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM89_16x8x32_F32E5M2E4M3F32_TN_CUBLASDX) {
                    return SM89_16x8x32_F32E5M2E4M3F32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM89_16x8x32_C32CE5M2CE4M3C32_TN_CUBLASDX) {
                    return SM89_16x8x32_C32CE5M2CE4M3C32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX) {
                    return SM89_16x8x32_F32E5M2E5M2F32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX) {
                    return SM89_16x8x32_C32CE5M2CE5M2C32_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM90_16x8x4_F64F64F64F64_TN) {
                    return cute::SM90_16x8x4_F64F64F64F64_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM90_16x8x8_F64F64F64F64_TN) {
                    return cute::SM90_16x8x8_F64F64F64F64_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM90_16x8x16_F64F64F64F64_TN) {
                    return cute::SM90_16x8x16_F64F64F64F64_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM90_16x8x4_C64C64C64C64_TN) {
                    return cute::SM90_16x8x4_C64C64C64C64_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM90_16x8x8_C64C64C64C64_TN) {
                    return cute::SM90_16x8x8_C64C64C64C64_TN {};
                }
                if constexpr (MmaAtom == mma_atom::SM90_16x8x16_C64C64C64C64_TN) {
                    return cute::SM90_16x8x16_C64C64C64C64_TN {};
                }
            }
        } // namespace cute_backend
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_CUTE_TENSOR_CONFIGS_HPP
