// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_SUGGESTED_LAYOUTS_HPP
#define CUBLASDX_SUGGESTED_LAYOUTS_HPP

#include "cute_tensor.hpp"
#include "cute_utils.hpp"
#include "cute_tensor_configs.hpp"

namespace cublasdx {
    namespace detail {
        namespace layout_database {

        using vectorizing_copy = cute::AutoVectorizingCopyWithAssumedAlignment<128>;
        using identity_swizzle = cute::Swizzle<0, 4, 3>;

        using cublasdx::detail::cute_backend::mma_atom;
        using cublasdx::detail::matrix;

        // Check if layout of size rows / cols can be created from a tile of size tile_rows / tile_cols
        constexpr bool is_divisible(int m, int n, int k, int div_m, int div_n, int div_k) {
            return (m % div_m == 0) and (n % div_n == 0) and (k % div_k == 0);
        }

        constexpr bool is_divisible(bool is_left, int rows, int cols, int div_rows, int div_cols) {
            return (is_left and (rows % div_rows == 0) and (cols % div_cols == 0)) or
                   (not is_left and (rows % div_cols == 0) and (cols % div_rows == 0));
        }

        template<int SM, int SMMin, int SMMax, int M, int N, int K,
                 int GEMMMDiv, int GEMMNDiv, int GEMMKDiv,
                 bool IsALayoutLeft, int AMDiv, int AKDiv, bool AllowDoubleAMDiv,
                 bool IsBLayoutLeft, int BKDiv, int BNDiv, bool AllowDoubleBKDiv,
                 typename ComputeType>
        using suggested_enable_if_t = COMMONDX_STL_NAMESPACE::enable_if_t<
                // Is this SM in the requested range
                (SM >= SMMin and (SMMax == 0 or SM <= SMMax)) and
                // Is this GEMM tileable from permuted MMA tile
                (is_divisible(M, N, K, GEMMMDiv, GEMMNDiv, GEMMKDiv)) and
                // Is this A Matrix tileable from ATile
                (is_divisible(IsALayoutLeft, M, K, AMDiv, AKDiv)) and
                // If a bigger config is available turn the smaller one off
                (AllowDoubleAMDiv or not is_divisible(IsALayoutLeft, M, K, 2 * AMDiv, AKDiv)) and
                // Is this B Matrix tileable from BTile
                (is_divisible(IsBLayoutLeft, K, N, BKDiv, BNDiv)) and
                // If a bigger config is available turn the smaller one off
                (AllowDoubleBKDiv or not is_divisible(IsBLayoutLeft, K, N, 2 * BKDiv, BNDiv)) and
                // LDSM is used only for 8/16/32 bit types
                ((sizeof(ComputeType) > sizeof(float)) or
                // Necessary for LDSM to have enough data to load in x4 mode
                ((M * K >= (2048 / sizeof(ComputeType))) and (K * N >= (2048 / sizeof(ComputeType)))))
        >;


        // Helper for creating a layout from atom and swizzle
        template<bool IsLayoutLeft, int Rows, int Cols, int TileRows, int TileCols, typename TileSwizzle = identity_swizzle>
        constexpr auto optimal_layout_impl() {
            using stride_atom = COMMONDX_STL_NAMESPACE::conditional_t<IsLayoutLeft, cute::LayoutLeft, cute::LayoutRight>;
            using shape_atom = cute::Shape<cute::Int<IsLayoutLeft ? TileRows : TileCols>,
                                           cute::Int<IsLayoutLeft ? TileCols : TileRows>>;
            using atom_layout = decltype(cute::make_layout(shape_atom{}, stride_atom{}));
            // If TileSwizzle has 0 in B this composition becomes an identity operation
            using swizzled_atom_layout = decltype(composition(TileSwizzle{}, cute::Int<0>{}, atom_layout{}));
            using global_shape = cute::Shape<cute::Int<Rows>,cute::Int<Cols>>;
            using global_layout = decltype(cute::make_layout(global_shape{}, stride_atom{}));

            return cute::tile_to_shape(swizzled_atom_layout{}, global_layout{});
        }

        // Fallback for layouts not divisible or not having atoms corresponding to their configs
        template<int Threads, int SM,
                 typename TA, bool IsALayoutLeft, int AlignmentA,
                 typename TB, bool IsBLayoutLeft, int AlignmentB,
                 typename TC, bool IsCLayoutLeft, int AlignmentC,
                 int M, int N, int K, typename RequireDivisibility = void>
        struct optimal_config {
            using a_stride = COMMONDX_STL_NAMESPACE::conditional_t<IsALayoutLeft, cute::LayoutLeft, cute::LayoutRight>;
            using a_layout = decltype(cute::make_layout(cute::Shape<cute::Int<M>, cute::Int<K>>{}, a_stride{}));
            using a_copy_op = cute::AutoVectorizingCopyWithAssumedAlignment<8 * AlignmentA>;

            using b_stride = COMMONDX_STL_NAMESPACE::conditional_t<IsBLayoutLeft, cute::LayoutLeft, cute::LayoutRight>;
            using b_layout = decltype(cute::make_layout(cute::Shape<cute::Int<K>, cute::Int<N>>{}, b_stride{}));
            using b_copy_op = cute::AutoVectorizingCopyWithAssumedAlignment<8 * AlignmentB>;

            using c_stride = COMMONDX_STL_NAMESPACE::conditional_t<IsCLayoutLeft, cute::LayoutLeft, cute::LayoutRight>;
            using c_layout = decltype(cute::make_layout(cute::Shape<cute::Int<M>, cute::Int<N>>{}, c_stride{}));
            using c_copy_op = cute::AutoVectorizingCopyWithAssumedAlignment<8 * AlignmentC>;

            using TiledMma = cute::TiledMMA<
                cute::MMA_Atom<cute::UniversalFMA<TC, TA, TB, TC>>,
                cute::Layout<cute::Shape<cute::_1, cute::Int<Threads>, cute::_1>>>;

            static constexpr bool optimal = false;
        };

        // General layout templates
        template<int M, int N, int K, bool IsALayoutLeft, bool IsBLayoutLeft, bool IsCLayoutLeft,
                 int ATileRows, int ATileCols, int BTileRows, int BTileCols,
                 typename ASwizzle, typename BSwizzle,
                 typename CopyOpLeft, typename CopyOpRight,
                 typename MmaAtom, typename MmaLayout, typename MmaTile>
        struct optimal_layouts {
            using a_layout = decltype(optimal_layout_impl<IsALayoutLeft, M, K, ATileRows, ATileCols, ASwizzle>());
            using a_copy_op = COMMONDX_STL_NAMESPACE::conditional_t<IsALayoutLeft, CopyOpLeft, CopyOpRight>;

            using b_layout = decltype(optimal_layout_impl<IsBLayoutLeft, K, N, BTileRows, BTileCols, BSwizzle>());
            using b_copy_op = COMMONDX_STL_NAMESPACE::conditional_t<IsBLayoutLeft, CopyOpRight, CopyOpLeft>;

            using c_stride = COMMONDX_STL_NAMESPACE::conditional_t<IsCLayoutLeft, cute::LayoutLeft, cute::LayoutRight>;
            using c_layout = decltype(cute::make_layout(cute::Shape<cute::Int<M>, cute::Int<N>>{}, c_stride{}));
            using c_copy_op = vectorizing_copy;

            using TiledMma = cute::TiledMMA<
                cute::MMA_Atom<MmaAtom>,
                MmaLayout,
                MmaTile>;

            static constexpr bool optimal = true;
        };

        template<typename MmaAtom, bool IsALayoutLeft, bool IsBLayoutLeft, bool IsCLayoutLeft, int M, int N, int K>
        using fp8_mma = optimal_layouts<M, N, K, IsALayoutLeft, IsBLayoutLeft, IsCLayoutLeft, 64, 16, 64, 16,
            cute::Swizzle<2, 4, 3>, cute::Swizzle<2, 4, 3>,
            vectorizing_copy, cute::SM75_U32x4_LDSM_N,
            MmaAtom,
            cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
            cute::Tile<cute::_32, cute::_32, cute::_32>>;

        template<typename MmaAtom, bool IsALayoutLeft, bool IsBLayoutLeft, bool IsCLayoutLeft, int M, int N, int K>
        using smaller_16bit_mma = optimal_layouts<M, N, K, IsALayoutLeft, IsBLayoutLeft, IsCLayoutLeft, 32, 8, 32, 8,
            cute::Swizzle<2, 3, 3>, cute::Swizzle<2, 3, 3>,
            cute::SM75_U16x8_LDSM_T, cute::SM75_U32x4_LDSM_N,
            MmaAtom,
            cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
            cute::Tile<cute::_32, cute::_32, cute::_16>>;

        template<typename MmaAtom, bool IsALayoutLeft, bool IsBLayoutLeft, bool IsCLayoutLeft, int M, int N, int K>
        using bigger_16bit_mma = optimal_layouts<M, N, K, IsALayoutLeft, IsBLayoutLeft, IsCLayoutLeft, 64, 8, 64, 8,
            cute::Swizzle<3, 3, 3>, cute::Swizzle<3, 3, 3>,
            cute::SM75_U16x8_LDSM_T, cute::SM75_U32x4_LDSM_N,
            MmaAtom,
            cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
            cute::Tile<cute::_32, cute::_32, cute::_16>>;

        template<bool IsALayoutLeft, bool IsBLayoutLeft, bool IsCLayoutLeft, int M, int N, int K>
        using tf32_mma = optimal_layouts<M, N, K, IsALayoutLeft, IsBLayoutLeft, IsCLayoutLeft, 32, 8, 32, 8,
            cute::Swizzle<3, 2, 3>, cute::Swizzle<3, 2, 3>,
            vectorizing_copy, cute::SM75_U32x4_LDSM_N,
            cute::SM80_16x8x8_F32TF32TF32F32_TN,
            cute::Layout<cute::Shape<cute::_2,cute::_2,cute::_1>,
                            cute::Stride<cute::_2, cute::_1, cute::_1>>, // 2x2x1 thread group
            cute::Tile<cute::_32, cute::_32, cute::_8>>;

        template<bool IsALayoutLeft, bool IsBLayoutLeft, bool IsCLayoutLeft, int M, int N, int K, typename SwizzleAtomA, typename SwizzleAtomB>
        using fp64_mma_ampere = optimal_layouts<M, N, K, IsALayoutLeft, IsBLayoutLeft, IsCLayoutLeft, 16, 4, 16, 4,
            SwizzleAtomA, SwizzleAtomB,
            vectorizing_copy, vectorizing_copy,
            cute::SM80_8x8x4_F64F64F64F64_TN,
            cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
            cute::Tile<cute::Layout<cute::Shape<cute::_16,cute::_2>,cute::Stride<cute::_2,cute::_1>>,        // 32x32x4 MMA with perm for load vectorization
                cute::Layout<cute::Shape<cute::_16,cute::_2>,cute::Stride<cute::_2,cute::_1>>,
                cute::Underscore>>;

        template<bool IsALayoutLeft, bool IsBLayoutLeft, bool IsCLayoutLeft, int M, int N, int K>
        using fp64_mma_hopper = optimal_layouts<M, N, K, IsALayoutLeft, IsBLayoutLeft, IsCLayoutLeft, 32, 16, 32, 16,
            identity_swizzle, identity_swizzle,
            vectorizing_copy, vectorizing_copy,
            cute::SM90_16x8x16_F64F64F64F64_TN,
            cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>,
            cute::Tile<cute::Underscore, cute::Underscore, cute::Underscore>>;


        // 8 bit suggested layouts
        #if (__CUDACC_VER_MAJOR__ >= 12) && (__CUDACC_VER_MINOR__ >= 4)

        template<int SM, bool IsALayoutLeft, bool IsBLayoutLeft, bool IsCLayoutLeft,
                 int M, int N, int K>
        struct optimal_config<128, SM,
                      float_e4m3_t, IsALayoutLeft, 16,
                      float_e4m3_t, IsBLayoutLeft, 16,
                      float, IsCLayoutLeft, 16,
                      M, N, K,
             suggested_enable_if_t<
                SM, 890, 0, M, N, K, 32, 32, 64,
                IsALayoutLeft, 64, 16, true,
                IsBLayoutLeft, 64, 16, true,
                float_e4m3_t
            >
        > : fp8_mma<cublasdx::detail::SM89_16x8x32_F32E4M3E4M3F32_TN, IsALayoutLeft, IsBLayoutLeft, IsCLayoutLeft, M, N, K>
        {};

        template<int SM, bool IsALayoutLeft, bool IsBLayoutLeft, bool IsCLayoutLeft,
                 int M, int N, int K>
        struct optimal_config<
             128, SM, float_e5m2_t, IsALayoutLeft, 16, float_e4m3_t, IsBLayoutLeft, 16, float, IsCLayoutLeft, 16, M, N, K,
             suggested_enable_if_t<
                SM, 890, 0, M, N, K, 32, 32, 64,
                IsALayoutLeft, 64, 16, true,
                IsBLayoutLeft, 64, 16, true,
                float_e4m3_t
            >
        > : fp8_mma<cublasdx::detail::SM89_16x8x32_F32E5M2E4M3F32_TN, IsALayoutLeft, IsBLayoutLeft, IsCLayoutLeft, M, N, K>
        {};

        template<int SM, bool IsALayoutLeft, bool IsBLayoutLeft, bool IsCLayoutLeft,
                 int M, int N, int K>
        struct optimal_config<
             128, SM, float_e4m3_t, IsALayoutLeft, 16, float_e5m2_t, IsBLayoutLeft, 16, float, IsCLayoutLeft, 16, M, N, K,
             suggested_enable_if_t<
                SM, 890, 0, M, N, K, 32, 32, 64,
                IsALayoutLeft, 64, 16, true,
                IsBLayoutLeft, 64, 16, true,
                float_e4m3_t
            >
        > : fp8_mma<cublasdx::detail::SM89_16x8x32_F32E4M3E5M2F32_TN, IsALayoutLeft, IsBLayoutLeft, IsCLayoutLeft, M, N, K>
        {};

        template<int SM, bool IsALayoutLeft, bool IsBLayoutLeft, bool IsCLayoutLeft,
                 int M, int N, int K>
        struct optimal_config<
             128, SM, float_e5m2_t, IsALayoutLeft, 16, float_e5m2_t, IsBLayoutLeft, 16, float, IsCLayoutLeft, 16, M, N, K,
             suggested_enable_if_t<
                SM, 890, 0, M, N, K, 32, 32, 64,
                IsALayoutLeft, 64, 16, true,
                IsBLayoutLeft, 64, 16, true,
                float_e4m3_t
            >
        > : fp8_mma<cublasdx::detail::SM89_16x8x32_F32E5M2E5M2F32_TN, IsALayoutLeft, IsBLayoutLeft, IsCLayoutLeft, M, N, K>
        {};

        #endif

        // 16 bit suggested layouts
        template<int SM, bool IsALayoutLeft, bool IsBLayoutLeft, bool IsCLayoutLeft,
                 int M, int N, int K>
        struct optimal_config<
             128, SM, cute::half_t, IsALayoutLeft, 16,
                      cute::half_t, IsBLayoutLeft, 16,
                      cute::half_t, IsCLayoutLeft, 16,
                      M, N, K,
             suggested_enable_if_t<
                SM, 800, 0, M, N, K, 32, 32, 16,
                IsALayoutLeft, 32, 8, false,
                IsBLayoutLeft, 32, 8, false,
                cute::half_t
            >
        > : smaller_16bit_mma<cute::SM80_16x8x16_F16F16F16F16_TN, IsALayoutLeft, IsBLayoutLeft, IsCLayoutLeft, M, N, K> { };

        template<int SM, bool IsALayoutLeft, bool IsBLayoutLeft, bool IsCLayoutLeft,
                 int M, int N, int K>
        struct optimal_config<
             128, SM, cute::half_t, IsALayoutLeft, 16,
                      cute::half_t, IsBLayoutLeft, 16,
                      float, IsCLayoutLeft, 16,
                      M, N, K,
             suggested_enable_if_t<
                SM, 800, 0, M, N, K, 32, 32, 16,
                IsALayoutLeft, 32, 8, false,
                IsBLayoutLeft, 32, 8, false,
                cute::half_t
            >
        > : smaller_16bit_mma<cute::SM80_16x8x16_F32F16F16F32_TN, IsALayoutLeft, IsBLayoutLeft, IsCLayoutLeft, M, N, K> { };

        template<int SM, bool IsALayoutLeft, bool IsBLayoutLeft, bool IsCLayoutLeft,
                 int M, int N, int K>
        struct optimal_config<
             128, SM, bfloat16_t, IsALayoutLeft, 16,
                      bfloat16_t, IsBLayoutLeft, 16,
                      float, IsCLayoutLeft, 16,
                      M, N, K,
             suggested_enable_if_t<
                SM, 800, 0, M, N, K, 32, 32, 16,
                IsALayoutLeft, 32, 8, false,
                IsBLayoutLeft, 32, 8, false,
                bfloat16_t
            >
        > : smaller_16bit_mma<cute::SM80_16x8x16_F32BF16BF16F32_TN, IsALayoutLeft, IsBLayoutLeft, IsCLayoutLeft, M, N, K> { };

        template<int SM, bool IsALayoutLeft, bool IsBLayoutLeft, bool IsCLayoutLeft,
                 int M, int N, int K>
        struct optimal_config<
             128, SM, cute::half_t, IsALayoutLeft, 16,
                      cute::half_t, IsBLayoutLeft, 16,
                      cute::half_t, IsCLayoutLeft, 16,
                      M, N, K,
             suggested_enable_if_t<
                SM, 800, 0, M, N, K, 32, 32, 16,
                IsALayoutLeft, 64, 8, true,
                IsBLayoutLeft, 64, 8, true,
                cute::half_t
            >
        > : bigger_16bit_mma<cute::SM80_16x8x16_F16F16F16F16_TN, IsALayoutLeft, IsBLayoutLeft, IsCLayoutLeft, M, N, K> { };


        template<int SM, bool IsALayoutLeft, bool IsBLayoutLeft, bool IsCLayoutLeft,
                 int M, int N, int K>
        struct optimal_config<
             128, SM, cute::half_t, IsALayoutLeft, 16,
                      cute::half_t, IsBLayoutLeft, 16,
                      float, IsCLayoutLeft, 16,
                      M, N, K,
             suggested_enable_if_t<
                SM, 800, 0, M, N, K, 32, 32, 16,
                IsALayoutLeft, 64, 8, true,
                IsBLayoutLeft, 64, 8, true,
                cute::half_t
            >
        > : bigger_16bit_mma<cute::SM80_16x8x16_F32F16F16F32_TN, IsALayoutLeft, IsBLayoutLeft, IsCLayoutLeft, M, N, K> { };

        template<int SM, bool IsALayoutLeft, bool IsBLayoutLeft, bool IsCLayoutLeft,
                 int M, int N, int K>
        struct optimal_config<
             128, SM, bfloat16_t, IsALayoutLeft, 16,
                      bfloat16_t, IsBLayoutLeft, 16,
                      float, IsCLayoutLeft, 16,
                      M, N, K,
             suggested_enable_if_t<
                SM, 800, 0, M, N, K, 32, 32, 16,
                IsALayoutLeft, 64, 8, true,
                IsBLayoutLeft, 64, 8, true,
                bfloat16_t
            >
        > : bigger_16bit_mma<cute::SM80_16x8x16_F32BF16BF16F32_TN, IsALayoutLeft, IsBLayoutLeft, IsCLayoutLeft, M, N, K>
        {};

        // 32 bit suggested layouts
        template<int SM, bool IsALayoutLeft, bool IsBLayoutLeft, bool IsCLayoutLeft,
                 int M, int N, int K>
        struct optimal_config<
             128, SM, tfloat32_t, IsALayoutLeft, 16,
                      tfloat32_t, IsBLayoutLeft, 16,
                      float, IsCLayoutLeft, 16,
                      M, N, K,
             suggested_enable_if_t<
                SM, 800, 0, M, N, K, 32, 32, 32,
                IsALayoutLeft, 32, 8, true,
                IsBLayoutLeft, 32, 8, true,
                cute::tfloat32_t
            >
        > : tf32_mma<IsALayoutLeft, IsBLayoutLeft, IsCLayoutLeft, M, N, K>
        {};

        // 64 bit suggested layouts
        template<int SM, bool IsALayoutLeft, bool IsBLayoutLeft, bool IsCLayoutLeft,
                 int M, int N, int K>
        struct optimal_config<
             128, SM, double, IsALayoutLeft, 16, double, IsBLayoutLeft, 16, double, IsCLayoutLeft, 16, M, N, K,
             suggested_enable_if_t<
                SM, 800, 890, M, N, K, 32, 32, 4,
                IsALayoutLeft, 16, 4, true,
                IsBLayoutLeft, 16, 4, true,
                double
            >
        > : fp64_mma_ampere<IsALayoutLeft, IsBLayoutLeft, IsCLayoutLeft, M, N, K,
            COMMONDX_STL_NAMESPACE::conditional_t<IsALayoutLeft, cute::Swizzle<2, 2, 2>, cute::Swizzle<2, 0, 4>>,
            COMMONDX_STL_NAMESPACE::conditional_t<IsBLayoutLeft, cute::Swizzle<2, 0, 4>, cute::Swizzle<2, 2, 2>>>
        {};

        template<int SM, bool IsALayoutLeft, bool IsBLayoutLeft, bool IsCLayoutLeft,
                 int M, int N, int K>
        struct optimal_config<
             128, SM, double, IsALayoutLeft, 16, double, IsBLayoutLeft, 16, double, IsCLayoutLeft, 16, M, N, K,
             suggested_enable_if_t<
                SM, 900, 0, M, N, K, 32, 16, 16,
                IsALayoutLeft, 16, 4, true,
                IsBLayoutLeft, 16, 4, true,
                double
            >
        > : fp64_mma_hopper<IsALayoutLeft, IsBLayoutLeft, IsCLayoutLeft, M, N, K>
        {};

        template<matrix Matrix, bool HasBlockDim, int BlockDim, int SM,
                 typename TA, bool IsALayoutLeft, int AlignmentA,
                 typename TB, bool IsBLayoutLeft, int AlignmentB,
                 typename TC, bool IsCLayoutLeft, int AlignmentC,
                 int M, int N, int K>
        __host__ __device__ __forceinline__
        constexpr auto get_optimal_layout() {
            using config = optimal_config<HasBlockDim ? BlockDim : 128, SM,
                                          TA, IsALayoutLeft, AlignmentA,
                                          TB, IsBLayoutLeft, AlignmentB,
                                          TC, IsCLayoutLeft, AlignmentC,
                                          M, N, K>;
            return cublasdx::detail::choose<Matrix>(typename config::a_layout{}, typename config::b_layout{}, typename config::c_layout{});
        }

        }
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_SUGGESTED_LAYOUTS_HPP
