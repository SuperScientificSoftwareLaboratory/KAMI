// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_DB_HPP
#define CUBLASDX_DATABASE_CUTE_DB_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/stl/tuple.hpp"

#include "commondx/device_info.hpp"
#include "commondx/type_list.hpp"

#include "cute_tensor.hpp"
#include "cute_tensor_configs.hpp"
#include "configs.hpp"

namespace cublasdx {
    namespace detail {
        namespace cute_backend {

// Find lower integer bound on square root of argument
// such that
// v = lower_int_sqrt(x)
// v * v <= x
// (v + 1) * (v + 1) >= x
constexpr int lower_int_sqrt(int v) {
    for(int i = 1; i < v; ++i) {
        if(i >= (v / i)) {
            return (v / i);
        }
    }

    return 1;
}

// Selects generated_config from commondx::type_list based on blockdim,
// if there is no such implementation in list search_by_blockdim::type is set to void.
template<int ThreadsAvailable, typename ImplementationList>
struct search_by_blockdim;

template<int ThreadsAvailable, typename GeneratedConfig>
struct search_by_blockdim<ThreadsAvailable, commondx::type_list<GeneratedConfig>> {
    using type = COMMONDX_STL_NAMESPACE::conditional_t<GeneratedConfig::blockdim == ThreadsAvailable, GeneratedConfig, void>;
};

template<int ThreadsAvailable, typename GeneratedConfig, typename... RemainingConfigs>
struct search_by_blockdim<ThreadsAvailable, commondx::type_list<GeneratedConfig, RemainingConfigs...>> {
    using type = COMMONDX_STL_NAMESPACE::conditional_t<
        GeneratedConfig::blockdim == ThreadsAvailable,
        GeneratedConfig,
        typename search_by_blockdim<ThreadsAvailable, commondx::type_list<RemainingConfigs...>>::type>;
};

template<int N>
constexpr int closest_multiple_of(int value) {
    return ((value + (N - 1)) / N) * N;
}

constexpr bool is_quadpair_mma(mma_atom atom) {
    return atom == mma_atom::SM70_8x8x4_F16F16F16F16_TN or
           atom == mma_atom::SM70_8x8x4_C16C16C16C16_TN_CUBLASDX or
           atom == mma_atom::SM70_8x8x4_F32F16F16F32_TN or
           atom == mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX;
}

template<mma_atom atom, unsigned int SM>
constexpr mma_atom apply_mma_workarounds([[maybe_unused]] const int m, [[maybe_unused]] const int n, [[maybe_unused]] const int k, [[maybe_unused]] const int blockdim) {
    if constexpr (atom == mma_atom::SM80_16x8x8_C32TC32TC32C32_TN_CUBLASDX) {
        // c32tc32tc32c32 mma:
        // * for CUDA prior than 12.5 update 1, use SM80_16x8x8_C32TC32TC32C32_TN_CUBLASDX only when all the m, n, k that are multiples of 16.
        //   otherwise, use universal fma.
    #if (__CUDACC_VER_MAJOR__ <= 11 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ <= 4) || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ == 5 && __CUDACC_VER_BUILD__ <= 40))
        return ((m % 16 == 0) && (n % 16 == 0) && (k % 16 == 0)) ? atom : mma_atom::universal_fma;
    #endif
    } else if constexpr (atom == mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX) {
        // c32c16c16c32 mma:
        // * for CUDA prior than 12.4, skip using SM70_8x8x4_C32C16C16C32_TN_CUBLASDX and always use universal fma
        // * only supports on SM70 and SM72
        static_assert(SM == 700 || SM == 720, "SM70_8x8x4_C32C16C16C32_TN_CUBLASDX only supports on SM70 or SM72");

    #if (__CUDACC_VER_MAJOR__ <= 11 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ <= 3))
        return mma_atom::universal_fma ;
    #else
        return ((m % 16 == 0) && (n % 16 == 0) && (k % 16 == 0) && (blockdim >= 64)) ? atom : mma_atom::universal_fma;
    #endif
    } else if constexpr (atom == mma_atom::SM70_8x8x4_F32F16F16F32_TN) {
        // f32f16f16f32 mma: only supports on SM70 and SM72
        static_assert(SM == 700 || SM == 720, "SM70_8x8x4_F32F16F16F32_TN only supports on SM70 or SM72");

        return ((m % 16 == 0) && (n % 16 == 0) && (k % 16 == 0) && (blockdim >= 64)) ? atom : mma_atom::universal_fma;
    } else if constexpr (atom == mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX or
                         atom == mma_atom::SM89_16x8x32_F32E4M3E5M2F32_TN_CUBLASDX or
                         atom == mma_atom::SM89_16x8x32_F32E4M3E4M3F32_TN_CUBLASDX or
                         atom == mma_atom::SM89_16x8x32_F32E5M2E4M3F32_TN_CUBLASDX or
                         atom == mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX or
                         atom == mma_atom::SM89_16x8x32_C32CE4M3CE5M2C32_TN_CUBLASDX or
                         atom == mma_atom::SM89_16x8x32_C32CE4M3CE4M3C32_TN_CUBLASDX or
                         atom == mma_atom::SM89_16x8x32_C32CE5M2CE4M3C32_TN_CUBLASDX) {
    #if (__CUDACC_VER_MAJOR__ <= 11 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ <= 3))
        return mma_atom::universal_fma;
    #else
        return atom;
    #endif
    }

    return atom;
}

template<bool HasBlockDim, int ThreadsAvailable, typename ConfigList>
using search_struct = COMMONDX_STL_NAMESPACE::conditional_t<HasBlockDim,
                                         search_by_blockdim<ThreadsAvailable, ConfigList>,
                                         commondx::type_list_element<0, ConfigList>>;

// Some MMAs mirror each other (such as FP32FP16FP16FP32 and FP32BF16BF16FP32)
// and they are all redirected to one matching config in the database
template<typename TA, typename TB, typename TC, typename SM, typename EnableIfHelper = void>
struct database_mapper {
   using a_type = TA;
   using b_type = TB;
   using c_type = TC;

   static constexpr bool decayed = false;
};

// Use FP16 config for FP32BF16 compute
template<typename SM>
struct database_mapper<bfloat16_t, bfloat16_t, float, SM,
    COMMONDX_STL_NAMESPACE::enable_if_t<
        SM::value >= 800
    >> {
   using a_type = half_t;
   using b_type = half_t;
   using c_type = float;

   static constexpr bool decayed = true;
};

// Use E5M2 config for E4M3 compute
template<typename TA, typename TB, typename SM>
struct database_mapper<TA, TB, float, SM,
    COMMONDX_STL_NAMESPACE::enable_if_t<
        COMMONDX_STL_NAMESPACE::is_same_v<TA, float_e4m3_t>  or
        COMMONDX_STL_NAMESPACE::is_same_v<TB, float_e4m3_t>
    >> {
   using a_type =
    COMMONDX_STL_NAMESPACE::conditional_t<
        COMMONDX_STL_NAMESPACE::is_same_v<TA, float_e4m3_t>,
        float_e5m2_t,
        TA
    >;
   using b_type =
    COMMONDX_STL_NAMESPACE::conditional_t<
        COMMONDX_STL_NAMESPACE::is_same_v<TB, float_e4m3_t>,
        float_e5m2_t,
        TB
    >;
   using c_type = float;

   static constexpr bool decayed = true;
};

template<typename ... Ts>
using database_mapper_a_t = typename database_mapper<Ts...>::a_type;

template<typename ... Ts>
using database_mapper_b_t = typename database_mapper<Ts...>::b_type;

template<typename ... Ts>
using database_mapper_c_t = typename database_mapper<Ts...>::c_type;

template<typename ... Ts>
constexpr bool database_mapper_v = database_mapper<Ts...>::decayed;

template<typename TA, typename TB, typename TC, typename SM>
struct database_mapper<TA, TB, TC, SM,
    COMMONDX_STL_NAMESPACE::enable_if_t<
        cutlass::is_complex<TA>::value &&
        cutlass::is_complex<TB>::value &&
        cutlass::is_complex<TC>::value
    >> {

   using a_value_t = typename TA::value_type;
   using b_value_t = typename TB::value_type;
   using c_value_t = typename TC::value_type;

   using value_type_decay = database_mapper<a_value_t, b_value_t, c_value_t, SM>;

   using a_type = cutlass::complex<typename value_type_decay::a_type>;
   using b_type = cutlass::complex<typename value_type_decay::b_type>;
   using c_type = cutlass::complex<typename value_type_decay::c_type>;

   static constexpr bool decayed = value_type_decay::decayed;
};

// Since the database for e.g. BF16 is checked with FP16 the atom
// needs to be switched afterwards to a matching BF16 one
template<typename AType, typename BType, typename CType, typename SM>
constexpr mma_atom map_mma_from_database(mma_atom atom) {
    if (database_mapper_v<AType, BType, CType, SM>) {
        switch(atom) {
            case mma_atom::SM75_16x8x8_F32F16F16F32_TN_CUBLASDX:
                return mma_atom::SM80_16x8x8_F32BF16BF16F32_TN;
            case mma_atom::SM75_16x8x8_C32C16C16C32_TN_CUBLASDX:
                return mma_atom::SM80_16x8x8_C32BC16BC16C32_TN_CUBLASDX;
            case mma_atom::SM80_16x8x16_F32F16F16F32_TN:
                return mma_atom::SM80_16x8x16_F32BF16BF16F32_TN;
            case mma_atom::SM80_16x8x16_C32C16C16C32_TN_CUBLASDX:
                return mma_atom::SM80_16x8x16_C32BC16BC16C32_TN_CUBLASDX;
            case mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX: {
                if(COMMONDX_STL_NAMESPACE::is_same_v<AType, float_e4m3_t> &&
                   COMMONDX_STL_NAMESPACE::is_same_v<BType, float_e4m3_t>) {
                    return mma_atom::SM89_16x8x32_F32E4M3E4M3F32_TN_CUBLASDX;
                } else if(COMMONDX_STL_NAMESPACE::is_same_v<AType, float_e4m3_t> &&
                   COMMONDX_STL_NAMESPACE::is_same_v<BType, float_e5m2_t>) {
                    return mma_atom::SM89_16x8x32_F32E4M3E5M2F32_TN_CUBLASDX;
                } else if(COMMONDX_STL_NAMESPACE::is_same_v<AType, float_e5m2_t> &&
                   COMMONDX_STL_NAMESPACE::is_same_v<BType, float_e4m3_t>) {
                    return mma_atom::SM89_16x8x32_F32E5M2E4M3F32_TN_CUBLASDX;
                }
            }
            case mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX: {
                using complex_e5m2 = cutlass::complex<float_e5m2_t>;
                using complex_e4m3 = cutlass::complex<float_e4m3_t>;

                if(COMMONDX_STL_NAMESPACE::is_same_v<AType, complex_e4m3> &&
                COMMONDX_STL_NAMESPACE::is_same_v<BType, complex_e4m3>) {
                    return mma_atom::SM89_16x8x32_C32CE4M3CE4M3C32_TN_CUBLASDX;
                } else if(COMMONDX_STL_NAMESPACE::is_same_v<AType, complex_e4m3> &&
                COMMONDX_STL_NAMESPACE::is_same_v<BType, complex_e5m2>) {
                    return mma_atom::SM89_16x8x32_C32CE4M3CE5M2C32_TN_CUBLASDX;
                } else if(COMMONDX_STL_NAMESPACE::is_same_v<AType, complex_e5m2> &&
                COMMONDX_STL_NAMESPACE::is_same_v<BType, complex_e4m3>) {
                    return mma_atom::SM89_16x8x32_C32CE5M2CE4M3C32_TN_CUBLASDX;
                }
            }
        }
    }

    return atom;
}

// RETURN TYE STD::TUPLE CONTAINS 3 ELEMENTS:
// 1. MMA_ATOM, Enum value pointing to which atom should be used
// 2. INT, TileX size
// 3. INT, TileY size
// Types passed as AType, BType, CType are CUTLASS types
// e.g. cute::half_t and not __half
template<typename AType,
         typename BType,
         typename CType,
         int M,
         int N,
         int K,
         int ThreadsAvailable,
         bool HasBlockDim,
         typename SM>
constexpr auto get_tile_config() {
    #ifdef __NVCOMPILER
    #pragma diag_suppress 185
    #pragma diag_suppress 111
    #endif

    // Decay datatypes to ones supported by the database
    using atype_db = database_mapper_a_t<AType, BType, CType, SM>;
    using btype_db = database_mapper_b_t<AType, BType, CType, SM>;
    using ctype_db = database_mapper_c_t<AType, BType, CType, SM>;

    if constexpr(M == 1 or N == 1 or K == 1) {
        constexpr int fallback_threads  = HasBlockDim ? ThreadsAvailable : cute::max(32, cute::min(1024, closest_multiple_of<32>(M * N / 4)));
        return COMMONDX_STL_NAMESPACE::make_tuple(mma_atom::universal_fma, 1, fallback_threads);
    }

    // Lambda which checks if record from entry list matches and is usable
    // in necessary circumstances
    const auto database_lookup = [](auto query) constexpr {
        if constexpr (decltype(query)::defined) {
            using entry = typename search_struct<HasBlockDim, ThreadsAvailable, typename decltype(query)::list>::type;
            if constexpr (!COMMONDX_STL_NAMESPACE::is_same_v<entry, void>) {
                if (entry::mma == apply_mma_workarounds<entry::mma, SM::value>(M, N, K, entry::blockdim)) {
                    constexpr auto inner_tiles = entry::tiles;
                    return COMMONDX_STL_NAMESPACE::make_tuple(true,
                        COMMONDX_STL_NAMESPACE::make_tuple(
                            map_mma_from_database<AType, BType, CType, SM>(entry::mma),
                            COMMONDX_STL_NAMESPACE::get<0>(inner_tiles), COMMONDX_STL_NAMESPACE::get<1>(inner_tiles)));
                }
            }
        }

        return COMMONDX_STL_NAMESPACE::make_tuple(false, COMMONDX_STL_NAMESPACE::make_tuple(mma_atom::universal_fma, 0, 0));
    };

    // To take both M and N under account use the square root of their product
    // as database entry size (database only has square records inside itself)
    constexpr int averaged_size = lower_int_sqrt(M * N);
    using mmm_query = database::generated_config<atype_db, btype_db, ctype_db, averaged_size, averaged_size, averaged_size, SM>;

    // No rounding
    auto mmm_entry = database_lookup(mmm_query{});
    if(COMMONDX_STL_NAMESPACE::get<0>(mmm_entry)) {
        return COMMONDX_STL_NAMESPACE::get<1>(mmm_entry);
    }

    // Rounding to 4 / 8 / (16 * n)
    constexpr int rounded = (M < 4) ? 4 : (M < 8) ? 8 : closest_multiple_of<16>(averaged_size);
    using rounded_mmm_query =
        database::generated_config<atype_db, btype_db, ctype_db, rounded, rounded, rounded, SM>;
    auto rounded_mmm_entry = database_lookup(rounded_mmm_query{});
    if(COMMONDX_STL_NAMESPACE::get<0>(rounded_mmm_entry)) {
        return COMMONDX_STL_NAMESPACE::get<1>(rounded_mmm_entry);
    }

    // If no record present in the database try finding a matching config with the heuristic
    constexpr auto default_mma        = get_default_mma<AType, BType, CType, SM::value>();
    using default_mma_t               = decltype(convert_mma_atom_to_cute<AType, BType, CType, default_mma>());
    constexpr int default_mma_thread_atom = decltype(cute::size(typename cute::MMA_Traits<default_mma_t>::ThrID {}))::value;
    constexpr int atom_c_size = cute::max(decltype(cute::size(typename cute::MMA_Traits<default_mma_t>::CLayout{}))::value, 4);
    constexpr int backup_threads = ((M * N) / atom_c_size) * default_mma_thread_atom;
    constexpr int heuristic_threads  = HasBlockDim ? ThreadsAvailable : cute::min(1024, cute::max(32, backup_threads));

    // If there's no chance matching to default_mma fallback to fma
    constexpr auto selected_mma_before_workarounds =
        (heuristic_threads % default_mma_thread_atom == 0) ? default_mma : mma_atom::universal_fma;
    // Apply workarounds
    constexpr auto selected_mma = apply_mma_workarounds<selected_mma_before_workarounds, SM::value>(M, N, K, heuristic_threads);
    using selected_mma_t  = decltype(convert_mma_atom_to_cute<AType, BType, CType, selected_mma>());
    constexpr int thread_atom = decltype(cute::size(typename cute::MMA_Traits<selected_mma_t>::ThrID {}))::value;

    const auto size_tile_x = decltype(cute::size<0>(typename cute::MMA_Traits<selected_mma_t>::Shape_MNK {}))::value;
    const auto size_tile_y = decltype(cute::size<1>(typename cute::MMA_Traits<selected_mma_t>::Shape_MNK {}))::value;

    // Because Quadpair MMAs use 8 threads per unit but 4 units are necessary, divide the initial number of
    // units by 4 and then multiply tile_x and tile_y by 2 respectively
    constexpr int units_to_divide = (heuristic_threads / thread_atom) / (is_quadpair_mma(selected_mma) ? 8 : 1);

    constexpr auto size_x = is_quadpair_mma(selected_mma) ? M / 4 : M;
    constexpr auto size_y = is_quadpair_mma(selected_mma) ? N / 2 : N;

    for (int tile_x = 1; tile_x <= units_to_divide; ++tile_x) {
        int tile_y = units_to_divide / tile_x;
        // This loop looks for such a division of ThreadsAvailable into TileX
        // and TileY that ThreadsAvailable == TileX * TileY and TileX > TileY
        // keeping TileX as small as possible. Ideally it will find square root
        // of ThreadsAvailable, but otherwise the most square combination
        // possible
        if (tile_x >= tile_y && tile_x * tile_y == units_to_divide
            && (tile_x * tile_y * thread_atom) >= (32 / (is_quadpair_mma(selected_mma) ? 8 : 1))
            && ((size_x > (size_tile_x * tile_x)) ||
                ((size_tile_x * tile_x > size_x) && (size_tile_x * tile_x - size_x < size_tile_x)))
            && ((size_y > (size_tile_y * tile_y)) ||
                ((size_tile_y * tile_y > size_y) && (size_tile_y * tile_y - size_y < size_tile_y)))) {
            // Handle quadpairs
            if constexpr (is_quadpair_mma(selected_mma)) {
                return COMMONDX_STL_NAMESPACE::make_tuple(selected_mma, 4 * tile_x, 2 * tile_y);
            } else {
                return COMMONDX_STL_NAMESPACE::make_tuple(selected_mma, tile_x, tile_y);
            }
        }
    }
    #ifdef __NVCOMPILER
    #pragma diag_warning 185
    #pragma diag_warning 111
    #endif

    // Final fallback
    return COMMONDX_STL_NAMESPACE::make_tuple(mma_atom::universal_fma, 1, heuristic_threads);
}

// Types passed here are CUTLASS types
template<typename AType,
         typename BType,
         typename CType,
         int M,
         int N,
         int K,
         int  ThreadsAvailable,
         bool HasBlockSize,
         typename SM>
struct generated_config_getter {
    static constexpr auto tile_config =
        get_tile_config<AType, BType, CType, M, N, K, ThreadsAvailable, HasBlockSize, SM>();
    using config = decltype(cute::make_tiled_mma(
        convert_mma_atom_to_cute<AType, BType, CType, COMMONDX_STL_NAMESPACE::get<0>(tile_config)>(),
        cute::Layout<cute::Shape<cute::Int<COMMONDX_STL_NAMESPACE::get<1>(tile_config)>, cute::Int<COMMONDX_STL_NAMESPACE::get<2>(tile_config)>>> {}));
};

// Types passed here are CUTLASS types
template<typename AType,
         typename BType,
         typename CType,
         int M,
         int N,
         int K,
         int  ThreadsAvailable,
         bool HasBlockSize,
         typename SM,
         bool     Benchmark = false, // Overrides database/heuristic parameters with (MmaAtom, TileX, TileY)
         mma_atom MmaAtom   = mma_atom::universal_fma, // Override
         int      TileX     = 0, // Override
         int      TileY     = 0> // Override
struct cute_config {
    using manual    = decltype(cute::make_tiled_mma(convert_mma_atom_to_cute<AType, BType, CType, MmaAtom>(),
                                                 cute::Layout<cute::Shape<cute::Int<TileX>, cute::Int<TileY>>> {}));
    using generated = typename generated_config_getter<AType,
                                                       BType,
                                                       CType,
                                                       M,
                                                       N,
                                                       K,
                                                       ThreadsAvailable,
                                                       HasBlockSize,
                                                       SM>::config;

    using config = COMMONDX_STL_NAMESPACE::conditional_t<Benchmark, manual, generated>;
    static_assert(!HasBlockSize || cute::size(config {}) <= ThreadsAvailable, "FMA CuTe tile sizes are improperly defined");
};

        } // namespace cute_backend
    } // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_CUTE_DB_HPP
