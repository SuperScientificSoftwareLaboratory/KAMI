// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_BLAS_EXECUTION_HPP
#define CUBLASDX_DETAIL_BLAS_EXECUTION_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/stl/tuple.hpp"

#include "blas_description.hpp"
#include "tensor.hpp"
#include "database/cute.hpp"
#include "database/suggested_layouts.hpp"

#include "../traits.hpp"

namespace cublasdx {
    namespace detail {

        template<typename ... TypeTuples>
        inline static constexpr bool __host__ __device__
        are_types_compatible_impl() {
            const auto check_if_compatible = [](const auto TT) constexpr {
                using TA = decltype(COMMONDX_STL_NAMESPACE::get<0>(TT));
                using TB = decltype(COMMONDX_STL_NAMESPACE::get<1>(TT));

                return (not COMMONDX_STL_NAMESPACE::is_void_v<TA>)
                    && (not COMMONDX_STL_NAMESPACE::is_void_v<TB>)
                    && (sizeof(TA)  == sizeof(TB))
                    && (alignof(TA) == alignof(TB));
            };

            return ((check_if_compatible(TypeTuples{}) && ...));
        }

        template<class value_type, unsigned a_size, unsigned b_size, unsigned c_size, unsigned a_alignment, unsigned b_alignment, unsigned c_alignment>
        struct smem_alignment {
            alignas(a_alignment) typename value_type::a_type a[a_size];
            alignas(b_alignment) typename value_type::b_type b[b_size];
            alignas(c_alignment) typename value_type::c_type c[c_size];
        };

        template<class value_type, unsigned a_alignment, unsigned b_alignment, unsigned c_alignment>
        struct smem_alignment_dyn {
            __device__ __host__ smem_alignment_dyn(const unsigned m,
                                                   const unsigned n,
                                                   const unsigned k,
                                                   const arrangement a_arr,
                                                   const arrangement b_arr,
                                                   const arrangement c_arr,
                                                   const unsigned int lda,
                                                   const unsigned int ldb,
                                                   const unsigned int ldc)
                : smem_alignment_dyn(calculate_matrix_size(lda, m, k, a_arr), calculate_matrix_size(ldb, k, n, b_arr), calculate_matrix_size(ldc, m, n, c_arr)){}

            __device__ __host__ smem_alignment_dyn(const unsigned a_size,
                                                   const unsigned b_size,
                                                   const unsigned c_size) {
                a_offset = 0;
                b_offset = detail::aligned_dynamic_size<b_alignment>(a_offset + a_size * sizeof(typename value_type::a_type));
                c_offset = detail::aligned_dynamic_size<c_alignment>(b_offset + b_size * sizeof(typename value_type::b_type));
                memory_usage = c_offset + c_size * sizeof(typename value_type::c_type) - a_offset;
            }
            __forceinline__ __device__ __host__ unsigned int get_a_offset() const { return a_offset; }
            __forceinline__ __device__ __host__ unsigned int get_b_offset() const { return b_offset; }
            __forceinline__ __device__ __host__ unsigned int get_c_offset() const { return c_offset; }
            __forceinline__ __device__ __host__ unsigned int get_memory_usage() const { return memory_usage; };
        private:
            unsigned int a_offset;
            unsigned int b_offset;
            unsigned int c_offset;
            unsigned int memory_usage;
        };

        template<class... Operators>
        class blas_execution: public blas_description<Operators...>, public commondx::detail::execution_description_expression
        {
            using base_type = blas_description<Operators...>;
            using this_type = blas_execution<Operators...>;

        protected:
            // Precision type
            using typename base_type::this_blas_precision;

            /// ---- Constraints

            // We need Block operator to be specified exactly once
            static constexpr bool has_one_block = has_at_most_one_of<operator_type::block, this_type>::value;
            static_assert(has_one_block, "Can't create blas function with two execution operators");
        };

        template<class value_type>
        struct cutlass_value_type {
            using a_type = convert_to_cutlass_type_t<typename value_type::a_type>;
            using b_type = convert_to_cutlass_type_t<typename value_type::b_type>;
            using c_type = convert_to_cutlass_type_t<typename value_type::c_type>;
        };

        template<class trans_op, class value_type, class cutlass_value_type = convert_to_cutlass_type_t<value_type>>
        struct transform_op_wrapper {
            trans_op op;
            __device__ __host__
            cutlass_value_type operator()(const cutlass_value_type arg) const {
                if constexpr (COMMONDX_STL_NAMESPACE::is_invocable_v<trans_op, cutlass_value_type> &&
                              COMMONDX_STL_NAMESPACE::is_convertible_v<COMMONDX_STL_NAMESPACE::decay_t<COMMONDX_STL_NAMESPACE::invoke_result_t<trans_op, cutlass_value_type>>, cutlass_value_type>) {
                    return op(arg);
                } else {
                    static_assert(COMMONDX_STL_NAMESPACE::is_convertible_v<COMMONDX_STL_NAMESPACE::decay_t<COMMONDX_STL_NAMESPACE::invoke_result_t<trans_op, value_type>>, value_type>,
                    "Functor must accept value of type value_type and return value convertible to type value_type");

                    auto result = static_cast<value_type>(op(cast_from_cutlass_type<value_type>(arg)));
                    return cast_to_cutlass_type<cutlass_value_type>(result);
                }
            }
        };

#if defined(__CUDACC_RTC__) || defined(_MSC_VER)
        template<class... Operators>
        class blas_block_execution_partial: public blas_execution<Operators...>
        {
            using base_type = blas_execution<Operators...>;
            using typename base_type::this_blas_precision;
            using this_blas_value_type = map_value_type<base_type::this_blas_type_v, this_blas_precision>;

        public:
            using a_value_type  = typename this_blas_value_type::a_type;
            using b_value_type  = typename this_blas_value_type::b_type;
            using c_value_type  = typename this_blas_value_type::c_type;
        };
#endif

        template<class... Operators>
        class blas_block_execution: public blas_execution<Operators...>
        {
            using this_type = blas_block_execution<Operators...>;
            using base_type = blas_execution<Operators...>;

            // Import precision type from base class
            using typename base_type::this_blas_precision;

            // Value type
            using this_blas_value_type = map_value_type<base_type::this_blas_type_v, this_blas_precision>;
            using this_cutlass_value_type = cutlass_value_type<this_blas_value_type>;

            /// ---- Suggestions
            using execution_suggestions =
                cute_backend::execution_suggestions<typename this_cutlass_value_type::a_type,
                                                    typename this_cutlass_value_type::b_type,
                                                    typename this_cutlass_value_type::c_type,
                                                    typename base_type::this_blas_alignment,
                                                    base_type::this_blas_size_m_v,
                                                    base_type::this_blas_size_n_v,
                                                    base_type::this_blas_size_k_v,
                                                    typename base_type::this_blas_arrangement,
                                                    typename base_type::this_blas_transpose_mode,
                                                    typename base_type::this_blas_sm>;

            /// ---- Traits

            // Block Dimension
            // * Default value: selected by implementation
            static constexpr bool has_block_dim = has_operator<operator_type::block_dim, base_type>::value;
            using default_blas_block_dim        = typename execution_suggestions::block_dim;
            using this_blas_block_dim           = get_or_default_t<operator_type::block_dim, base_type, default_blas_block_dim>;
            static constexpr auto this_blas_block_dim_v = this_blas_block_dim::value;

            static constexpr bool has_ld = has_operator<operator_type::ld, base_type>::value;

            /// ---- Checks

            static constexpr bool valid_block_dim = this_blas_block_dim::flat_size >= 32 && this_blas_block_dim::flat_size <= 1024;
            static_assert(valid_block_dim,
                          "Provided block dimension is invalid, BlockDim<> must have at least 32 threads, and can't "
                          "have more than 1024 threads.");

            /// ---- Backend

            // CuTe backend implementation
            using exec_t = cute_backend::execution<typename this_cutlass_value_type::a_type,
                                                   typename this_cutlass_value_type::b_type,
                                                   typename this_cutlass_value_type::c_type,
                                                   typename base_type::this_blas_alignment,
                                                   base_type::this_blas_size_m_v,
                                                   base_type::this_blas_size_n_v,
                                                   base_type::this_blas_size_k_v,
                                                   typename base_type::this_blas_arrangement,
                                                   typename base_type::this_blas_transpose_mode,
                                                   typename base_type::this_blas_sm,
                                                   this_blas_block_dim,
                                                   this_type::has_block_dim>;

            inline static constexpr dim3 get_suggested_block_dim() {
                static_assert(base_type::is_complete, "Can't provide suggested block dimensions, description is not complete");
                return execution_suggestions::block_dim::value;
            }

            inline static constexpr dim3 get_block_dim() {
                static_assert(base_type::is_complete, "Can't provide block dimensions, description is not complete");
                if constexpr(has_block_dim) {
                    return this_blas_block_dim_v;
                }
                return get_suggested_block_dim();
            }

            template<typename TA, typename TB, typename TC>
            static constexpr bool are_types_compatible() {
                return detail::are_types_compatible_impl<
                    COMMONDX_STL_NAMESPACE::tuple<a_value_type, TA>,
                    COMMONDX_STL_NAMESPACE::tuple<b_value_type, TB>,
                    COMMONDX_STL_NAMESPACE::tuple<c_value_type, TC>
                >();
            }

            template<typename Alpha, typename TA, typename TB, typename Beta, typename TC>
            static constexpr bool are_types_compatible() {
                return detail::are_types_compatible_impl<
                    COMMONDX_STL_NAMESPACE::tuple<c_value_type, Alpha>,
                    COMMONDX_STL_NAMESPACE::tuple<a_value_type, TA>,
                    COMMONDX_STL_NAMESPACE::tuple<b_value_type, TB>,
                    COMMONDX_STL_NAMESPACE::tuple<c_value_type, Beta>,
                    COMMONDX_STL_NAMESPACE::tuple<c_value_type, TC>
                >();
            }

            template<typename ... Ts>
            using execute_enable_if_t = COMMONDX_STL_NAMESPACE::enable_if_t<
                (are_types_compatible<Ts...>())>;

            template<typename ... Ts>
            using execute_disable_if_t = COMMONDX_STL_NAMESPACE::enable_if_t<
                not (are_types_compatible<Ts...>())>;

            template<typename MemOp, typename Engine>
            static constexpr bool is_functor_compatible(MemOp, Engine) {
                using value_type = typename Engine::value_type;
                return COMMONDX_STL_NAMESPACE::is_convertible_v<
                            COMMONDX_STL_NAMESPACE::decay_t<
                                COMMONDX_STL_NAMESPACE::invoke_result_t<
                                    MemOp, value_type>>,
                        value_type>;
            }


            template<class ALayout, class BLayout, class CLayout>
            static constexpr bool are_layout_acceptable() {
                return cute::is_layout<ALayout>::value &&
                       cute::is_layout<BLayout>::value &&
                       cute::is_layout<CLayout>::value;
            }

            template<arrangement Arr, typename SizeA, typename SizeB, typename LD>
            __forceinline__ __host__ __device__ constexpr static auto get_layout_gmem_impl(SizeA sa, SizeB sb, LD ld) {
                static_assert(cute::is_integral<SizeA>::value && cute::is_integral<SizeB>::value &&
                              cute::is_integral<LD>::value);
                return cute_backend::make_layout_from_arrangement<Arr>(sa, sb, ld);
            }

            template<arrangement Arr, typename SizeA, typename SizeB, typename LD>
            __forceinline__ __host__ __device__ constexpr static auto
            get_layout_smem_impl(SizeA sa, SizeB sb, LD ld) {
                return get_layout_gmem_impl<Arr>(sa, sb, ld);
            }

            template<matrix M>
            __forceinline__ __host__ __device__ constexpr static auto get_layout_gmem() {
                using rows                = cute::Int<M == matrix::B ? exec_t::k : exec_t::m>; // A - m, B - k, C - m
                using columns             = cute::Int<M == matrix::A ? exec_t::k : exec_t::n>; // A - k, B - n,  C - n
                constexpr arrangement arr = choose<M>(base_type::this_blas_arrangement_a,
                                                      base_type::this_blas_arrangement_b,
                                                      base_type::this_blas_arrangement_c);
                constexpr int         ld  = (arr == col_major) ? choose<M>(exec_t::m, exec_t::k, exec_t::m) : choose<M>(exec_t::k, exec_t::n, exec_t::n);
                return pointer_layout {gmem_tag {}, get_layout_gmem_impl<arr>(rows {}, columns {}, cute::Int<ld> {})};
            }

            template<matrix M>
            __forceinline__ __host__ __device__ constexpr static auto get_layout_gmem(const unsigned int ld) {
                using rows                = cute::Int<M == matrix::B ? exec_t::k : exec_t::m>;
                using columns             = cute::Int<M == matrix::A ? exec_t::k : exec_t::n>;
                constexpr arrangement arr = choose<M>(base_type::this_blas_arrangement_a,
                                                      base_type::this_blas_arrangement_b,
                                                      base_type::this_blas_arrangement_c);
                return pointer_layout {gmem_tag {}, get_layout_gmem_impl<arr>(rows {}, columns {}, ld)};
            }

            template<matrix M, typename IntegralType, IntegralType LD>
            __forceinline__ __host__ __device__ constexpr static auto get_layout_gmem(const COMMONDX_STL_NAMESPACE::integral_constant<IntegralType, LD> ld) {
                static_assert(cute::is_integral<IntegralType>::value, "Compile time leading dimension must be of integral type");
                using rows                = cute::Int<M == matrix::B ? exec_t::k : exec_t::m>;
                using columns             = cute::Int<M == matrix::A ? exec_t::k : exec_t::n>;
                constexpr arrangement arr = choose<M>(base_type::this_blas_arrangement_a,
                                                      base_type::this_blas_arrangement_b,
                                                      base_type::this_blas_arrangement_c);

                constexpr bool valid_ld =
                    (LD >= ((arr == arrangement::col_major) ? rows::value : columns::value));

                static_assert(valid_ld || (M != matrix::A),
                    "Incorrect leading dimension for A matrix, LDA must be greater than its leading size");
                static_assert(valid_ld || (M != matrix::B),
                    "Incorrect leading dimension for B matrix, LDB must be greater than its leading size");
                static_assert(valid_ld || (M != matrix::C),
                    "Incorrect leading dimension for C matrix, LDC must be greater than its leading size");

                return pointer_layout {gmem_tag {}, get_layout_gmem_impl<arr>(rows {}, columns {}, cute::Int<LD> {})};
            }

            // Get shared memory layout
            template<matrix M>
            __forceinline__ __host__ __device__ constexpr static auto get_layout_smem() {
                using rows                = cute::Int<M == matrix::B ? exec_t::k : exec_t::m>;
                using columns             = cute::Int<M == matrix::A ? exec_t::k : exec_t::n>;
                constexpr int         ld  = choose<M>(lda, ldb, ldc);
                constexpr arrangement arr = choose<M>(base_type::this_blas_arrangement_a,
                                                      base_type::this_blas_arrangement_b,
                                                      base_type::this_blas_arrangement_c);
                return pointer_layout(smem_tag {}, get_layout_smem_impl<arr>(rows {}, columns {}, cute::Int<ld> {}));
            }

            template<matrix M>
            __forceinline__ __host__ __device__ constexpr static auto get_layout_smem(const unsigned int ld) {
                using rows                = cute::Int<M == matrix::B ? exec_t::k : exec_t::m>;
                using columns             = cute::Int<M == matrix::A ? exec_t::k : exec_t::n>;
                constexpr arrangement arr = choose<M>(base_type::this_blas_arrangement_a,
                                                      base_type::this_blas_arrangement_b,
                                                      base_type::this_blas_arrangement_c);
                return pointer_layout {smem_tag {}, get_layout_smem_impl<arr>(rows {}, columns {}, ld)};
            }


            template<matrix M>
            __forceinline__ __host__ __device__ constexpr static auto
            suggest_layout_smem() {
                using a_type = typename this_cutlass_value_type::a_type;
                using b_type = typename this_cutlass_value_type::b_type;
                using c_type = typename this_cutlass_value_type::c_type;
                constexpr bool is_a_left = base_type::this_blas_arrangement_a == arrangement::col_major;
                constexpr bool is_b_left = base_type::this_blas_arrangement_b == arrangement::col_major;
                constexpr bool is_c_left = base_type::this_blas_arrangement_c == arrangement::col_major;

                auto db_entry = cublasdx::detail::layout_database::get_optimal_layout<M,
                    has_block_dim, max_threads_per_block,
                    base_type::this_blas_sm_v,
                    a_type, is_a_left, a_alignment,
                    b_type, is_b_left, b_alignment,
                    c_type, is_c_left, c_alignment,
                    base_type::this_blas_size_m_v,
                    base_type::this_blas_size_n_v,
                    base_type::this_blas_size_k_v>();

                return pointer_layout(smem_tag{}, db_entry);
            }

        public:
            using a_value_type = typename this_blas_value_type::a_type;
            using b_value_type = typename this_blas_value_type::b_type;
            using c_value_type = typename this_blas_value_type::c_type;

            template<typename ... Ts>
            __forceinline__ __host__ __device__ constexpr static auto
            get_layout_gmem_a(Ts ... ts) {
                return get_layout_gmem<matrix::A>(ts...);
            }

            template<typename ... Ts>
            __forceinline__ __host__ __device__ constexpr static auto
            get_layout_gmem_b(Ts ... ts) {
                return get_layout_gmem<matrix::B>(ts...);
            }

            template<typename ... Ts>
            __forceinline__ __host__ __device__ constexpr static auto
            get_layout_gmem_c(Ts ... ts) {
                return get_layout_gmem<matrix::C>(ts...);
            }

            template<typename ... Ts>
            __forceinline__ __host__ __device__ constexpr static auto
            get_layout_smem_a(Ts ... ts) {
                return get_layout_smem<matrix::A>(ts...);
            }

            template<typename ... Ts>
            __forceinline__ __host__ __device__ constexpr static auto
            get_layout_smem_b(Ts ... ts) {
                return get_layout_smem<matrix::B>(ts...);
            }

            template<typename ... Ts>
            __forceinline__ __host__ __device__ constexpr static auto
            get_layout_smem_c(Ts ... ts) {
                return get_layout_smem<matrix::C>(ts...);
            }

            __forceinline__ __host__ __device__ constexpr static auto
            suggest_layout_smem_a() {
                return suggest_layout_smem<matrix::A>();
            }

            __forceinline__ __host__ __device__ constexpr static auto
            suggest_layout_smem_b() {
                return suggest_layout_smem<matrix::B>();
            }

            __forceinline__ __host__ __device__ constexpr static auto
            suggest_layout_smem_c() {
                return get_layout_smem<matrix::C>();
            }

            // T - can be any type if its alignment and size are the same as those of ::value_type
            template<class AEngine, class ALayout,
                     class BEngine, class BLayout,
                     class CEngine, class CLayout,
                     class ALoadOp = identity, class BLoadOp = identity,
                     class CLoadOp = identity, class CStoreOp = identity>
            inline __device__ auto execute(const typename CEngine::value_type& alpha,
                                           cublasdx::tensor<AEngine, ALayout>  tensor_a,
                                           cublasdx::tensor<BEngine, BLayout>  tensor_b,
                                           const typename CEngine::value_type& beta,
                                           cublasdx::tensor<CEngine, CLayout>  tensor_c,
                                           const ALoadOp&                      a_load_op  = {},
                                           const BLoadOp&                      b_load_op  = {},
                                           const CLoadOp&                      c_load_op  = {},
                                           const CStoreOp&                     c_store_op = {})
                -> execute_enable_if_t<typename CEngine::value_type /* Alpha */,
                                       typename AEngine::value_type,
                                       typename BEngine::value_type,
                                       typename CEngine::value_type /* Beta */,
                                       typename CEngine::value_type>{

                using AShape = decltype(shape(ALayout{}));
                using BShape = decltype(shape(BLayout{}));
                using CShape = decltype(shape(CLayout{}));

                // Check if sizes are static
                static_assert(cute::is_static_v<AShape> && cute::is_static_v<BShape> && cute::is_static_v<CShape>,
                              "All layout shapes must be static, only strides can be dynamic");

                // Check if layout shapes are 2D and non-hierarchical
                // Check if layout shapes are compatible with operator defined shapes
                static_assert(
                    rank(AShape{}) == 2 && size(cute::get<0>(AShape{})) == base_type::this_blas_size_m_v &&
                                           size(cute::get<1>(AShape{})) == base_type::this_blas_size_k_v &&

                    rank(BShape{}) == 2 && size(cute::get<0>(BShape{})) == base_type::this_blas_size_k_v &&
                                           size(cute::get<1>(BShape{})) == base_type::this_blas_size_n_v &&

                    rank(CShape{}) == 2 && size(cute::get<0>(CShape{})) == base_type::this_blas_size_m_v &&
                                           size(cute::get<1>(CShape{})) == base_type::this_blas_size_n_v,
                    "Tensor API currently supports only \
                     hierarchical 2D tensors sizes of which \
                     match operator provided sizes"
                );

                using a_engine_type = typename AEngine::value_type;
                using b_engine_type = typename BEngine::value_type;
                using c_engine_type = typename CEngine::value_type;
                static_assert(is_functor_compatible(ALoadOp{}, AEngine{}),
                    "ALoadOp functor must accept value of tensor_a type and return value convertible to tensor_a type");
                static_assert(is_functor_compatible(BLoadOp{}, BEngine{}),
                    "BLoadOp functor must accept value of tensor_b type and return value convertible to tensor_b type");
                static_assert(is_functor_compatible(CLoadOp{}, CEngine{}),
                    "CLoadOp functor must accept value of tensor_c type and return value convertible to tensor_c type");
                static_assert(is_functor_compatible(CStoreOp{}, CEngine{}),
                    "CStoreOp functor must accept value of tensor_c type and return value convertible to tensor_c type");

                transform_op_wrapper<ALoadOp, a_value_type> cutlass_a_load_op{a_load_op};
                transform_op_wrapper<BLoadOp, b_value_type> cutlass_b_load_op{b_load_op};
                transform_op_wrapper<CLoadOp, c_value_type> cutlass_c_load_op{c_load_op};
                transform_op_wrapper<CStoreOp, c_value_type> cutlass_c_store_op{c_store_op};

                exec_t::tensor_gemm(
                    cute::recast<typename this_cutlass_value_type::a_type>(tensor_a),
                    cute::recast<typename this_cutlass_value_type::b_type>(tensor_b),
                    cute::recast<typename this_cutlass_value_type::c_type>(tensor_c),
                    cast_to_cutlass_type<typename this_cutlass_value_type::c_type>(alpha),
                    cast_to_cutlass_type<typename this_cutlass_value_type::c_type>(beta),
                    cutlass_a_load_op,
                    cutlass_b_load_op,
                    cutlass_c_load_op,
                    cutlass_c_store_op);
            }

            template<class AEngine, class ALayout,
                     class BEngine, class BLayout,
                     class CEngine, class CLayout,
                     class ALoadOp, class BLoadOp,
                     class CLoadOp, class CStoreOp>
            inline __device__ auto execute(const typename CEngine::value_type& /* alpha */,
                                           cublasdx::tensor<AEngine, ALayout>  /* tensor_a */,
                                           cublasdx::tensor<BEngine, BLayout>  /* tensor_b */,
                                           const typename CEngine::value_type& /* beta */,
                                           cublasdx::tensor<CEngine, CLayout>  /* tensor_c */,
                                           [[maybe_unused]] const ALoadOp&  a_load_op  = identity {},
                                           [[maybe_unused]] const BLoadOp&  b_load_op  = identity {},
                                           [[maybe_unused]] const CLoadOp&  c_load_op  = identity {},
                                           [[maybe_unused]] const CStoreOp& c_store_op = identity {})
                -> execute_disable_if_t<typename CEngine::value_type, /* Alpha */
                                        typename AEngine::value_type,
                                        typename BEngine::value_type,
                                        typename CEngine::value_type, /* Beta */
                                        typename CEngine::value_type> {

                constexpr bool condition = are_types_compatible<
                    typename CEngine::value_type /* Alpha */,
                    typename AEngine::value_type,
                    typename BEngine::value_type,
                    typename CEngine::value_type /* Beta */,
                    typename CEngine::value_type>();

                static_assert(condition, "Incorrect types for inputs, LD operator used or TransposeMode used. \
                    Ensure input types for A, B and C match the types \
                    indicated in the Precision<...> operator.");
            }

            // T - can be any type if its alignment and size are the same as those of ::value_type
            template<class TA, class TB, class TC,
                     class ALoadOp = identity, class BLoadOp = identity,
                     class CLoadOp = identity, class CStoreOp = identity>
            inline __device__ auto execute(const TC&          alpha,
                                           TA*                matrix_a,
                                           const unsigned int lda,
                                           TB*                matrix_b,
                                           const unsigned int ldb,
                                           const TC&          beta,
                                           TC*                matrix_c,
                                           const unsigned int ldc,
                                           const ALoadOp&     a_load_op  = {},
                                           const BLoadOp&     b_load_op  = {},
                                           const CLoadOp&     c_load_op  = {},
                                           const CStoreOp&    c_store_op = {}) //
                -> execute_enable_if_t<TA, TB, TC> {
                cute::Tensor ta = cublasdx::make_tensor(
                    cute::make_smem_ptr(matrix_a),
                    cute_backend::make_layout_from_arrangement<base_type::this_blas_arrangement_a>(
                        cute::Int<exec_t::m>{}, cute::Int<exec_t::k>{}, lda
                    )
                );
                cute::Tensor tb = cublasdx::make_tensor(
                    cute::make_smem_ptr(matrix_b),
                    cute_backend::make_layout_from_arrangement<base_type::this_blas_arrangement_b>(
                        cute::Int<exec_t::k>{}, cute::Int<exec_t::n>{}, ldb
                    )
                );
                cute::Tensor tc = cublasdx::make_tensor(
                    cute::make_smem_ptr(matrix_c),
                    cute_backend::make_layout_from_arrangement<base_type::this_blas_arrangement_c>(
                        cute::Int<exec_t::m>{}, cute::Int<exec_t::n>{}, ldc
                    )
                );

                execute(alpha, ta, tb, beta, tc,
                        compose_functors(cute_backend::get_load_op_from_transpose<base_type::this_blas_transpose_mode_a>(), a_load_op),
                        compose_functors(cute_backend::get_load_op_from_transpose<base_type::this_blas_transpose_mode_b>(), b_load_op),
                        c_load_op,
                        c_store_op);
            }

            // T - can be any type if its alignment and size are the same as those of ::value_type
            template<class TA, class TB, class TC,
                     class ALoadOp = identity, class BLoadOp = identity,
                     class CLoadOp = identity, class CStoreOp = identity>
            inline __device__ auto execute(const TC        alpha,
                                           TA*             matrix_a,
                                           TB*             matrix_b,
                                           const TC        beta,
                                           TC*             matrix_c,
                                           const ALoadOp&  a_load_op  = {},
                                           const BLoadOp&  b_load_op  = {},
                                           const CLoadOp&  c_load_op  = {},
                                           const CStoreOp& c_store_op = {}) //
                -> execute_enable_if_t<TA, TB, TC> {
                cute::Tensor ta = cublasdx::make_tensor(
                    cute::make_smem_ptr(matrix_a),
                    cute_backend::make_layout_from_arrangement<base_type::this_blas_arrangement_a>(
                        cute::Int<exec_t::m>{}, cute::Int<exec_t::k>{}, cute::Int<this_type::lda>{}
                    )
                );
                cute::Tensor tb = cublasdx::make_tensor(
                    cute::make_smem_ptr(matrix_b),
                    cute_backend::make_layout_from_arrangement<base_type::this_blas_arrangement_b>(
                        cute::Int<exec_t::k>{}, cute::Int<exec_t::n>{}, cute::Int<this_type::ldb>{}
                    )
                );
                cute::Tensor tc = cublasdx::make_tensor(
                    cute::make_smem_ptr(matrix_c),
                    cute_backend::make_layout_from_arrangement<base_type::this_blas_arrangement_c>(
                        cute::Int<exec_t::m>{}, cute::Int<exec_t::n>{}, cute::Int<this_type::ldc>{}
                    )
                );

                execute(alpha, ta, tb, beta, tc,
                        compose_functors(cute_backend::get_load_op_from_transpose<base_type::this_blas_transpose_mode_a>(), a_load_op),
                        compose_functors(cute_backend::get_load_op_from_transpose<base_type::this_blas_transpose_mode_b>(), b_load_op),
                        c_load_op,
                        c_store_op);
            }

            template<class TA, class TB, class TC,
                     class ALoadOp = identity,  class BLoadOp  = identity,
                     class CLoadOp = identity,  class CStoreOp = identity>
            inline __device__ auto execute(const TC /* alpha */,
                                           TA* /* matrix_a */,
                                           const unsigned int /* lda */,
                                           TB* /* matrix_b */,
                                           const unsigned int /* ldb */,
                                           const TC /* beta */,
                                           TC* /* matrix_c */,
                                           const unsigned int /* ldc */,
                                           const ALoadOp& /* a_load_op */ = {},
                                           const BLoadOp& /* b_load_op */ = {},
                                           const CLoadOp& /* c_load_op */ = {},
                                           const CStoreOp& /* c_store_op */ = {}) //
                -> execute_disable_if_t<TA, TB, TC> {
                static constexpr bool condition = are_types_compatible<TA, TB, TC>();

                static_assert(condition,
                "Incorrect types for inputs or lacking TransposeMode operator.  \
                Ensure input types for A, B and C match the types \
                indicated in the Precision<...> operator.");
            }

            template<class TA, class TB, class TC,
                     class ALoadOp = identity, class BLoadOp = identity,
                     class CLoadOp = identity, class CStoreOp = identity>
            inline __device__ auto execute(const TC /* alpha */,
                                           TA* /* matrix_a */,
                                           TB* /* matrix_b */,
                                           const TC /* beta */,
                                           TC* /* matrix_c */,
                                           const ALoadOp& /* a_load_op */ = {},
                                           const BLoadOp& /* b_load_op */ = {},
                                           const CLoadOp& /* c_load_op */ = {},
                                           const CStoreOp& /* c_store_op */ = {}) //
                -> execute_disable_if_t<TA, TB, TC> {
                static constexpr bool condition = are_types_compatible<TA, TB, TC>();

                static_assert(condition,
                "Incorrect types for inputs or lacking TransposeMode operator.  \
                Ensure input types for A, B and C match the types \
                indicated in the Precision<...> operator.");
            }

            inline static constexpr unsigned int get_shared_memory_size() {
                static_assert(base_type::is_complete, "Can't calculate shared memory, description is not complete");
                using smem_alignment_t = smem_alignment<this_blas_value_type,
                                                        base_type::this_blas_a_size,
                                                        base_type::this_blas_b_size,
                                                        base_type::this_blas_c_size,
                                                        a_alignment, b_alignment, c_alignment>;
                return sizeof(smem_alignment_t);
            }

            inline static auto __device__ slice_shared_memory(char* smem_ptr) {
                static_assert(base_type::is_complete, "Can't slice shared memory, description is not complete");
                using smem_alignment_t = smem_alignment<this_blas_value_type,
                                                        base_type::this_blas_a_size,
                                                        base_type::this_blas_b_size,
                                                        base_type::this_blas_c_size,
                                                        a_alignment, b_alignment, c_alignment>;
                auto smem_align = (smem_alignment_t*)smem_ptr;
                return COMMONDX_STL_NAMESPACE::make_tuple(smem_align->a, smem_align->b, smem_align->c);
            }

            inline static constexpr unsigned int get_shared_memory_size(unsigned int lda, unsigned int ldb, unsigned int ldc) {
                static_assert(base_type::is_complete, "Can't calculate shared memory, description is not complete");
                smem_alignment_dyn<this_blas_value_type, a_alignment, b_alignment, c_alignment> smem_align(base_type::this_blas_size_m_v,
                                                                                                           base_type::this_blas_size_n_v,
                                                                                                           base_type::this_blas_size_k_v,
                                                                                                           base_type::this_blas_arrangement_a,
                                                                                                           base_type::this_blas_arrangement_b,
                                                                                                           base_type::this_blas_arrangement_c,
                                                                                                           lda, ldb, ldc);
                return smem_align.get_memory_usage();
            }

            inline static auto __device__ slice_shared_memory(char* smem_ptr, unsigned int lda, unsigned int ldb, unsigned int ldc) {
                static_assert(base_type::is_complete, "Can't slice shared memory, description is not complete");
                smem_alignment_dyn<this_blas_value_type, a_alignment, b_alignment, c_alignment> smem_align(base_type::this_blas_size_m_v,
                                                                                                           base_type::this_blas_size_n_v,
                                                                                                           base_type::this_blas_size_k_v,
                                                                                                           base_type::this_blas_arrangement_a,
                                                                                                           base_type::this_blas_arrangement_b,
                                                                                                           base_type::this_blas_arrangement_c,
                                                                                                           lda, ldb, ldc);
                return COMMONDX_STL_NAMESPACE::make_tuple(reinterpret_cast<a_value_type*>(&smem_ptr[smem_align.get_a_offset()]),
                                                          reinterpret_cast<b_value_type*>(&smem_ptr[smem_align.get_b_offset()]),
                                                          reinterpret_cast<c_value_type*>(&smem_ptr[smem_align.get_c_offset()]));
            }

            template<class ALayout, class BLayout, class CLayout>
            inline static constexpr auto get_shared_memory_size(ALayout a_layout, BLayout b_layout, CLayout c_layout)
                -> COMMONDX_STL_NAMESPACE::enable_if_t<are_layout_acceptable<ALayout, BLayout, CLayout>(), unsigned> {
                static_assert(base_type::is_complete, "Can't calculate shared memory, description is not complete");
                smem_alignment_dyn<this_blas_value_type, a_alignment, b_alignment, c_alignment> smem_align(cute::cosize(a_layout),
                                                                                                           cute::cosize(b_layout),
                                                                                                           cute::cosize(c_layout));
                return smem_align.get_memory_usage();
            }

            template<class ALayout, class BLayout, class CLayout>
            inline static __device__ auto slice_shared_memory(char* smem_ptr, ALayout a_layout, BLayout b_layout, CLayout c_layout)
                -> COMMONDX_STL_NAMESPACE::enable_if_t<are_layout_acceptable<ALayout, BLayout, CLayout>(),
                                                       COMMONDX_STL_NAMESPACE::tuple<a_value_type*, b_value_type*, c_value_type*>> {
                static_assert(base_type::is_complete, "Can't slice shared memory, description is not complete");
                smem_alignment_dyn<this_blas_value_type, a_alignment, b_alignment, c_alignment> smem_align(cute::cosize(a_layout),
                                                                                                           cute::cosize(b_layout),
                                                                                                           cute::cosize(c_layout));
                return COMMONDX_STL_NAMESPACE::make_tuple(reinterpret_cast<a_value_type*>(&smem_ptr[smem_align.get_a_offset()]),
                                                          reinterpret_cast<b_value_type*>(&smem_ptr[smem_align.get_b_offset()]),
                                                          reinterpret_cast<c_value_type*>(&smem_ptr[smem_align.get_c_offset()]));
            }

            // Number of elements in A, B, C matrices (includes padding / leading dimensions)
            // (ld * cols)
            static constexpr unsigned int a_size = base_type::this_blas_a_size;
            static constexpr unsigned int b_size = base_type::this_blas_b_size;
            static constexpr unsigned int c_size = base_type::this_blas_c_size;

            // Leading dimensions of A, B, C matrices
            static constexpr unsigned int lda = base_type::this_blas_lda;
            static constexpr unsigned int ldb = base_type::this_blas_ldb;
            static constexpr unsigned int ldc = base_type::this_blas_ldc;

            // Pointer alignments of A, B, C matrices in bytes
            static constexpr unsigned int a_alignment = base_type::this_blas_alignment_a;
            static constexpr unsigned int b_alignment = base_type::this_blas_alignment_b;
            static constexpr unsigned int c_alignment = base_type::this_blas_alignment_c;

            // Logical dimensions of A, B, C matrices
            // (row; cols)
            static constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int> a_dim = base_type::this_blas_a_dim;
            static constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int> b_dim = base_type::this_blas_b_dim;
            static constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int> c_dim = base_type::this_blas_c_dim;

            static constexpr dim3         suggested_block_dim = get_suggested_block_dim();
            static constexpr dim3         block_dim           = get_block_dim();
            static constexpr unsigned int shared_memory_size  = get_shared_memory_size();

            static constexpr unsigned int max_threads_per_block         = block_dim.x * block_dim.y * block_dim.z;
            static constexpr unsigned int min_blocks_per_multiprocessor = 1;
        };

        template<class... Operators>
        struct make_description {
        private:
            static constexpr bool has_block_operator     = has_operator<operator_type::block, blas_operator_wrapper<Operators...>>::value;
            static constexpr bool has_execution_operator = has_block_operator;

            // Workaround (NVRTC/MSVC)
            //
            // For NVRTC we need to utilize a in-between class called blas_block_execution_partial, otherwise
            // we run into a complation error if Block() is added to description before BLAS description is
            // complete, example:
            //
            // Fails on NVRTC:
            //     Size<...>() + Function<...>() + Type<...>() + Precision<...>() + Block() + SM<700>()
            // Works on NVRTC:
            //     Size<...>() + Function<...>() + Type<...>() + Precision<...>() + SM<700>() + Block()
            //
            // This workaround disables some useful diagnostics based on static_asserts.
#if defined(__CUDACC_RTC__) || defined(_MSC_VER)
            using operator_wrapper_type = blas_operator_wrapper<Operators...>;
            using execution_type =
                typename COMMONDX_STL_NAMESPACE::conditional<is_complete_blas<operator_wrapper_type>::value,
                                                             blas_block_execution<Operators...>,
                                                             blas_block_execution_partial<Operators...>>::type;
#else
            using execution_type = blas_block_execution<Operators...>;
#endif
            using description_type = blas_description<Operators...>;

        public:
            using type = typename COMMONDX_STL_NAMESPACE::
                conditional<has_execution_operator, execution_type, description_type>::type;
        };

        template<class... Operators>
        using make_description_t = typename make_description<Operators...>::type;
    } // namespace detail

    template<class Operator1, class Operator2>
    __host__ __device__ __forceinline__ auto operator+(const Operator1&, const Operator2&) //
        -> typename COMMONDX_STL_NAMESPACE::enable_if<commondx::detail::are_operator_expressions<Operator1, Operator2>::value,
                                   detail::make_description_t<Operator1, Operator2>>::type {
        return detail::make_description_t<Operator1, Operator2>();
    }

    template<class... Operators1, class Operator2>
    __host__ __device__ __forceinline__ auto operator+(const detail::blas_description<Operators1...>&,
                                                       const Operator2&) //
        -> typename COMMONDX_STL_NAMESPACE::enable_if<commondx::detail::is_operator_expression<Operator2>::value,
                                   detail::make_description_t<Operators1..., Operator2>>::type {
        return detail::make_description_t<Operators1..., Operator2>();
    }

    template<class Operator1, class... Operators2>
    __host__ __device__ __forceinline__ auto operator+(const Operator1&,
                                                       const detail::blas_description<Operators2...>&) //
        -> typename COMMONDX_STL_NAMESPACE::enable_if<commondx::detail::is_operator_expression<Operator1>::value,
                                   detail::make_description_t<Operator1, Operators2...>>::type {
        return detail::make_description_t<Operator1, Operators2...>();
    }

    template<class... Operators1, class... Operators2>
    __host__ __device__ __forceinline__ auto operator+(const detail::blas_description<Operators1...>&,
                                                       const detail::blas_description<Operators2...>&) //
        -> detail::make_description_t<Operators1..., Operators2...> {
        return detail::make_description_t<Operators1..., Operators2...>();
    }
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_BLAS_EXECUTION_HPP
