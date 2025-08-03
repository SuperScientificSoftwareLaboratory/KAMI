// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_BLAS_DESCRIPTION_HPP
#define CUBLASDX_DETAIL_BLAS_DESCRIPTION_HPP

#include "commondx/detail/stl/type_traits.hpp"

#include "commondx/traits/detail/get.hpp"
#include "commondx/detail/expressions.hpp"

#include "../operators.hpp"
#include "../traits/detail/description_traits.hpp"

#include "blas_checks.hpp"

#define STRINGIFY(s) XSTRINGIFY(s)
#define XSTRINGIFY(s) #s

namespace cublasdx {
    namespace detail {
        constexpr unsigned int calculate_matrix_size(unsigned int ld, unsigned int x, unsigned int y, arrangement arr) {
            return ld * ((arr == arrangement::col_major) ? y : x);
        }

        constexpr unsigned int calculate_matrix_size(unsigned int ld, unsigned int x, unsigned int y, transpose_mode tmode) {
            return ld * ((tmode == N) ? y : x);
        }

        template<size_t org_size, size_t alignment = 16>
        inline constexpr size_t aligned_size() {
            return ((org_size + alignment - 1) / alignment) * alignment;
        }
        template<size_t alignment = 16>
        inline constexpr size_t aligned_dynamic_size(const size_t org_size) {
            return ((org_size + alignment - 1) / alignment) * alignment;
        }

        template<class... Operators>
        class blas_operator_wrapper: public commondx::detail::description_expression { };

        template<class... Operators>
        class blas_description: public commondx::detail::description_expression
        {
            using description_type = blas_operator_wrapper<Operators...>;

        protected:
            /// ---- Traits

            // Size
            // * Default value: NONE
            // * If there is no size, then dummy size is (8, 8, 8). This is required value for M, N sized don't break.
            // * Values of has_size or is_complete should be checked before using this property.
            static constexpr bool has_size           = has_operator<operator_type::size, description_type>::value;
            using dummy_default_blas_size            = Size<8, 8, 8>;
            using this_blas_size                     = get_or_default_t<operator_type::size, description_type, dummy_default_blas_size>;
            static constexpr auto this_blas_size_m_v = this_blas_size::m;
            static constexpr auto this_blas_size_n_v = this_blas_size::n;
            static constexpr auto this_blas_size_k_v = this_blas_size::k;

            // Type (real, complex)
            // * Default value: real
            using this_blas_type                   = get_or_default_t<operator_type::type, description_type, default_blas_type_operator>;
            static constexpr auto this_blas_type_v = this_blas_type::value;

            // Function
            // * Default value: NONE
            // * Dummy value: MM
            static constexpr bool has_function = has_operator<operator_type::function, description_type>::value;
            using dummy_default_blas_function  = Function<function::MM>;
            using this_blas_function           = get_or_default_t<operator_type::function, description_type, dummy_default_blas_function>;
            static constexpr auto this_blas_function_v = this_blas_function::value;

            // Precision
            // * Default: A, B, C are all float
            using this_blas_precision   = get_or_default_t<operator_type::precision, description_type, default_blas_precision_operator>;

            // SM
            // * Default value: NONE
            // * Dummy value: 700
            static constexpr bool has_sm         = has_operator<operator_type::sm, description_type>::value;
            using dummy_default_blas_sm          = SM<700>;
            using this_blas_sm                   = get_or_default_t<operator_type::sm, description_type, dummy_default_blas_sm>;
            static constexpr auto this_blas_sm_v = this_blas_sm::value;

#if (__GNUC__ < 8) // Bug workaround
        public:
#endif
            // Arrangement, TransposeMode
            static constexpr bool has_arrangement    = has_operator<operator_type::arrangement, description_type>::value;
            static constexpr bool has_transpose_mode = has_operator<operator_type::transpose_mode, description_type>::value;
#if (__GNUC__ < 8) // Bug workaround
        protected:
#endif

            // Arrangement
            // * Default value: col_major, col_major, col_major
            using default_blas_arrangement =
                COMMONDX_STL_NAMESPACE::conditional_t<
                    has_transpose_mode,
                    typename convert_to_arrangement<
                        get_or_default_t<operator_type::transpose_mode, description_type, default_blas_transpose_mode_operator>
                    >::type,
                    default_blas_arrangement_operator
                >;
            using this_blas_arrangement =
                get_or_default_t<operator_type::arrangement, description_type, default_blas_arrangement>;
            static constexpr auto this_blas_arrangement_a = this_blas_arrangement::a;
            static constexpr auto this_blas_arrangement_b = this_blas_arrangement::b;
            static constexpr auto this_blas_arrangement_c = this_blas_arrangement::c;

            // TransposeMode
            // * Default value: N, N
            using default_blas_transpose_mode  =
                COMMONDX_STL_NAMESPACE::conditional_t<
                    has_arrangement,
                    typename convert_to_transpose_mode<
                        get_or_default_t<operator_type::arrangement, description_type, default_blas_arrangement_operator>
                    >::type,
                    default_blas_transpose_mode_operator
                >;
            using this_blas_transpose_mode =
                get_or_default_t<operator_type::transpose_mode, description_type, default_blas_transpose_mode>;
            static constexpr auto this_blas_transpose_mode_a = this_blas_transpose_mode::a_transpose_mode;
            static constexpr auto this_blas_transpose_mode_b = this_blas_transpose_mode::b_transpose_mode;

            // // FillMode
            // // * Default value: NONE
            // // * Dummy value: fill_mode::lower
            // static constexpr bool has_fill_mode = has_operator<operator_type::fill_mode, description_type>::value;
            // using dummy_default_blas_fill_mode  = FillMode<fill_mode::lower>;
            // using this_blas_fill_mode = get_or_default_t<operator_type::fill_mode, description_type, dummy_default_blas_fill_mode>;
            // static constexpr auto this_blas_fill_mode_v = this_blas_fill_mode::value;

            // // Diagonal
            // // * Default value: NONE
            // // * Dummy value: diagonal::non_unit
            // static constexpr bool has_diagonal = has_operator<operator_type::diagonal, description_type>::value;
            // using dummy_default_blas_diagonal  = Diagonal<diagonal::non_unit>;
            // using this_blas_diagonal           = get_or_default_t<operator_type::diagonal, description_type, dummy_default_blas_diagonal>;
            // static constexpr auto this_blas_diagonal_v = this_blas_diagonal::value;

            // // Side
            // // * Default value: NONE
            // // * Dummy value: side::left
            // static constexpr bool has_side         = has_operator<operator_type::side, description_type>::value;
            // using dummy_default_blas_side          = Side<side::left>;
            // using this_blas_side                   = get_or_default_t<operator_type::side, description_type, dummy_default_blas_side>;
            // static constexpr auto this_blas_side_v = this_blas_side::value;

            // Alignment
            static constexpr bool has_alignment = has_operator<operator_type::alignment, description_type>::value;
            using default_blas_alignment =
                Alignment<alignof(typename map_value_type<this_blas_type_v, this_blas_precision>::a_type),
                          alignof(typename map_value_type<this_blas_type_v, this_blas_precision>::b_type),
                          alignof(typename map_value_type<this_blas_type_v, this_blas_precision>::c_type)>;
            using this_blas_alignment           = get_or_default_t<operator_type::alignment, description_type, default_blas_alignment>;
            static constexpr auto this_blas_alignment_a = this_blas_alignment::a;
            static constexpr auto this_blas_alignment_b = this_blas_alignment::b;
            static constexpr auto this_blas_alignment_c = this_blas_alignment::c;

            // LeadingDimension
            static constexpr bool has_ld = has_operator<operator_type::ld, description_type>::value;
            static constexpr unsigned int default_lda =
                ((this_blas_arrangement_a == arrangement::col_major) ? this_blas_size_m_v : this_blas_size_k_v);
            static constexpr unsigned int default_ldb =
                ((this_blas_arrangement_b == arrangement::col_major) ? this_blas_size_k_v : this_blas_size_n_v);
            static constexpr unsigned int default_ldc =
                ((this_blas_arrangement_c == arrangement::col_major) ? this_blas_size_m_v : this_blas_size_n_v);
#if defined(__NVCC__) && (__CUDACC_VER_MAJOR__ <= 11) && (__CUDACC_VER_MINOR__ <= 5)
            // NVCC 11.4/11.5 workaround for incorrect error:
            // error: ‘constexpr const unsigned int cublasdx::detail::blas_description<...>::default_lda’ is protected within this context
            using dummy_default_blas_ld               = LeadingDimension<1, 1, 1>;
            static constexpr auto this_blas_lda =
                has_ld ? get_or_default_t<operator_type::ld, description_type, dummy_default_blas_ld>::a : default_lda;
            static constexpr auto this_blas_ldb =
                has_ld ? get_or_default_t<operator_type::ld, description_type, dummy_default_blas_ld>::b : default_ldb;
            static constexpr auto this_blas_ldc =
                has_ld ? get_or_default_t<operator_type::ld, description_type, dummy_default_blas_ld>::c : default_ldc;
#elif defined(__NVCC__) && (__CUDACC_VER_MAJOR__ <= 11)
            using default_blas_ld = LeadingDimension<default_lda, default_ldb, default_ldc>;
            using this_blas_ld = get_or_default_t<operator_type::ld, description_type, default_blas_ld>;
            static constexpr auto this_blas_lda = this_blas_ld::a;
            static constexpr auto this_blas_ldb = this_blas_ld::b;
            static constexpr auto this_blas_ldc = this_blas_ld::c;
#else
            // NVCC 12.X.X workaround
            using dummy_default_blas_ld               = LeadingDimension<1, 1, 1>;
            static constexpr auto this_blas_lda =
                has_ld ? get_or_default_t<operator_type::ld, description_type, dummy_default_blas_ld>::a : default_lda;
            static constexpr auto this_blas_ldb =
                has_ld ? get_or_default_t<operator_type::ld, description_type, dummy_default_blas_ld>::b : default_ldb;
            static constexpr auto this_blas_ldc =
                has_ld ? get_or_default_t<operator_type::ld, description_type, dummy_default_blas_ld>::c : default_ldc;
            using this_blas_ld = LeadingDimension<this_blas_lda, this_blas_ldb, this_blas_ldc>;
#endif

            // Number of real elements in each matrix (includes padding)
            static constexpr auto this_blas_a_size =
                calculate_matrix_size(this_blas_lda, this_blas_size_m_v, this_blas_size_k_v, this_blas_arrangement_a);
            static constexpr auto this_blas_b_size =
                calculate_matrix_size(this_blas_ldb, this_blas_size_k_v, this_blas_size_n_v, this_blas_arrangement_b);
            static constexpr auto this_blas_c_size =
                calculate_matrix_size(this_blas_ldc, this_blas_size_m_v, this_blas_size_n_v, this_blas_arrangement_c);

            // Logical dimensions
            // rows, cols
            static constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int> this_blas_a_dim {
                default_lda,
                ((this_blas_arrangement_a == arrangement::col_major) ? this_blas_size_k_v : this_blas_size_m_v)};
            static constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int> this_blas_b_dim {
                default_ldb,
                ((this_blas_arrangement_b == arrangement::col_major) ? this_blas_size_n_v : this_blas_size_k_v)};
            static constexpr COMMONDX_STL_NAMESPACE::tuple<unsigned int, unsigned int> this_blas_c_dim {
                default_ldc,
                ((this_blas_arrangement_c == arrangement::col_major) ? this_blas_size_n_v : this_blas_size_m_v)};

            // True if description is complete description
            static constexpr bool is_complete = is_complete_description<description_type>::value;

            /// ---- Constraints

            // We can only have one of each option

            // Main operators
            static constexpr bool has_one_function       = has_at_most_one_of<operator_type::function, description_type>::value;
            static constexpr bool has_one_precision      = has_at_most_one_of<operator_type::precision, description_type>::value;
            static constexpr bool has_one_size           = has_at_most_one_of<operator_type::size, description_type>::value;
            static constexpr bool has_one_sm             = has_at_most_one_of<operator_type::sm, description_type>::value;
            static constexpr bool has_one_type           = has_at_most_one_of<operator_type::type, description_type>::value;
            static constexpr bool has_one_block_dim      = has_at_most_one_of<operator_type::block_dim, description_type>::value;
            // static constexpr bool has_one_side           = has_at_most_one_of<operator_type::side, description_type>::value;
            // static constexpr bool has_one_diagonal       = has_at_most_one_of<operator_type::diagonal, description_type>::value;
            static constexpr bool has_one_alignment      = has_at_most_one_of<operator_type::alignment, description_type>::value;
            static constexpr bool has_one_ld             = has_at_most_one_of<operator_type::ld, description_type>::value;
            // static constexpr bool has_one_fill_mode      = has_at_most_one_of<operator_type::fill_mode, description_type>::value;
            static constexpr bool has_one_transpose_mode = has_at_most_one_of<operator_type::transpose_mode, description_type>::value;
            static constexpr bool has_one_arrangement     = has_at_most_one_of<operator_type::arrangement, description_type>::value;

            static_assert(has_one_function, "Can't create blas function with two Function<> expressions");
            static_assert(has_one_precision, "Can't create blas function with two Precision<> expressions");
            static_assert(has_one_size, "Can't create blas function with two Size<> expressions");
            static_assert(has_one_sm, "Can't create blas function with two SM<> expressions");
            static_assert(has_one_type, "Can't create blas function with two Type<> expressions");
            static_assert(has_one_block_dim, "Can't create blas function with two BlockDim<> expressions");
            // static_assert(has_one_side, "Can't create blas function with two Side<> expressions");
            // static_assert(has_one_diagonal, "Can't create blas function with two Diagonal<> expressions");
            static_assert(has_one_alignment, "Can't create blas function with two Alignment<> expressions");
            static_assert(has_one_ld, "Can't create blas function with two LeadingDimension<> expressions");
            // static_assert(has_one_fill_mode, "Can't create blas function with two FillMode<> expressions");
            static_assert(has_one_transpose_mode, "Can't create blas function with two TransposeMode<> expressions");
            static_assert(has_one_arrangement, "Can't create blas function with two Arrangement<> expressions");

            // Operators checks

            // // For TRSM FillMode must be upper or lower
            // static constexpr bool valid_trsm_fill_mode =
            //     !has_function ||
            //     !(this_blas_function_v == function::TRSM) ||
            //     !has_fill_mode ||
            //     ((this_blas_fill_mode_v == fill_mode::upper) || (this_blas_fill_mode_v == fill_mode::lower));
            // static_assert(valid_trsm_fill_mode, "Provided fill mode is invalid, for TRSM fill mode must be fill_mode::lower or fill_mode::upper");

            // // For Diagonal and Side can only be defined with TRSM
            // static constexpr bool valid_mm_description_no_trsm_ops =
            //     !has_function ||
            //     !(this_blas_function_v != function::TRSM) ||
            //     !(has_diagonal || has_side);
            // static_assert(valid_mm_description_no_trsm_ops, "For operators Side<> and Diagonal<> can only be used with TRSM function");

            // Arrangement and TransposeMode
            static_assert(!(has_arrangement && has_transpose_mode), "Can't create blas function with Arrangement<> and TransposeMode<> expressions");

            // Alignment check
            static constexpr bool valid_alignment_a =
                ((this_blas_alignment_a % default_blas_alignment::a) == 0) && (this_blas_alignment_a >= default_blas_alignment::a);
            static_assert(valid_alignment_a, "Incorrect alignment for matrix A; it has to be a multiple of type of matrix A");
            static constexpr bool valid_alignment_b =
                ((this_blas_alignment_b % default_blas_alignment::b) == 0) && (this_blas_alignment_b >= default_blas_alignment::b);
            static_assert(valid_alignment_b, "Incorrect alignment for matrix B; it has to be a multiple of type of matrix B");
            static constexpr bool valid_alignment_c =
                ((this_blas_alignment_c % default_blas_alignment::c) == 0) && (this_blas_alignment_c >= default_blas_alignment::c);
            static_assert(valid_alignment_c, "Incorrect alignment for matrix C; it has to be a multiple of type of matrix C");

            // Leading dimensions check
            // NN --> >=LD(M, K, M)
            // TN --> >=LD(K, K, M)
            // NT --> >=LD(M, N, M)
            // TT --> >=LD(K, N, M)
            static constexpr bool valid_lda =
                !has_ld ||
                !has_size ||
                !(this_blas_function_v == function::MM) ||
                (this_blas_lda >= default_lda);
            static_assert(valid_lda || (this_blas_arrangement_a != arrangement::col_major),
                "Incorrect leading dimension for A matrix, LDA must be greater than M");
            static_assert(valid_lda || (this_blas_arrangement_a == arrangement::col_major),
                "Incorrect leading dimension for A matrix, LDA must be greater than K");
            static constexpr bool valid_ldb =
                !has_ld ||
                !has_size ||
                !(this_blas_function_v == function::MM) ||
                (this_blas_ldb >= default_ldb);
            static_assert(valid_ldb || (this_blas_arrangement_b != arrangement::col_major),
                "Incorrect leading dimension for B matrix, LDB must be greater than K");
            static_assert(valid_ldb || (this_blas_arrangement_b == arrangement::col_major),
                "Incorrect leading dimension for B matrix, LDB must be greater than N");
            static constexpr bool valid_ldc =
                !has_ld ||
                !has_size ||
                !(this_blas_function_v == function::MM) ||
                (this_blas_ldc >= default_ldc);
            static_assert(valid_ldc || (this_blas_arrangement_c != arrangement::col_major),
                "Incorrect leading dimension for C matrix, LDC must be greater than M");
            static_assert(valid_ldc || (this_blas_arrangement_c == arrangement::col_major),
                "Incorrect leading dimension for C matrix, LDB must be greater than N");

            // Size, precision, type, sm check

            // GEMM
            // Size
            static constexpr bool valid_size_for_block_gemm =
                !has_size ||
                !has_function ||
                !has_sm ||
                !(this_blas_function_v == function::MM) ||
                (is_supported_logical_size<this_blas_precision, this_blas_type_v, this_blas_size, this_blas_sm_v>::value);
            static_assert(valid_size_for_block_gemm,
                          "Provided size (M, N, K) for GEMM exceeds maximum supported for selected precision and type. "
                          "Matrices A, B, and C must fit into shared memory.");
            // LD
            static constexpr bool valid_ld_for_block_gemm =
                !has_size ||
                !has_ld ||
                !has_function ||
                !has_sm ||
                !(this_blas_function_v == function::MM) ||
                (is_supported_real_size<this_blas_precision, this_blas_type_v, this_blas_a_size, this_blas_b_size, this_blas_c_size, this_blas_sm_v>::value);
            static_assert(valid_ld_for_block_gemm,
                          "Provided leading dimensions for GEMM exceeds maximum supported for selected precision and type. "
                          "Matrices A, B, and C must fit into shared memory.");

            /// ---- End of Constraints
        };

        template<>
        class blas_description<>: public commondx::detail::description_expression {};
    } // namespace detail
} // namespace cublasdx

#undef STRINGIFY
#undef XSTRINGIFY

#endif // CUBLASDX_DETAIL_BLAS_DESCRIPTION_HPP
