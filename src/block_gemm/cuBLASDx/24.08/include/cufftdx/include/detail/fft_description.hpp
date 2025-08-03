// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_DETAIL_FFT_DESCRIPTION_HPP
#define CUFFTDX_DETAIL_FFT_DESCRIPTION_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#else
#    include <type_traits>
#endif

#include <cuda_fp16.h>

#include "fft_sizes.hpp"
#include "../operators.hpp"
#include "../traits/detail/check_and_get_trait.hpp"
#include "../traits/detail/description_traits.hpp"
#include "../traits/detail/make_complex_type.hpp"
#include "../database/database.hpp"

#include "commondx/detail/expressions.hpp"
namespace cufftdx {
    namespace detail {
        template<class... Operators>
        class fft_operator_wrapper: public commondx::detail::description_expression
        {
        };

        template<class... Operators>
        class fft_description: public commondx::detail::description_expression
        {
            using description_type = fft_operator_wrapper<Operators...>;

        protected:
            /// ---- Traits

            // Size
            // * Default value: NONE
            // * If there is no size, then dummy size is 2. This is required so this_fft_size_v does not break.
            // * Values of has_size or is_complete should be checked before using this property.
            static constexpr bool has_size        = has_operator<fft_operator::size, description_type>::value;
            using dummy_default_fft_size          = Size<2>;
            using this_fft_size                   = get_or_default_t<fft_operator::size, description_type, dummy_default_fft_size>;
            static constexpr auto this_fft_size_v = this_fft_size::value;

            // Type (C2C, C2R, R2C)
            // * Default value: C2C
            static constexpr bool has_type        = has_operator<fft_operator::type, description_type>::value;
            using default_fft_type                = Type<fft_type::c2c>;
            using this_fft_type                   = get_or_default_t<fft_operator::type, description_type, default_fft_type>;
            static constexpr auto this_fft_type_v = this_fft_type::value;

            // Real FFT Options (Natural, Packed, Full)
            // * Default value: Natural
            static constexpr bool has_real_fft_options   = has_operator<fft_operator::real_fft_options, description_type>::value;
            using default_real_fft_options               = RealFFTOptions<complex_layout::natural, real_mode::normal>;
            using this_real_fft_options                  = get_or_default_t<fft_operator::real_fft_options, description_type, default_real_fft_options>;
            static constexpr auto this_fft_complex_layout_v = this_real_fft_options::layout;
            static constexpr auto this_fft_real_mode_v   = this_real_fft_options::mode;

            // Direction
            // * Default value: NONE
            // * Direction can be deduced from FFT Type
            // * If there is no direction and we can't deduced it, dummy direction is FORWARD.This is required so
            // this_fft_direction_v does not break.
            // * Values of has_size or is_complete should be checked before using this property.
            static constexpr bool has_direction = has_operator<fft_operator::direction, description_type>::value;
            using deduced_fft_direction         = deduce_direction_type_t<this_fft_type>;
            using dummy_default_fft_direction   = Direction<fft_direction::forward>;
            using this_fft_direction =
                get_or_default_t<fft_operator::direction,
                                 description_type,
                                 CUFFTDX_STD::conditional_t<!CUFFTDX_STD::is_void<deduced_fft_direction>::value,
                                                            deduced_fft_direction,
                                                            dummy_default_fft_direction>>;
            static constexpr auto this_fft_direction_v = this_fft_direction::value;

            // Precision
            // * Default: float

            static constexpr bool has_precision   = has_operator<fft_operator::precision, description_type>::value;
            using this_fft_precision =
                get_or_default_t<fft_operator::precision, description_type, default_fft_precision_operator>;
            using this_fft_precision_t = typename this_fft_precision::type;

            // True if description is complete FFT description
            static constexpr bool is_complete = is_complete_description<description_type>::value;

            // SM
            static constexpr bool has_sm        = has_operator<fft_operator::sm, description_type>::value;
            using dummy_default_fft_sm          = SM<700>;
            using this_fft_sm                   = get_or_default_t<fft_operator::sm, description_type, dummy_default_fft_sm>;
            static constexpr auto this_fft_sm_v = this_fft_sm::value;

            /// ---- Constraints

            // Not-implemented-yet / disabled features

            static constexpr bool has_block_dim = has_operator<fft_operator::block_dim, description_type>::value;
#ifndef CUFFTDX_DETAIL_TEST_ENABLE_BLOCKDIM
            static_assert(!has_block_dim, "BlockDim<> feature is not implemented yet");
#endif

            // We can only have one of each option

            // Main operators
            static constexpr bool has_one_direction =
                has_at_most_one_of<fft_operator::direction, description_type>::value;
            static constexpr bool has_one_precision =
                has_at_most_one_of<fft_operator::precision, description_type>::value;
            static constexpr bool has_one_size = has_at_most_one_of<fft_operator::size, description_type>::value;
            static constexpr bool has_one_sm   = has_at_most_one_of<fft_operator::sm, description_type>::value;
            static constexpr bool has_one_type = has_at_most_one_of<fft_operator::type, description_type>::value;

            static_assert(has_one_direction, "Can't create FFT with two Direction<> expressions");
            static_assert(has_one_precision, "Can't create FFT with two Precision<> expressions");
            static_assert(has_one_size, "Can't create FFT with two Size<> expressions");
            static_assert(has_one_sm, "Can't create FFT with two SM<> expressions");
            static_assert(has_one_type, "Can't create FFT with two Type<> expressions");

            // Block-only operators
            static constexpr bool has_one_ept =
                has_at_most_one_of<fft_operator::elements_per_thread, description_type>::value;
            static constexpr bool has_one_fpb =
                has_at_most_one_of<fft_operator::ffts_per_block, description_type>::value;
            static constexpr bool has_one_block_dim =
                has_at_most_one_of<fft_operator::block_dim, description_type>::value;

            static_assert(has_one_ept, "Can't create FFT with two ElementsPerThread<> expressions");
            static_assert(has_one_fpb, "Can't create FFT with two FFTsPerBlock<> expressions");
            static_assert(has_one_block_dim, "Can't create FFT with two BlockDim<> expressions");

            // Mutually exclusive options
            static constexpr bool c2r_type_forward_dir =
                !has_direction || !(CUFFTDX_STD::is_same<this_fft_type, Type<fft_type::c2r>>::value &&
                                    CUFFTDX_STD::is_same<this_fft_direction, Direction<fft_direction::forward>>::value);
            static constexpr bool r2c_type_inverse_dir =
                !has_direction || !(CUFFTDX_STD::is_same<this_fft_type, Type<fft_type::r2c>>::value &&
                                    CUFFTDX_STD::is_same<this_fft_direction, Direction<fft_direction::inverse>>::value);

            static_assert(c2r_type_forward_dir, "Can't create Complex-to-Real FFT with forward direction");
            static_assert(r2c_type_inverse_dir, "Can't create Real-to-Complex FFT with inverse direction");

            // If size is odd and real FFT is being performed, packed layout does not make sense
            static constexpr bool odd_size_and_packed_complex_layout =
                !has_real_fft_options ||
                !has_size ||
                (this_fft_size_v % 2 == 0) || (real_fft_layout_of<description_type>::value != complex_layout::packed);

            static_assert(odd_size_and_packed_complex_layout, "Can't create Real FFT with odd size and packed layout");

            // Check if the execution will happen on block or per thread
            static constexpr bool is_thread_execution = has_n_of<1, fft_operator::thread, description_type>::value;
            static constexpr bool is_block_execution  = has_n_of<1, fft_operator::block, description_type>::value;
	        static constexpr bool has_execution_operator = is_thread_execution || is_block_execution;

            // Verify fold optimization state correctness
            static constexpr bool is_power_of_2       = (this_fft_size_v > 0) && ((this_fft_size_v & (this_fft_size_v - 1)) == 0);
            static constexpr bool is_even             = (this_fft_size_v > 0) && (this_fft_size_v % 2 == 0);

            static constexpr bool is_block_fold_optimizable =
                                                        this_fft_type_v != fft_type::c2c &&
                                                        is_power_of_2 &&
                                                        (get_or_default_t<fft_operator::elements_per_thread,
                                                                          description_type,
                                                                          ElementsPerThread<0>>::value <= this_fft_size_v);

            static constexpr bool is_thread_fold_optimizable =
                                                        this_fft_type_v != fft_type::c2c &&
                                                        is_even;

            static constexpr bool block_fold_optimization_supported =
                !is_block_execution ||
                !has_real_fft_options || !has_size ||
                this_fft_real_mode_v != real_mode::folded ||
                is_block_fold_optimizable;

            static constexpr bool thread_fold_optimization_supported =
                !is_thread_execution ||
                !has_real_fft_options || !has_size ||
                this_fft_real_mode_v != real_mode::folded ||
                is_thread_fold_optimizable;

            static_assert(block_fold_optimization_supported, "To enable block fold optimization size must be a power of 2, "
                                                             "type cannot be c2c and elements per thread value has "
                                                             "to be less then size");

            static_assert(thread_fold_optimization_supported, "To enable thread fold optimization size must be even, "
                                                             "type cannot be c2c");

            static constexpr unsigned int this_sm_max_size_v = max_block_size<this_fft_precision_t>(this_fft_sm_v);
            static constexpr bool block_max_size_full_layout =
                // This may be unclear:
                // If thread execution is turned on, we do not have to check
                // this condition and so we set it to true.
		        !has_execution_operator ||
                is_thread_execution ||
                !has_precision ||
                !has_sm ||
                !has_real_fft_options ||
                real_fft_layout_of<description_type>::value != complex_layout::full ||
                this_fft_size_v <= this_sm_max_size_v;

            static_assert(block_max_size_full_layout, "Max block EPT (64 for FP16 and FP32, 32 for FP64) cannot be used with full layout");

            static constexpr unsigned int this_thread_max_size_v = max_thread_size<this_fft_precision_t>();
            static constexpr bool thread_max_ept_full_layout =
                // This may be unclear:
                // If block execution is turned on, we do not have to check
                // this condition and so we set it to true.
		        !has_execution_operator ||
                is_block_execution ||
                real_fft_layout_of<description_type>::value != complex_layout::full ||
                this_fft_size_v <= this_thread_max_size_v;

            static_assert(thread_max_ept_full_layout, "Thread FFT sizes above complex-to-complex limit cannot be used with full layout");
            /// ---- End of Constraints
        };

        template<>
        class fft_description<>: public commondx::detail::description_expression
        {
        };
    } // namespace detail
} // namespace cufftdx

#endif // CUFFTDX_DETAIL_FFT_DESCRIPTION_HPP
