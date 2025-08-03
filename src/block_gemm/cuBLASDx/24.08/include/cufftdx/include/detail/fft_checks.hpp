// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_DETAIL_FFT_CHECKS_HPP
#define CUFFTDX_DETAIL_FFT_CHECKS_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#else
#    include <type_traits>
#endif

#include <cuda_fp16.h>

#include "fft_sizes.hpp"
#include "../operators.hpp"
#include "../traits/detail/bluestein_helpers.hpp"
#include "../traits/fft_traits.hpp"

namespace cufftdx {
    namespace detail {
        // LoosenRestriction is a bool flag signaling that we suspect that the description may not be
        // complete yet in a way which influences the maximal sizes. It is passed as 'true' whenever
        // there is no Type<> operator, or if the type is R2C/C2R and there is no RealFFTOptions
        // operator, since if RealFFTOptions is passed with real_mode::folded set as the
        // second argument, the size posssible to be executed is doubled in effect.
        //
        // Provision of LoosenRestriction flag allows such construction
        // using FFT_base = decltype(Block() + Size<MAX_R2C_SIZE>()) ---> MAX_R2C_SIZE is C2C unsupported
        // using FFT_r2c = decltype(FFT_base() + Type<fft_type::r2c>() + RealFFTOptions<normal, folded>())
        // using FFT_c2r = decltype(FFT_base() + Type<fft_type::c2r>() + RealFFTOptions<normal, folded>())
        template<class Precision, unsigned int Size, unsigned int Arch, bool LoosenRestriction = false>
        class is_supported
        {
            // architecture with biggest allowed sizes
            static constexpr unsigned int max_arch = 900;
            // Max supported sizes, if -1 passed then SM is ignored and global max size is checked
            static constexpr auto effective_arch            = (Arch == unsigned(-1)) ? max_arch : Arch;
            static constexpr auto effective_max_blue_size   = (LoosenRestriction ? 2 : 1) * max_block_size<double>(effective_arch);
            static constexpr auto effective_max_thread_size = (LoosenRestriction ? 2 : 1) * max_thread_size<Precision>();
            static constexpr auto blue_size                 = detail::get_bluestein_size(Size);

        public:
            static constexpr auto effective_max_block_size = (LoosenRestriction ? 2 : 1) * max_block_size<Precision>(effective_arch);
            static constexpr bool is_sm_supported          = effective_max_block_size > 0;

            static constexpr bool thread_value     = (is_sm_supported && (Size <= effective_max_thread_size) && (Size >= 2));
            static constexpr bool block_value      = ((Size <= effective_max_block_size) && (Size >= 2));
            static constexpr bool blue_block_value = ((Size == effective_max_block_size) || (blue_size <= effective_max_blue_size) && (blue_size >= 2));

            static constexpr bool value = block_value || blue_block_value;
        };

        template<class Precision,
                 class EPT,
                 real_mode RealMode,
                 class block_fft_record_t,
                 bool PresentInDatabase = block_fft_record_t::defined>
        struct is_ept_supported: public CUFFTDX_STD::false_type {
        };

        template<class Precision, class EPT, real_mode RealMode, class block_fft_record_t>
        struct is_ept_supported<Precision, EPT, RealMode, block_fft_record_t, true> {
            // Get default implementation
            using default_block_config_t =
                typename database::detail::type_list_element<0, typename block_fft_record_t::blobs>::type;
            // Get default EPT
            using default_ept = ElementsPerThread<default_block_config_t::elements_per_thread>;
// Select transposition types to look for in the database
#ifdef CUFFTDX_DETAIL_BLOCK_FFT_ENFORCE_X_TRANSPOSITION
            static constexpr unsigned int this_fft_trp_option_v = 1;
#elif defined(CUFFTDX_DETAIL_BLOCK_FFT_ENFORCE_XY_TRANSPOSITION)
            static constexpr unsigned int this_fft_trp_option_v = 2;
#else
            static constexpr unsigned int this_fft_trp_option_v = 0;
#endif
            // Search for implementation
            using this_fft_elements_per_thread =
                CUFFTDX_STD::conditional_t<!CUFFTDX_STD::is_void<EPT>::value, EPT, default_ept>;
            static constexpr auto this_fft_ept_v_full = this_fft_elements_per_thread::value;
            static constexpr auto this_fft_ept_v      = (!CUFFTDX_STD::is_void<EPT>::value &&
                                                    RealMode == real_mode::folded)
                                                            ? this_fft_ept_v_full / 2
                                                            : this_fft_ept_v_full;

            using this_fft_block_fft_implementation =
                typename database::detail::search_by_ept<this_fft_ept_v,
                                                         Precision,
                                                         this_fft_trp_option_v,
                                                         typename block_fft_record_t::blobs>::type;
            static constexpr bool implementation_exists =
                !CUFFTDX_STD::is_void<this_fft_block_fft_implementation>::value;

        public:
            static constexpr bool value = CUFFTDX_STD::is_void<EPT>::value ? true : implementation_exists;
        };

        template<class Precision,
                 fft_type      Type,
                 fft_direction Direction,
                 unsigned      Size,
                 bool          Block,
                 bool          Thread,
                 class EPT, // void if not set
                 complex_layout  RealLayout,
                 real_mode    RealMode,
                 unsigned int Arch>
        class is_supported_helper
        {
            // Checks
            static_assert(Block || Thread,
                          "To check if an FFT description is supported on a given architecture it has to have Block or "
                          "Thead execution operator");

            static constexpr auto effective_execution_size = (RealMode == real_mode::folded)
                                                                 ? Size / 2
                                                                 : Size;

            static constexpr auto effective_memory_size = (RealMode == real_mode::folded && RealLayout != complex_layout::full)
                                                              ? Size / 2
                                                              : Size;

            static constexpr auto effective_type = (RealMode == real_mode::folded)
                                                       ? cufftdx::fft_type::c2c
                                                       : Type;

            static constexpr bool is_supported_thread     = is_supported<Precision, effective_memory_size, Arch>::thread_value;
            static constexpr bool is_supported_block      = is_supported<Precision, effective_memory_size, Arch>::block_value;
            static constexpr bool is_supported_block_blue = is_supported<Precision, effective_memory_size, Arch>::blue_block_value;


            static constexpr bool requires_block_blue =
                Block && is_bluestein_required<effective_execution_size, Precision, Direction, effective_type, Arch>::value;

            // Check if EPT is supported
            using block_fft_record_t =
                cufftdx::database::detail::block_fft_record<effective_execution_size, Precision, effective_type, Direction, Arch>;

            static constexpr bool is_ept_supported_v = is_ept_supported<Precision, EPT, RealMode, block_fft_record_t>::value;

            // Check if EPT is supported
            static constexpr auto blue_size = detail::get_bluestein_size(effective_execution_size);
            using block_fft_record_blue_t =
                cufftdx::database::detail::block_fft_record<blue_size, Precision, effective_type, Direction, Arch>;
            static constexpr bool is_ept_supported_blue_v = is_ept_supported<Precision, EPT, RealMode, block_fft_record_blue_t>::value;

        public:
            static constexpr bool value =
                (Thread && is_supported_thread) ||                                                    // Thread
                (Block && is_supported_block && is_ept_supported_v) ||                                // Block
                (Block && is_supported_block_blue && requires_block_blue && is_ept_supported_blue_v); // Blue
        };
    } // namespace detail

    // Check if description is supported on given Architecture
    template<class Description, unsigned int Architecture>
    struct is_supported:
        public CUFFTDX_STD::bool_constant<
            detail::is_supported_helper<precision_of_t<Description>,
                                        type_of<Description>::value,
                                        direction_of<Description>::value,
                                        size_of<Description>::value,
                                        detail::has_operator<fft_operator::block, Description>::value,
                                        detail::has_operator<fft_operator::thread, Description>::value,
                                        detail::get_t<fft_operator::elements_per_thread, Description>,
                                        real_fft_layout_of<Description>::value,
                                        real_fft_mode_of<Description>::value,
                                        Architecture>::value> {
    };

    template<class Description, unsigned int Architecture>
    inline constexpr bool is_supported_v = is_supported<Description, Architecture>::value;
} // namespace cufftdx

#endif // CUFFTDX_DETAIL_FFT_CHECKS_HPP
