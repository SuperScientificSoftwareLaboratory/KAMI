#ifndef CUFFTDX_DETAIL_PROCESSING_FFT_PREPROCESS_HPP
#define CUFFTDX_DETAIL_PROCESSING_FFT_PREPROCESS_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#else
#    include <type_traits>
#endif

#include <cuda_fp16.h>

#include "../utils.hpp"

#include "preprocess_fold.hpp"

namespace cufftdx {
    namespace detail {

        // C2C
        template<class FFT, class ComplexType>
        inline __device__ auto preprocess(ComplexType* /* input */) //
            -> CUFFTDX_STD::enable_if_t<(type_of<FFT>::value == fft_type::c2c)> {
            // NOP, C2C and C2R with ept == 2 don't require any preprocess
        }

        // C2R AND ept == 2
        template<class FFT, class ComplexType>
        inline __device__ auto preprocess(ComplexType* input)                      //
            -> CUFFTDX_STD::enable_if_t<((type_of<FFT>::value == fft_type::c2r) && //
                                         (FFT::elements_per_thread == 2) &&
                                         real_fft_mode_of<FFT>::value != real_mode::folded)> {
            if constexpr (real_fft_layout_of<FFT>::value == complex_layout::packed) {
                input[1].x = input[0].y;
                input[0].y = input[1].y = get_zero<typename ComplexType::value_type>();
            }
        }

        // R2C AND ept > 2
        template<class FFT, class ComplexType>
        inline __device__ auto preprocess(ComplexType* input) //
            -> CUFFTDX_STD::enable_if_t<(type_of<FFT>::value == fft_type::r2c &&
                                         real_fft_mode_of<FFT>::value != real_mode::folded)> {
            using scalar_type         = typename ComplexType::value_type;
            static constexpr auto ept = FFT::elements_per_thread;

            // Input has packed real values (this means .y has real values), this
            // unpacks input so every complex value is {real, 0}
            for (unsigned int i = ept; i > 1; i--) {
                (reinterpret_cast<scalar_type*>(input))[2 * i - 1] = get_zero<scalar_type>();
                (reinterpret_cast<scalar_type*>(input))[2 * i - 2] = (reinterpret_cast<scalar_type*>(input))[i - 1];
            }

            // input[0].x is in the right position from the start, just need to set .y to zero
            input[0].y = get_zero<scalar_type>();
        }

        // R2C AND ept > 2
        template<class FFT, class ComplexType>
        inline __device__ auto preprocess(ComplexType* input) //
            -> CUFFTDX_STD::enable_if_t<(type_of<FFT>::value == fft_type::r2c &&
                                         real_fft_mode_of<FFT>::value == real_mode::folded)> {
            // NOP
        }

        // C2R AND ept > 2
        template<class FFT, class ComplexType>
        inline __device__ auto preprocess(ComplexType* input) //
            -> CUFFTDX_STD::enable_if_t<(type_of<FFT>::value == fft_type::c2r) &&
                                        (FFT::elements_per_thread > 2) &&
                                        real_fft_mode_of<FFT>::value != real_mode::folded> {
            using scalar_type                               = typename ComplexType::value_type;
            constexpr auto                         real_lay = real_fft_layout_of<FFT>::value;
            [[maybe_unused]] static constexpr auto ept      = FFT::elements_per_thread;

            // If ept is even we need to fill one value less
            if constexpr (real_lay == complex_layout::natural) {
                for (unsigned int i = 0; i < (ept + 1) / 2 - 1; ++i) {
                    input[ept - 1 - i] = input[i + 1];
                    // conjugate
                    input[ept - 1 - i].y = -input[ept - 1 - i].y;
                }
            } else if constexpr (real_lay == complex_layout::packed) {
                input[ept / 2].x = input[0].y;
                input[0].y = input[ept / 2].y = get_zero<scalar_type>();
                for (unsigned int i = 0; i < (ept + 1) / 2 - 1; ++i) {
                    input[ept - 1 - i] = input[i + 1];
                    // conjugate
                    input[ept - 1 - i].y = -input[ept - 1 - i].y;
                }
            } else if constexpr (real_lay == complex_layout::full) {
                // nop
            }

            input[0].y = get_zero<scalar_type>();
            if(ept % 2 == 0) {
                input[ept / 2].y = get_zero<scalar_type>();
            }
        }

        // C2R AND ept > 2
        template<class FFT, class ComplexType>
        inline __device__ auto preprocess(ComplexType* input) //
            -> CUFFTDX_STD::enable_if_t<(type_of<FFT>::value == fft_type::c2r) &&
                                        (FFT::elements_per_thread > 2) &&
                                        real_fft_mode_of<FFT>::value == real_mode::folded> {
            preprocess_fold_c2r<FFT>(input);
        }
    } // namespace detail
} // namespace cufftdx

#endif
