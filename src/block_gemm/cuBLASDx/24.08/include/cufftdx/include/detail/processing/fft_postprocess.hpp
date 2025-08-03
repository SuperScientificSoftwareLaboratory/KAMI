#ifndef CUFFTDX_DETAIL_PROCESSING_FFT_POSTPROCESS_HPP
#define CUFFTDX_DETAIL_PROCESSING_FFT_POSTPROCESS_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#else
#    include <type_traits>
#endif

#include <cuda_fp16.h>

#include "postprocess_fold.hpp"

namespace cufftdx {
    namespace detail {
        // C2C
        template<class FFT, class ComplexType>
        inline __device__ auto postprocess(ComplexType* input) //
            -> CUFFTDX_STD::enable_if_t<(type_of<FFT>::value == fft_type::c2c)> {
            // NOP, C2C doesn't require postprocess
        }

        // R2C
        template<class FFT, class ComplexType>
        inline __device__ auto postprocess(ComplexType* input) //
            -> CUFFTDX_STD::enable_if_t<(type_of<FFT>::value == fft_type::r2c &&
                                         real_fft_mode_of<FFT>::value != real_mode::folded)> {
            [[maybe_unused]] static constexpr auto ept = FFT::effective_elements_per_thread;

            if constexpr (real_fft_layout_of<FFT>::value == complex_layout::full) {
                // already is in full layout
            } else if constexpr (real_fft_layout_of<FFT>::value == complex_layout::natural) {
                // implicitly already in natural layout (if the rest is ignored)
            } else if constexpr (real_fft_layout_of<FFT>::value == complex_layout::packed) {
                // we need to pack real part of element N/2 into imaginary part of element 0
                if constexpr (size_of<FFT>::value > FFT::elements_per_thread) {
                    if (threadIdx.x == 0) {
                        input[0].y = input[ept / 2].x;
                    }
                } else {
                    input[0].y = input[ept / 2].x;
                }
            }
        }

        template<class FFT, class ComplexType>
        inline __device__ auto postprocess(ComplexType* input) //
            -> CUFFTDX_STD::enable_if_t<(type_of<FFT>::value == fft_type::r2c &&
                                         real_fft_mode_of<FFT>::value == real_mode::folded)> {
            postprocess_fold_r2c<FFT>(input, static_cast<ComplexType*>(nullptr));
        }

        // C2R
        template<class FFT, class ComplexType>
        inline __device__ auto postprocess(ComplexType* input) //
            -> CUFFTDX_STD::enable_if_t<(type_of<FFT>::value == fft_type::c2r &&
                                         real_fft_mode_of<FFT>::value != real_mode::folded)> {
            using scalar_type         = typename ComplexType::value_type;
            static constexpr auto ept = FFT::elements_per_thread;

            // Pack real values
            for (unsigned int i = 1; i < ept; i++) {
                (reinterpret_cast<scalar_type*>(input))[i] = (reinterpret_cast<scalar_type*>(input))[2 * i];
            }
        }

        // C2R
        template<class FFT, class ComplexType>
        inline __device__ auto postprocess(ComplexType* input) //
            -> CUFFTDX_STD::enable_if_t<(type_of<FFT>::value == fft_type::c2r &&
                                         real_fft_mode_of<FFT>::value == real_mode::folded)> {
            // NOP
        }
    } // namespace detail
} // namespace cufftdx

#endif
