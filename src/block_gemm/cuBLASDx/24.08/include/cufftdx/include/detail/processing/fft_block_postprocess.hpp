#ifndef CUFFTDX_DETAIL_PROCESSING_FFT_BLOCK_POSTPROCESS_HPP
#define CUFFTDX_DETAIL_PROCESSING_FFT_BLOCK_POSTPROCESS_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#else
#    include <type_traits>
#endif

#include <cuda_fp16.h>

#include "fft_postprocess.hpp"

#include "postprocess_fold.hpp"
#include "preprocess_fold.hpp"

namespace cufftdx {
    namespace detail {

        // R2C for Bluestein packed
        template<class FFT, class ComplexType>
        inline __device__ auto postprocess_bluestein_r2c_packed(ComplexType* input, ComplexType* smem)
            -> CUFFTDX_STD::enable_if_t<(type_of<FFT>::value == fft_type::r2c)> {

            static constexpr auto ept = FFT::elements_per_thread;



            if constexpr (size_of<FFT>::value > ept) {
                ComplexType* smem_fft_batch = smem + threadIdx.y;
                __syncthreads();

                if (((size_of<FFT>::value / 2) % FFT::stride) == threadIdx.x) {
                    smem_fft_batch[0].x = input[((size_of<FFT>::value / 2) - threadIdx.x) / FFT::stride].x;
                }

                __syncthreads();

                if (threadIdx.x == 0) {
                    input[0].y = smem_fft_batch[0].x;
                }
            } else {
                input[0].y = input[ept / 2].x;
            }
        }

        // All non-optimized
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_postprocess(ComplexType* input, ComplexType* smem) -> CUFFTDX_STD::enable_if_t<real_fft_mode_of<FFT>::value != real_mode::folded || Bluestein> {
            // Same implementation as thread_postprocess
            if constexpr (Bluestein &&
                          real_fft_layout_of<FFT>::value == complex_layout::packed &&
                          type_of<FFT>::value == fft_type::r2c) {
                postprocess_bluestein_r2c_packed<FFT>(input, smem);
            } else {
                postprocess<FFT>(input);
            }
        }

        // fold-optimized R2C
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_postprocess(ComplexType* input, ComplexType* smem) -> CUFFTDX_STD::enable_if_t<!Bluestein && real_fft_mode_of<FFT>::value == real_mode::folded &&
                                                                                                                    (type_of<FFT>::value == fft_type::r2c)> {
            postprocess_fold_r2c<FFT>(input, smem);
        }

        // fold-optimized C2R
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_postprocess(ComplexType* input, ComplexType* /* smem */) -> CUFFTDX_STD::enable_if_t<!Bluestein && real_fft_mode_of<FFT>::value == real_mode::folded &&
                                                                                                                          (type_of<FFT>::value == fft_type::c2r)> {
            // NOP, fold-optimized C2R doesn't require any postprocess
        }
    } // namespace detail
} // namespace cufftdx

#endif
