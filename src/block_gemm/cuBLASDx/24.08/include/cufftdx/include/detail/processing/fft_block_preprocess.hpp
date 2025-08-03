#ifndef CUFFTDX_DETAIL_PROCESSING_FFT_BLOCK_PREPROCESS_HPP
#define CUFFTDX_DETAIL_PROCESSING_FFT_BLOCK_PREPROCESS_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#else
#    include <type_traits>
#endif

#include <cuda_fp16.h>

#include "fft_preprocess.hpp"

#include "postprocess_fold.hpp"
#include "preprocess_fold.hpp"
#include "../utils.hpp"

namespace cufftdx {
    namespace detail {
        // Registers API

        // C2C
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess(ComplexType* /* input */, ComplexType* /* smem */) //
            -> CUFFTDX_STD::enable_if_t<(type_of<FFT>::value == fft_type::c2c)> {
            // NOP, C2C and C2R with ept == 2 don't require any preprocess
        }

        // assymetric R2C
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess(ComplexType* input, ComplexType* /* smem */) -> CUFFTDX_STD::enable_if_t<(type_of<FFT>::value == fft_type::r2c && real_fft_mode_of<FFT>::value != real_mode::folded)> {
            // Same implementation as thread_preprocess
            preprocess<FFT>(input);
        }

        // fold-optimized R2C
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess(ComplexType* input, ComplexType* /* smem */) -> CUFFTDX_STD::enable_if_t<(type_of<FFT>::value == fft_type::r2c && real_fft_mode_of<FFT>::value == real_mode::folded)> {
            // NOP, fold-optimized R2C doesn't require any preprocess
        }

        // C2R, EPT == SIZE
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess(ComplexType* input, ComplexType* /* smem */) //
            -> CUFFTDX_STD::enable_if_t<(type_of<FFT>::value == fft_type::c2r) &&
                                        real_fft_mode_of<FFT>::value == real_mode::normal &&
                                        (FFT::elements_per_thread == size_of<FFT>::value)> {
            // Same implementation as thread_preprocess
            preprocess<FFT>(input);
        }

        // C2R, EPT < SIZE, CT assymetric
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess(ComplexType* input, ComplexType* smem) //
            -> CUFFTDX_STD::enable_if_t<!Bluestein && (type_of<FFT>::value == fft_type::c2r) &&
                                        (FFT::elements_per_thread < size_of<FFT>::value) &&
                                        real_fft_mode_of<FFT>::value != real_mode::folded> {
            using scalar_type                                       = typename ComplexType::value_type;
            [[maybe_unused]] static constexpr auto ept              = FFT::elements_per_thread;
            [[maybe_unused]] static constexpr auto fft_size         = size_of<FFT>::value;
            [[maybe_unused]] static constexpr auto real_lay         = real_fft_layout_of<FFT>::value;
            [[maybe_unused]] static constexpr bool fft_size_is_even = (fft_size % 2) == 0;


            if constexpr (real_lay == complex_layout::full) {
                // nop, since
                // shared_to_registers
                // loads all data
                // the synchronization happens
                // inside the FFT function
            } else {
                __syncthreads();
                // Move to the part of shared memory for that FFT batch
                ComplexType* smem_fft_batch = smem + (threadIdx.y * fft_size / 2);
                for (unsigned int i = 0; i < (ept / 2); i++) {
                    if (!(threadIdx.x == 0 && i == 0)) {
                        smem_fft_batch[threadIdx.x + (i * (fft_size / ept)) - 1] = input[i];
                    }
                }

                if (!fft_size_is_even) {
                    constexpr unsigned int i     = ept / 2;
                    unsigned int           index = threadIdx.x + (i * (fft_size / ept)) - 1;
                    if (index < (fft_size / 2)) {
                        smem_fft_batch[index] = input[i];
                    }
                }

                __syncthreads();

                const unsigned int reversed_thread_id = (fft_size / ept) - threadIdx.x;
                const unsigned int loop_upper         = ((threadIdx.x == 0) && fft_size_is_even) ? (ept / 2) - 1 : ept / 2;
                for (unsigned int i = 0; i < ept / 2; i++) {
                    if (i < loop_upper) {
                        input[ept - 1 - i] = smem_fft_batch[reversed_thread_id + (i * (fft_size / ept)) - 1];
                        // conjugate
                        input[ept - 1 - i].y = -input[ept - 1 - i].y;
                    }
                }

                if constexpr (real_lay == complex_layout::natural && !fft_size_is_even) {
                    constexpr unsigned int i     = ept / 2;
                    unsigned int           index = reversed_thread_id + (i * (fft_size / ept)) - 1;
                    if (index < (fft_size / 2)) {
                        input[i] = smem_fft_batch[index];
                        // conjugate
                        input[i].y = -input[i].y;
                    }
                }

            }

            if(threadIdx.x == 0) {
                if constexpr (real_lay == complex_layout::packed) {
                    input[ept / 2].x = input[0].y;
                }
                if constexpr(fft_size_is_even) {
                    input[ept / 2].y = get_zero<typename ComplexType::value_type>();
                }
                input[0].y       = get_zero<typename ComplexType::value_type>();
            }
        }

        // C2R, EPT < SIZE, CT fold-optimized
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess(ComplexType* input, ComplexType* smem) //
            -> CUFFTDX_STD::enable_if_t<!Bluestein && (type_of<FFT>::value == fft_type::c2r) &&
                                        (FFT::elements_per_thread <= size_of<FFT>::value) &&
                                        real_fft_mode_of<FFT>::value == real_mode::folded> {
            preprocess_fold_c2r<FFT>(input, smem);
        }

        // C2R, EPT < SIZE, Bluestein
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess(ComplexType* input, ComplexType* smem) //
            -> CUFFTDX_STD::enable_if_t<Bluestein && (type_of<FFT>::value == fft_type::c2r) &&
                                        (FFT::elements_per_thread < get_bluestein_size(size_of<FFT>::value))> {
            using scalar_type                                            = typename ComplexType::value_type;
            [[maybe_unused]] static constexpr auto         ept           = FFT::elements_per_thread;
            [[maybe_unused]] static constexpr auto         fft_size      = size_of<FFT>::value;
            [[maybe_unused]] static constexpr auto         fft_blue_size = get_bluestein_size(fft_size);
            [[maybe_unused]] static constexpr unsigned int stride        = fft_blue_size / ept;
            [[maybe_unused]] static constexpr bool is_packed = real_fft_layout_of<FFT>::value == complex_layout::packed;

            // max_meaningful_ept limits number of loops
            [[maybe_unused]] static constexpr unsigned int max_meaningful_ept = (fft_size / 2 + 1 + (stride - 1)) / stride;

            // Move to the part of shared memory for that FFT batch
            [[maybe_unused]] ComplexType* smem_fft_batch = smem + (threadIdx.y * (fft_blue_size / 2));

            if constexpr (real_fft_layout_of<FFT>::value == complex_layout::full) {
                // nop, since
                // shared_to_registers
                // loads all data
                // synchronization happens
                // inside the FFT function
            } else {
                __syncthreads();

                for (unsigned i = 0; i < max_meaningful_ept /*ept/2*/; i++) {
                    unsigned index = (i * stride) + threadIdx.x;
                    if (index < (fft_size / 2 + !is_packed)) {
                        if (!(threadIdx.x == 0 && i == 0)) {
                            smem_fft_batch[index - 1] = input[i];
                        }
                    }
                }

                if constexpr (is_packed) {
                    if (threadIdx.x == 0) {
                        smem_fft_batch[fft_size / 2 - 1].x = input[0].y;
                    }
                }

                __syncthreads();


                // max_meaningful_ept_2 limits number of loops
                static constexpr unsigned int max_meaningful_ept_2 =
                    ept > (2 * max_meaningful_ept) ? ept : (2 * max_meaningful_ept);
                for (unsigned i = (max_meaningful_ept - 1); i < max_meaningful_ept_2; i++) {
                    unsigned int index = (i * stride) + threadIdx.x;
                    if ((index >= (fft_size / 2 + 1)) && (index < fft_size)) {
                        input[i] = smem_fft_batch[(fft_size - index) - 1];
                        // conjugate
                        input[i].y = -input[i].y;
                    }
                }
            }

            // Zero out elements required to be 0
            // to ensure hermitian input
            if (threadIdx.x == 0) {
                input[0].y = get_zero<scalar_type>();
            }

            if ((fft_size % 2 == 0) && threadIdx.x == (fft_size / 2 % stride)) {
                if constexpr (is_packed) {
                    input[max_meaningful_ept - 1] = smem_fft_batch[fft_size / 2 - 1];
                }
                input[max_meaningful_ept - 1].y = get_zero<scalar_type>();
            }
        }

        // Shared memory API

        // C2C
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess_shared_api(ComplexType* /* input */, ComplexType* /* smem */) //
            -> CUFFTDX_STD::enable_if_t<(type_of<FFT>::value == fft_type::c2c)> {
            // NOP, C2C and C2R with ept == 2 don't require any preprocess
        }

        // non-optimized R2C
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess_shared_api(ComplexType* input, ComplexType* /* smem */) //
            -> CUFFTDX_STD::enable_if_t<(type_of<FFT>::value == fft_type::r2c) &&
                                        real_fft_mode_of<FFT>::value != real_mode::folded> {
            // Same implementation as thread_preprocess
            preprocess<FFT>(input);
        }

        // fold-optimized R2C
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess_shared_api(ComplexType* input, ComplexType* /* smem */) //
            -> CUFFTDX_STD::enable_if_t<(type_of<FFT>::value == fft_type::r2c) &&
                                        real_fft_mode_of<FFT>::value == real_mode::folded> {
            // NOP, fold-optimized R2C doesn't require any preprocess
        }

        // C2R, EPT == SIZE
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess_shared_api(ComplexType* input, ComplexType* /* smem */) //
            -> CUFFTDX_STD::enable_if_t<(type_of<FFT>::value == fft_type::c2r) &&
                                        real_fft_mode_of<FFT>::value == real_mode::normal &&
                                        (FFT::elements_per_thread == size_of<FFT>::value)> {
            // Same implementation as thread_preprocess
            preprocess<FFT>(input);
        }

        // C2R, EPT < SIZE, CT assymetric
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess_shared_api(ComplexType* input, ComplexType* smem) //
            -> CUFFTDX_STD::enable_if_t<!Bluestein && (type_of<FFT>::value == fft_type::c2r) &&
                                        (FFT::elements_per_thread < size_of<FFT>::value) &&
                                        real_fft_mode_of<FFT>::value != real_mode::folded> {
            using scalar_type                               = typename ComplexType::value_type;
            [[maybe_unused]] static constexpr auto ept      = FFT::elements_per_thread;
            [[maybe_unused]] static constexpr auto fft_size = size_of<FFT>::value;

            [[maybe_unused]] static constexpr auto real_lay = real_fft_layout_of<FFT>::value;


            if constexpr (real_lay == complex_layout::full) {
                // nop, since
                // shared_to_registers
                // loads all data
            } else {
                // Move to the part of shared memory for that FFT batch
                ComplexType* smem_fft_batch = smem + (threadIdx.y * FFT::input_length);
                for (unsigned int i = ept / 2; i < ept; i++) {
                    if (real_lay == complex_layout::packed &&
                        threadIdx.x == 0 && i == ept / 2) {
                        input[ept / 2].x = input[0].y;
                    } else {
                        auto index = threadIdx.x + i * (fft_size / ept);
                        if (index > fft_size / 2) {
                            input[i]   = smem_fft_batch[fft_size - index];
                            input[i].y = -input[i].y;
                        }
                    }
                }
            }

            if(threadIdx.x == 0) {
                input[0].y       = get_zero<typename ComplexType::value_type>();
                if(fft_size % 2 == 0) {
                    input[ept / 2].y = get_zero<typename ComplexType::value_type>();
                }
            }
        }

        // C2R, EPT < SIZE, CT fold-optimized
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess_shared_api(ComplexType* input, ComplexType* smem) //
            -> CUFFTDX_STD::enable_if_t<!Bluestein && (type_of<FFT>::value == fft_type::c2r) &&
                                        (FFT::elements_per_thread <= size_of<FFT>::value) &&
                                        real_fft_mode_of<FFT>::value == real_mode::folded> {
            preprocess_fold_c2r<FFT, true>(input, smem);
        }

        // C2R, EPT < SIZE, Bluestein
        template<class FFT, bool Bluestein, class ComplexType>
        inline __device__ auto block_preprocess_shared_api(ComplexType* input, ComplexType* smem) //
            -> CUFFTDX_STD::enable_if_t<Bluestein && (type_of<FFT>::value == fft_type::c2r) &&
                                        (FFT::elements_per_thread < get_bluestein_size(size_of<FFT>::value))> {
            using scalar_type                            = typename ComplexType::value_type;
            static constexpr auto         ept            = FFT::elements_per_thread;
            static constexpr auto         fft_size       = size_of<FFT>::value;
            static constexpr auto         fft_blue_size  = get_bluestein_size(fft_size);
            static constexpr unsigned int stride         = fft_blue_size / ept;

            [[maybe_unused]] ComplexType* smem_fft_batch = smem + (threadIdx.y * FFT::input_length);

            if constexpr (real_fft_layout_of<FFT>::value == complex_layout::full) {
                // nop, since
                // shared_to_registers
                // loads all data
            } else {
                // Move to the part of shared memory for that FFT batch
                static constexpr unsigned int first_missing_index = ((fft_size / 2 + 1) + (stride - 1)) / stride;
                static constexpr unsigned int last_missing_index =
                    ept > (2 * first_missing_index) ? ept : (2 * first_missing_index);
                for (unsigned i = (first_missing_index - 1); i < last_missing_index; i++) {
                    unsigned int index = (i * stride) + threadIdx.x;
                    if ((index >= (fft_size / 2 + 1)) && (index < fft_size)) {
                        input[i]   = smem_fft_batch[(fft_size - index)];
                        input[i].y = -input[i].y;
                    }
                }
            }

            // Zero out elements required to be 0
            // to ensure hermitian input
            if (threadIdx.x == 0) {
                input[0].y = get_zero<typename ComplexType::value_type>();
            }

            if ((fft_size % 2 == 0) && threadIdx.x == (fft_size / 2 % stride)) {
                if constexpr (real_fft_layout_of<FFT>::value == complex_layout::packed) {
                        input[fft_size / (2 * stride)]   = smem_fft_batch[fft_size / 2 - 1];
                }
                input[fft_size / (2 * stride)].y = get_zero<typename ComplexType::value_type>();
            }
        }
    } // namespace detail
} // namespace cufftdx

#endif
