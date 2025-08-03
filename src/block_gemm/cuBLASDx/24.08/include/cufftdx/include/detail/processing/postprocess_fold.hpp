#ifndef CUFFTDX_DETAIL_PROCESSING_POSTPROCESS_FOLD_HPP
#define CUFFTDX_DETAIL_PROCESSING_POSTPROCESS_FOLD_HPP

#include "lut.hpp"
#include "../utils.hpp"


namespace cufftdx {
    namespace detail {

        // Helper structure allowing to roll all execution properties into
        // a single interface
        template<typename FFT>
        struct post {
            using compute_t                                 = typename FFT::value_type;
            using simple_t                                  = typename compute_t::value_type;
            static constexpr unsigned int ept               = FFT::effective_elements_per_thread;
            static constexpr unsigned int half_ept          = (ept % 2 == 0) ? (ept / 2) : (ept / 2 + 1);
            static constexpr unsigned int complex_fft_size  = size_of<FFT>::value / 2;
            static constexpr unsigned int half_complex_size = (complex_fft_size % 2 == 0) ? (complex_fft_size / 2) : (complex_fft_size / 2 + 1);
            static constexpr unsigned int stride            = complex_fft_size / ept;
            static constexpr bool         is_block          = size_of<FFT>::value > FFT::elements_per_thread;

            // Decide whether it's worth to reorder first and middle elements
            // and simulate their symmetricity. This allows to get rid of
            // conditional processing from unrolled loops and get a more
            // performant execution.
            static __device__ constexpr bool
            needs_loop_reorder() {
                constexpr auto size       = size_of<FFT>::value;
                using precision           = typename precision_of<FFT>::type;
                constexpr bool is_natural = real_fft_layout_of<FFT>::value == complex_layout::natural;

                // The first and middle element need an extra register,
                // which is available only in the natural layout.
                if constexpr (!is_natural) {
                    return false;
                }

                // FP16 R2C is always faster with conditional processing
                constexpr bool fp16_case = false;

                // Sizes for which conditional processing inside loop slows down execution for FP32 precision
                constexpr bool fp32_case = CUFFTDX_STD::is_same_v<precision, float> && (size == 128 || size == 65536);

                // Sizes for which conditional processing inside loop slows down execution for FP64 precision
                constexpr bool fp64_case = CUFFTDX_STD::is_same_v<precision, double> && (size == 32768);

                return fp16_case || fp32_case || fp64_case;
            }

            static constexpr bool loop_reorder = needs_loop_reorder();

            // Only first iteration of the processing loop needs the conditional
            // statement inside. To allow for efficient unrolling it's better
            // to extract it and perform all remaining iterations uniformly.
            // This can be thought of as manual unrolling.
            // ===
            // Used only when needs_loop_reorder() returns false
            static __forceinline__ __device__ void
            process_first_and_middle_elements(compute_t* rmem) {
                if constexpr (real_fft_layout_of<FFT>::value == complex_layout::natural ||
                              real_fft_layout_of<FFT>::value == complex_layout::full) {
                    // Since the elements at indices 0 and N/2 will have their sine twiddle
                    // parts equal to 0, they do not follow the regular symmetry and need
                    // to be treated differently from the rest
                    rmem[ept].x       = rmem[0].x - rmem[0].y;
                    rmem[ept].y       = get_zero<simple_t>();

                    rmem[0].x         = rmem[0].x + rmem[0].y;
                    rmem[0].y         = get_zero<simple_t>();

                    if constexpr(ept % 2 == 0) {
                        rmem[half_ept].y = -rmem[half_ept].y;
                    }
                } else if constexpr (real_fft_layout_of<FFT>::value == complex_layout::packed) {
                    // This follows the ::natural approach, but instead of storing these two
                    // values separately it packs them together into rmem[0]
                    rmem[0]         = compute_t {rmem[0].x + rmem[0].y, rmem[0].x - rmem[0].y};
                    if constexpr(ept % 2 == 0) {
                        rmem[half_ept].y = -rmem[half_ept].y;
                    }
                }
            }

            // Use the hermicity of an R2C FFT output and a basic Cooley-Tukey
            // butterfly to combine elements of 2 merged FFTs of length N/2 into results
            // of a single FFT of length N
            static __device__ __forceinline__ void symmetric_process(compute_t& v0, compute_t& v1, const compute_t& twiddle) {
                const simple_t A1 = get_val<simple_t>(0.5) * (v0.x + v1.x);
                const simple_t B2 = get_val<simple_t>(0.5) * (v0.x - v1.x);
                const simple_t A2 = get_val<simple_t>(0.5) * (v0.y + v1.y);
                const simple_t B1 = get_val<simple_t>(0.5) * (v0.y - v1.y);

                v0.x = A1 + (A2 * twiddle.x + B2 * twiddle.y);
                v1.x = A1 - (A2 * twiddle.x + B2 * twiddle.y);
                v0.y = (A2 * twiddle.y - B2 * twiddle.x) + B1;
                v1.y = (A2 * twiddle.y - B2 * twiddle.x) - B1;
            }

            // Duplicate and swap first and middle elements
            // to allow for simulating their symmetricity.
            // ===
            // Used only in cases enumerated in: needs_loop_reorder()
            static __forceinline__ __device__ void
            duplicate_first_element(compute_t* rmem) {
                // If there is only 1 thread performing the FFT
                // it's unnecessary to check its index. Otherwise
                // only the first of each threadgroup needs to
                // perform this swap.
                if (!is_block || threadIdx.x == 0) {
                    // Middle element to the end of rmem, to be processed later.
                    // Putting 0th element in its place.
                    rmem[ept]     = rmem[half_ept];
                    rmem[half_ept] = rmem[0];
                }
            }

            // Change the data layout in each thread from canonically strided:
            // a[idx], a[idx + stride], a[idx + 2 * stride] ... a[idx + EPT * stride]
            // to symmetric:
            // a[idx], a[idx + stride] ... a[N - idx - stride], a[N - idx]
            // to allow for in-thread processing of symmetric elements
            static __forceinline__ __device__ void
            block_reorder_strided_to_symmetric(compute_t* rmem, compute_t* smem) {
                __syncthreads();
                // Exchange elements for postprocessing
                // Store 2nd half of rmem to shared
                unsigned int index = threadIdx.x;
                for (unsigned i = 0; i < half_ept; i++) {
                    smem[index] = rmem[i + half_ept];
                    index += stride;
                }
                __syncthreads();
                // Load to data from other threads to 2nd half of rmem
                index = threadIdx.x;
                for (unsigned i = 0; i < half_ept; i++) {
                    const unsigned mirrored_index = index ? (half_complex_size - index) : 0;
                    rmem[i + half_ept]             = smem[mirrored_index];
                    index += stride;
                }
            }

            // After duplicating first element before, now
            // swap the second copy with the last element,
            // conjugating the latter to postprocess it
            // ===
            // Used only in cases enumerated in: needs_loop_reorder()
            static __forceinline__ __device__ void
            swap_and_postprocess_last_element(compute_t* rmem) {
                if (!is_block || (threadIdx.x == 0)) {
                    // Middle element processing:
                    // This is equivalent of doing postprocess(rmem[ept], rmem[ept], twiddle)
                    // with twiddle equal to (0.0, -1.0).
                    rmem[ept].y = -rmem[ept].y;

                    // Switch back
                    compute_t tmp = rmem[ept];
                    rmem[ept]     = rmem[half_ept];
                    rmem[half_ept] = tmp;
                }
            }

            // Change the data layout from symmetric in each thread:
            // a[idx], a[idx + stride] ... a[N - idx - stride], a[N - idx]
            // to canonically strided:
            // a[idx], a[idx + stride], a[idx + 2 * stride] ... a[idx + EPT * stride]
            // to allow for in-thread processing of symmetric elements
            static __forceinline__ __device__ void
            block_reorder_symmetric_to_strided(compute_t* rmem, compute_t* smem) {
                // Multiple Thread Symmetrical -> Canonically Strided
                // Will use Shared memory for inter-thread exchanges
                __syncthreads();
                // Exchange elements back before storing to global
                // Store 2nd half of rmem to shared
                unsigned int index = threadIdx.x;
                for (unsigned i = 0; i < half_ept; i++) {
                    const unsigned mirrored_index = index ? (half_complex_size - index) : 0;
                    smem[mirrored_index]          = rmem[i + half_ept];
                    index += stride;
                }
                __syncthreads();
                // Load to data from other threads to 2nd half of rmem
                index = threadIdx.x;
                for (unsigned i = 0; i < half_ept; i++) {
                    rmem[i + half_ept] = smem[index];
                    index += stride;
                }
                // If the layout is full it is needed to duplicate and conjugate the
                // data in the entire threadgroup, to get N final elements from N/2 + 1
                // FFT results
                if constexpr (real_fft_layout_of<FFT>::value == complex_layout::full) {
                    // Duplicate data already in shared memory,
                    // indexed N/4 to N/2 - 1, where N is full FFT size
                    index = threadIdx.x;
                    for (unsigned i = 0; i < half_ept; i++) {
                        if (threadIdx.x == 0 && i == 0) {
                            rmem[i + ept + half_ept]   = smem[0];
                            rmem[i + ept + half_ept].y = -rmem[i + ept + half_ept].y;
                        } else {
                            rmem[i + ept]   = smem[half_complex_size - index];
                            rmem[i + ept].y = -rmem[i + ept].y;
                        }
                        index += stride;
                    }
                    __syncthreads();
                    // Store first ept/2 registers into shared memory,
                    // elements indexed 0 to N/4 - 1, where N is full FFT size
                    index = threadIdx.x;
                    for (unsigned i = 0; i < half_ept; i++) {
                        smem[index] = rmem[i];
                        index += stride;
                    }
                    __syncthreads();
                    // Read and conjugate the elements from previous step
                    // After this loop all threads contain 2 * EPT elements,
                    // N in total (since EPT * BLOCKDIMX == N/2, not N)
                    index = threadIdx.x;
                    for (unsigned i = 0; i < half_ept; i++) {
                        if (threadIdx.x > 0 || i > 0) {
                            rmem[i + half_ept + ept]   = smem[half_complex_size - index];
                            rmem[i + ept + half_ept].y = -rmem[i + ept + half_ept].y;
                        }
                        index += stride;
                    }
                }
            }

            // If the layout is full it is needed to duplicate and conjugate the
            // register data, 2*EPT in total (since EPT == N/2, not N)
            // a1 a2 a3 a4 -> a1 a2 a3 a4 conj(a3) conj(a2)
            static __forceinline__ __device__ void
            thread_append_hermitian_half(compute_t* rmem) {
                for (int i = 0; i < ept - 1; ++i) {
                    rmem[ept + 1 + i]   = rmem[ept - 1 - i];
                    rmem[ept + 1 + i].y = -rmem[ept + 1 + i].y;
                }
            }
        };

        // [Postprocess]
        //
        // N/2 complex results are postprocessed into {(N/2 + 1) / (N/2) / (N)} results of R2C FFT.
        // Output length depends on chosen complex layout:
        // natural -> N/2 + 1 complex outputs (first and middle elements must have 0 in their imaginary parts)
        // packed -> N/2 complex inputs (first and middle elements combined in 0th index)
        // full -> N complex inputs (fully unrolled hermitian sequence)
        //
        // We need to pair first half with it's mirrored reflection, and apply postprocess
        // to them. 0th and middle elements are the exceptions in the process:
        // * 0th element should be paired with itself and saved to 0th and last position.
        // * Middle (8) element should be paired with itself and single results should be saved to middle position.
        //
        // Example for natural layout:
        // nx = 32, compute_nx = 16, ept = 4
        // 0: postprocess(value0, value0)  -> save(value0, 0), save(value0,  compute_nx-0=16);
        // 0: postprocess(value1, value15) -> save(value1, 1), save(value15, compute_nx-1=15);
        // 1: postprocess(value2, value14) -> save(value2, 2), save(value14, compute_nx-2=14);
        // 1: postprocess(value3, value13) -> save(value3, 3), save(value13, compute_nx-3=13);
        // 2: postprocess(value4, value12) -> save(value4, 4), save(value12, compute_nx-4=12);
        // 2: postprocess(value5, value11) -> save(value5, 5), save(value11, compute_nx-5=11);
        // 3: postprocess(value6, value10) -> save(value6, 6), save(value10, compute_nx-6=10);
        // 3: postprocess(value7, value9)  -> save(value7, 7), save(value9,  compute_nx-7=9);
        // 0: postprocess(value8, value8)  -> save(value8, 8)
        template<class FFT, class ComplexType>
        inline __device__ auto postprocess_fold_r2c(ComplexType* rmem, ComplexType* smem) -> CUFFTDX_STD::enable_if_t<size_of<FFT>::value != FFT::elements_per_thread> {
            ComplexType  twiddle;
            unsigned int index = threadIdx.x;

            const unsigned int shared_offset = threadIdx.y * post<FFT>::half_complex_size;
            smem += shared_offset;

            // Shuffle data through shared memory so that each thread
            // contains symmetric values, ready for postprocessing
            post<FFT>::block_reorder_strided_to_symmetric(rmem, smem);

            // This needs to be performed only when we
            // are simulating symmetricity for the first
            // and middle elements.
            // ===
            // Used only in cases enumerated in: needs_loop_reorder()
            if constexpr (post<FFT>::loop_reorder) {
                post<FFT>::duplicate_first_element(rmem);
            } else {
                if (threadIdx.x == 0) {
                    post<FFT>::process_first_and_middle_elements(rmem);
                } else {
                    twiddle = detail::compute_twiddle<typename post<FFT>::simple_t>(index, size_of<FFT>::value);
                    post<FFT>::symmetric_process(rmem[0], rmem[post<FFT>::half_ept], twiddle);
                }
                index += post<FFT>::stride;
            }

            // Postprocess all elements, applying the properties of real valued FFT
            // (hermicity of output) and basic Cooley-Tukey butterfly, to get length-N
            // FFT output from 2 batched length N/2 FFTs
            for (unsigned i = (post<FFT>::loop_reorder ? 0 : 1); i < post<FFT>::half_ept; i++) {
                twiddle = detail::compute_twiddle<typename post<FFT>::simple_t>(index, size_of<FFT>::value);
                post<FFT>::symmetric_process(rmem[i], rmem[i + post<FFT>::half_ept], twiddle);
                index += post<FFT>::stride;
            }

            // This needs to be performed only when we
            // are simulating symmetricity for the first
            // and middle elements.
            // ===
            // Used only in cases enumerated in: needs_loop_reorder()
            if constexpr (post<FFT>::loop_reorder) {
                post<FFT>::swap_and_postprocess_last_element(rmem);
            }

            // Shuffle data through shared memory so that each thread
            // contains values in canonical strided order, expected as
            // output after postprocessing
            post<FFT>::block_reorder_symmetric_to_strided(rmem, smem);
        }

        // Single thread
        template<class FFT, class ComplexType>
        inline __device__ void postprocess_fold_r2c(ComplexType* rmem) {
            ComplexType twiddle;

            post<FFT>::process_first_and_middle_elements(rmem);

            // Postprocess all elements, applying the properties of real valued FFT
            // (hermicity of output) and basic Cooley-Tukey butterfly, to get length-N
            // FFT output from 2 batched length N/2 FFTs
            for (unsigned i = 1; i < post<FFT>::half_ept; i++) {
                const unsigned mirrored_index = post<FFT>::complex_fft_size - i;
                twiddle                       = detail::compute_twiddle<typename post<FFT>::simple_t>(i, size_of<FFT>::value);
                post<FFT>::symmetric_process(rmem[i], rmem[mirrored_index], twiddle);
            }

            // If the layout is full it is needed to duplicate and conjugate the
            // register data to get N elements from N/2 [+1]
            if constexpr (real_fft_layout_of<FFT>::value == complex_layout::full) {
                post<FFT>::thread_append_hermitian_half(rmem);
            }
        }

        // Redirect to function which explicitly does not use shared memory
        template<class FFT, class ComplexType>
        inline __device__ auto postprocess_fold_r2c(ComplexType* rmem, ComplexType* /*smem*/) -> CUFFTDX_STD::enable_if_t<size_of<FFT>::value == FFT::elements_per_thread> {
            postprocess_fold_r2c<FFT>(rmem);
        }
    } // namespace detail
} // namespace cufftdx

#endif // CUFFTDX_DETAIL_POSTPROCESS_FOLD_HPP
