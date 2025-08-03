#ifndef CUFFTDX_DETAIL_PROCESSING_PREPROCESS_FOLD_HPP
#define CUFFTDX_DETAIL_PROCESSING_PREPROCESS_FOLD_HPP

#include "lut.hpp"
#include "../utils.hpp"

namespace cufftdx {
    namespace detail {

        // Helper structure allowing to roll all execution properties into
        // a single interface
        template<typename FFT>
        struct pre {
            using compute_t                                = typename FFT::value_type;
            using simple_t                                 = typename compute_t::value_type;
            static constexpr unsigned int complex_fft_size = size_of<FFT>::value / 2;
            static constexpr unsigned int half_complex_size = (complex_fft_size % 2 == 0) ? (complex_fft_size / 2) : (complex_fft_size / 2 + 1);
            static constexpr unsigned int ept              = FFT::effective_elements_per_thread;
            static constexpr unsigned int half_ept         = (ept % 2 == 0) ? (ept / 2) : (ept / 2 + 1);
            static constexpr unsigned int stride           = complex_fft_size / ept;
            static constexpr bool         is_block         = size_of<FFT>::value > FFT::elements_per_thread;

            // Decide whether it's worth to reorder first and middle elements
            // and simulate their symmetricity. This allows to get rid of
            // conditional processing from unrolled loops and get a more
            // performant execution.
            static __device__ constexpr bool
            needs_loop_reorder_pre() {
                constexpr auto size       = size_of<FFT>::value;
                using precision           = typename precision_of<FFT>::type;
                constexpr bool is_natural = real_fft_layout_of<FFT>::value == complex_layout::natural;

                // The first and middle element need an extra register,
                // which is available only in the natural layout.
                if constexpr (!is_natural) {
                    return false;
                }

                // Sizes for which conditional processing inside loop slows down execution for FP16 precision
                constexpr bool fp16_case = CUFFTDX_STD::is_same_v<precision, __half> && (size == 2048 || size == 4096 ||
                                                                                         size == 8192 || size == 16384 ||
                                                                                         size == 32768 || size == 65536);

                // Sizes for which conditional processing inside loop slows down execution for FP32 precision
                constexpr bool fp32_case = CUFFTDX_STD::is_same_v<precision, float> && (size == 2048 || size == 8192 ||
                                                                                        size == 16384 || size == 32768);

                // Sizes for which conditional processing inside loop slows down execution for FP64 precision
                constexpr bool fp64_case = CUFFTDX_STD::is_same_v<precision, double> && (size == 128 || size == 512 ||
                                                                                         size == 1024 || size == 4096 ||
                                                                                         size == 8192);

                return fp16_case || fp32_case || fp64_case;
            }

            static constexpr bool loop_reorder = needs_loop_reorder_pre();

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
                    rmem[0]       = compute_t {rmem[0].x + rmem[ept].x, rmem[0].x - rmem[ept].x};
                    if constexpr(ept % 2 == 0) {
                        rmem[half_ept].x = cufftdx::detail::get_val<simple_t>(2.0) * rmem[half_ept].x;
                        rmem[half_ept].y = cufftdx::detail::get_val<simple_t>(-2.0) * rmem[half_ept].y;
                    }
                } else if constexpr (real_fft_layout_of<FFT>::value == complex_layout::packed) {
                    // This follows the ::natural approach, but instead of storing these two
                    // values separately it packs them together into rmem[0]
                    rmem[0]       = compute_t {rmem[0].x + rmem[0].y, rmem[0].x - rmem[0].y};
                    if constexpr(ept % 2 == 0) {
                        rmem[half_ept].x = cufftdx::detail::get_val<simple_t>(2.0) * rmem[half_ept].x;
                        rmem[half_ept].y = cufftdx::detail::get_val<simple_t>(-2.0) * rmem[half_ept].y;
                    }
                }
            }

            // Use the hermicity of an C2R FFT input and a basic Cooley-Tukey
            // butterfly to combine elements of frequency domain of length N
            // into 2 separate frequency domain inputs of length N/2
            static __device__ __forceinline__ void symmetric_process(compute_t& v0, compute_t& v1, const compute_t& twiddle) {
                const compute_t A = {v0.x + v1.x, v0.y + v1.y};
                const compute_t B = {v0.x - v1.x, v0.y - v1.y};

                v0.x = A.x - (A.y * twiddle.x - B.x * twiddle.y);
                v1.x = A.x + (A.y * twiddle.x - B.x * twiddle.y);
                v0.y = (B.x * twiddle.x + A.y * twiddle.y) + B.y;
                v1.y = (B.x * twiddle.x + A.y * twiddle.y) - B.y;
            }

            // Reorder last and middle elements to allow for
            // processing last's symmetricity with first.
            // ===
            // Used only in cases enumerated in: needs_loop_reorder()
            template<typename compute_t>
            static __forceinline__ __device__ void
            reorder_last_and_middle_element(compute_t* rmem) {
                if (!is_block || threadIdx.x == 0) {
                    // The last and first value's imaginary parts are irrelevant in C2R symmetric FFT.
                    // The algorithm requires them to be zero, otherwise it will break.
                    rmem[ept].y = get_zero<simple_t>();
                    rmem[0].y   = get_zero<simple_t>();
                    // Middle element to the end of rmem, to be processed later.
                    // Putting 16th element in its place.
                    compute_t tmp = rmem[half_ept];
                    rmem[half_ept] = rmem[ept];
                    rmem[ept]     = tmp;
                }
            }

            // Change the data layout in each thread from canonically strided:
            // a[idx], a[idx + stride], a[idx + 2 * stride] ... a[idx + EPT * stride]
            // to symmetric:
            // a[idx], a[idx + stride] ... a[N - idx - stride], a[N - idx]
            // to allow for in-thread processing of symmetric elements
            template<bool IsShared>
            static __forceinline__ __device__ void
            block_reorder_strided_to_symmetric(compute_t* rmem, compute_t* smem) {
                __syncthreads();
                unsigned int index = threadIdx.x;
                if constexpr (!IsShared) {
                    // Exchange elements for preprocessing
                    // Store 2nd half of rmem to shared
                    for (unsigned i = 0; i < half_ept; i++) {
                        smem[index] = rmem[i + half_ept];
                        index += stride;
                    }
                    __syncthreads();
                    index = threadIdx.x;
                }
                // Load to data from other threads to 2nd half of rmem
                for (unsigned i = 0; i < half_ept; i++) {
                    const unsigned mirrored_index = index ? (half_complex_size - index) : 0;
                    rmem[i + half_ept]             = smem[mirrored_index];
                    index += stride;
                }
            }

            // After moving middle element before, now
            // we swap it back into its initial place,
            // and process it symmetrically with itself
            // ===
            // Used only in cases enumerated in: needs_loop_reorder()
            static __forceinline__ __device__ void
            swap_and_preprocess_last_element(compute_t* rmem) {
                if (!is_block || threadIdx.x == 0) {
                    // Middle element processing:
                    // This is equivalent of doing preprocess(rmem[ept], rmem[ept], twiddle)
                    // with twiddle equal to (0.0, -1.0).
                    rmem[half_ept].x = get_val<simple_t>(2.0f) * rmem[ept].x;
                    rmem[half_ept].y = get_val<simple_t>(-2.0f) * rmem[ept].y;
                }
            }

            // Change the data layout from symmetric in each thread:
            // a[idx], a[idx + stride] ... a[N - idx - stride], a[N - idx]
            // to canonically strided:
            // a[idx], a[idx + stride], a[idx + 2 * stride] ... a[idx + EPT * stride]
            // to allow for in-thread processing of symmetric elements
            static __forceinline__ __device__ void
            block_reorder_symmetric_to_strided(compute_t* rmem, compute_t* smem) {
                __syncthreads();
                // Exchange elements back before storing to global
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
        };

        // [Preprocess]
        //
        // {(N/2 + 1) / (N/2) / (N)} complex inputs are preprocessed into N/2 complex inputs of C2R.
        // Input length depends on chosen complex layout:
        // natural -> N/2 + 1 complex inputs (first and middle elements must have 0 in their imaginary parts)
        // packed -> N/2 complex inputs (first and middle elements combined in 0th index)
        // full -> N complex inputs (fully unrolled hermitian sequence)
        //
        // We need to pair first half with it's mirrored reflection, and apply preprocess
        // to them. 0th and middle elements are the exceptions in the process (in the natural and full layout,
        // in packed these elements are combined together in the first element):
        // * 0th element should be paired with 16th and result saved to 0th position only.
        // * Middle (8) element should be paired with itself and single result should be saved to middle position.
        //
        // Example for natural layout:
        // nx = 32, compute_nx = 16, ept = 4
        // 0: preprocess(value0, value16)  > save(value0, 0) -> they both have only real parts
        // 0: preprocess(value1, value15) -> save(value1, 1), save(value15, compute_nx-1=15);
        // 1: preprocess(value2, value14) -> save(value2, 2), save(value14, compute_nx-2=14);
        // 1: preprocess(value3, value13) -> save(value3, 3), save(value13, compute_nx-3=13);
        // 2: preprocess(value4, value12) -> save(value4, 4), save(value12, compute_nx-4=12);
        // 2: preprocess(value5, value11) -> save(value5, 5), save(value11, compute_nx-5=11);
        // 3: preprocess(value6, value10) -> save(value6, 6), save(value10, compute_nx-6=10);
        // 3: preprocess(value7, value9)  -> save(value7, 7), save(value9,  compute_nx-7=9);
        // 0: preprocess(value8, value8)  -> save(value8, 8)

        // Many threads
        template<class FFT, bool IsShared = false, class ComplexType = typename FFT::value_type>
        inline __device__ auto preprocess_fold_c2r(ComplexType* rmem, ComplexType* smem) -> CUFFTDX_STD::enable_if_t<size_of<FFT>::value != FFT::elements_per_thread> {
            ComplexType  twiddle;
            unsigned int index = threadIdx.x;

            const unsigned int shared_offset = IsShared
                                                   ? threadIdx.y * FFT::input_length
                                                   : threadIdx.y * pre<FFT>::half_complex_size;
            smem += shared_offset;

            if constexpr (IsShared) {
                // If Shared API is used, there is no need to store data
                // from registers to shared, as it's already in there
                smem += pre<FFT>::half_complex_size;
            }

            // Shuffle data through shared memory so that each thread
            // contains symmetric values, ready for preprocessing
            pre<FFT>::block_reorder_strided_to_symmetric<IsShared>(rmem, smem);

            // This needs to be performed only when we
            // are simulating symmetricity for the last
            // and middle elements.
            // ===
            // Used only in cases enumerated in: needs_loop_reorder()
            if constexpr (pre<FFT>::loop_reorder) {
                pre<FFT>::reorder_last_and_middle_element(rmem);
            } else {
                if (threadIdx.x == 0) {
                    pre<FFT>::process_first_and_middle_elements(rmem);

                } else {
                    twiddle = detail::compute_twiddle<typename pre<FFT>::simple_t>(index, size_of<FFT>::value);
                    pre<FFT>::symmetric_process(rmem[0], rmem[pre<FFT>::half_ept], twiddle);
                }
                index += pre<FFT>::stride;
            }

            // Preprocess all elements, applying the properties of real valued FFT
            // (hermicity of input) and basic Cooley-Tukey butterfly, to get 2 length-N/2
            // frequency domain inputs from 1 length N sequence
            for (unsigned i = (pre<FFT>::loop_reorder ? 0 : 1); i < pre<FFT>::half_ept; i++) {
                twiddle = detail::compute_twiddle<typename pre<FFT>::simple_t>(index, size_of<FFT>::value);
                pre<FFT>::symmetric_process(rmem[i], rmem[i + pre<FFT>::half_ept], twiddle);
                index += pre<FFT>::stride;
            }

            // This needs to be performed only when we
            // are simulating symmetricity for the last
            // and middle elements.
            // ===
            // Used only in cases enumerated in: needs_loop_reorder()
            if constexpr (pre<FFT>::loop_reorder) {
                pre<FFT>::swap_and_preprocess_last_element(rmem);
            }

            // Shuffle data through shared memory so that each thread
            // contains values in canonical strided order, expected as
            // output after preprocessing
            pre<FFT>::block_reorder_symmetric_to_strided(rmem, smem);
        }

        // Single thread
        template<class FFT, class ComplexType = typename FFT::value_type>
        inline __device__ void preprocess_fold_c2r(ComplexType* rmem) {
            ComplexType twiddle;

            pre<FFT>::process_first_and_middle_elements(rmem);

            // Preprocess all elements, applying the properties of real valued FFT
            // (hermicity of input) and basic Cooley-Tukey butterfly, to get 2 length-N/2
            // frequency domain inputs from 1 length N sequence
            for (unsigned i = 1; i < pre<FFT>::half_ept; i++) {
                const unsigned mirrored_index = pre<FFT>::complex_fft_size - i;
                twiddle                       = detail::compute_twiddle<typename pre<FFT>::simple_t>(i, size_of<FFT>::value);
                pre<FFT>::symmetric_process(rmem[i], rmem[mirrored_index], twiddle);
            }
        }

        // Single thread
        template<class FFT, bool IsShared = false, class ComplexType = typename FFT::value_type>
        inline __device__ auto preprocess_fold_c2r(ComplexType* rmem, ComplexType* /*smem*/) -> CUFFTDX_STD::enable_if_t<size_of<FFT>::value == FFT::elements_per_thread> {
            preprocess_fold_c2r<FFT>(rmem);
        }
    } // namespace detail
} // namespace cufftdx


#endif // CUFFTDX_DETAIL_PREPROCESS_FOLD_HPP
