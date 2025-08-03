// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_DETAIL_FFT_EXECUTION_HPP
#define CUFFTDX_DETAIL_FFT_EXECUTION_HPP

#ifdef CUFFTDX_DETAIL_USE_CUDA_STL
#    include <cuda/std/type_traits>
#else
#    include <type_traits>
#endif

#ifndef __CUDACC_RTC__
#    include <cassert>
#endif

#ifndef __CUDACC_RTC__
#    include <cuda_runtime_api.h> // cudaError_t
#endif

#include <cuda_fp16.h>

#include "fft_checks.hpp"
#include "fft_description.hpp"
#include "workspace.hpp"

#include "../traits/detail/ldg_type.hpp"

#include "processing/processing.hpp"

#define STRINGIFY(s) XSTRINGIFY(s)
#define XSTRINGIFY(s) #s

namespace cufftdx {
    namespace detail {
        template<class T, unsigned int Size>
        struct normalize_helper {
            inline __device__ void operator()(T& value) { value /= Size; }
        };

        template<unsigned int Size>
        struct normalize_helper<complex<__half2>, Size> {
            inline __device__ void operator()(complex<__half2>& value) {
                value.x /= __half2 {Size, Size};
                value.y /= __half2 {Size, Size};
            }
        };

        template<unsigned int Size, class T>
        inline __device__ void normalize(T& value) {
            return normalize_helper<T, Size>()(value);
        };

        inline constexpr __device__ unsigned int divide_based_on_layout(complex_layout  this_fft_complex_layout,
                                                                        unsigned int value) {
            switch (this_fft_complex_layout) {
                case complex_layout::natural: return value / 2 + 1;
                case complex_layout::packed: return value / 2;
                case complex_layout::full: return value;
            }

            return value;
        }

        inline constexpr __device__ unsigned int get_output_ept(fft_type this_fft_type, complex_layout this_fft_complex_layout, unsigned int ept, unsigned int effective_ept) {
            if (this_fft_type == cufftdx::fft_type::c2r) {
                return effective_ept;
            } else if (this_fft_type == cufftdx::fft_type::r2c) {
                return divide_based_on_layout(this_fft_complex_layout, ept);
            } else if (this_fft_type == cufftdx::fft_type::c2c) {
                return effective_ept;
            }

            return effective_ept;
        };

        inline constexpr __device__ unsigned int get_input_ept(fft_type this_fft_type, complex_layout this_fft_complex_layout, unsigned int ept, unsigned int effective_ept) {
            if (this_fft_type == cufftdx::fft_type::r2c) {
                return effective_ept;
            } else if (this_fft_type == cufftdx::fft_type::c2r) {
                return divide_based_on_layout(this_fft_complex_layout, ept);
            } else if (this_fft_type == cufftdx::fft_type::c2c) {
                return effective_ept;
            }

            return effective_ept;
        }

        inline static constexpr __device__ auto get_input_length(fft_type this_fft_type, complex_layout this_fft_complex_layout, unsigned int size, unsigned int execution_size) {
            const auto complex_len = divide_based_on_layout(this_fft_complex_layout, size);
            return this_fft_type == fft_type::c2r ? complex_len : execution_size;
        }

        inline static constexpr __device__ auto get_output_length(fft_type this_fft_type, complex_layout this_fft_complex_layout, unsigned int size, unsigned int execution_size) {
            const auto complex_len = divide_based_on_layout(this_fft_complex_layout, size);
            return this_fft_type == fft_type::r2c ? complex_len : execution_size;
        }


        // This is necessary since when performing R2C or C2R FFT, the actual C2C implementation FFT happening inside
        // can be of a different size than the real FFT. To verify existence in database we need to be able to check
        // the effective execution size
        inline static constexpr int get_fft_execution_size(real_mode mode, complex_layout /* layout */, unsigned int size) {
            const bool is_folded = mode == real_mode::folded;
            return is_folded ? size / 2 : size;
        }

        // This is necessary since when performing R2C or C2R FFT, the actual C2C implementation FFT happening inside
        // can be of a different size than the real FFT, but its memory storage size is dependent on complex layout
        // provided. Although both complex_layout::natural and complex_layout::packed have a memory size equal to that of
        // their equivalent execution size C2C FFTs, the full layout is twice that. This means that in special cases
        // of performing the maximal sizes, the full option is not supported, since it exceeds the limit twice.
        inline static constexpr int get_fft_memory_size(real_mode mode, complex_layout layout, unsigned int size) {
            const bool is_folded = mode == real_mode::folded;
            return (is_folded && layout != complex_layout::full)
                       ? size / 2
                       : size;
        }

        template<class... Operators>
        class fft_execution: public fft_description<Operators...>, public commondx::detail::execution_description_expression
        {
            using base_type      = fft_description<Operators...>;
            using execution_type = fft_execution<Operators...>;

        protected:
            // Precision type
            using typename base_type::this_fft_precision_t;

            /// ---- Constraints

            // We need Block or Thread to be specified exactly once
            static constexpr bool is_thread_execution = has_n_of<1, fft_operator::thread, execution_type>::value;
            static constexpr bool is_block_execution  = has_n_of<1, fft_operator::block, execution_type>::value;
            static_assert((is_thread_execution != is_block_execution), "Can't create FFT with two execution operators");
        };

#if defined(__CUDACC_RTC__) || defined(_MSC_VER)
        template<class... Operators>
        class fft_thread_execution_partial: public fft_execution<Operators...>
        {
            using base_type = fft_execution<Operators...>;
            using typename base_type::this_fft_precision_t;

        public:
            using value_type  = typename make_complex_type<this_fft_precision_t>::cufftdx_type;
        };
#endif
        template<class... Operators>
        class fft_thread_execution: public fft_execution<Operators...>
        {
            using this_type = fft_thread_execution<Operators...>;
            using base_type = fft_execution<Operators...>;
            using typename base_type::this_fft_precision_t;
            using host_workspace_type = cufftdx::detail::empty_workspace;

#ifndef __CUDACC_RTC__
            template<class FFT>
            friend typename FFT::host_workspace_type cufftdx::make_workspace(cudaError_t&, cudaStream_t) noexcept;

            template<class FFT>
            friend typename FFT::host_workspace_type cufftdx::make_workspace(cudaStream_t);
#endif // __CUDACC_RTC__

        protected:
            // Thread can't have block-only operators
            static constexpr bool has_block_only_operators = has_any_block_operator<base_type>::value;
            static_assert(!has_block_only_operators, "FFT for thread execution can't contain block-only operators");

            static constexpr bool is_folded = real_fft_mode_of<base_type>::value == real_mode::folded;

            // Size of the equivalent C2C FFT that will be stored
            // (full layout doubles it and limits max size)
            static constexpr auto fft_execution_size = get_fft_execution_size(base_type::this_fft_real_mode_v,
                                                                              base_type::this_fft_complex_layout_v,
                                                                              base_type::this_fft_size_v);

            static constexpr auto fft_memory_size = get_fft_memory_size(base_type::this_fft_real_mode_v,
                                                                        base_type::this_fft_complex_layout_v,
                                                                        base_type::this_fft_size_v);

            static constexpr auto effective_fft_type_v = is_folded ? fft_type::c2c : base_type::this_fft_type_v;

            template<class Precision>
            static constexpr bool is_valid_size_for_thread() {
                // We can't trivially pass size here, since R2C and C2R executions allow for doubling the
                // maximal execution size (by halving it with simulated C2C execution). We need to expect
                // type underconstruction and this flag is responsible for this detection. If we suspect
                // that the user may want to pass additional operators which will halve the execution size
                // we need to provision for this possibility by not making default assumptions.
                //
                // If underconstructed FFT type is tried for execution it will fail at check_and_get_traits
                constexpr bool relax_max_size = !this_type::has_type || (this_type::this_fft_type_v != fft_type::c2c && !this_type::has_real_fft_options);

                using is_size_supported = is_supported<Precision, fft_memory_size, unsigned(-1), relax_max_size>;
                return !base_type::has_size || // Size<> was not defined
                       !(CUFFTDX_STD::is_same<this_fft_precision_t, Precision>::value) ||
                       is_size_supported::thread_value;
            }
            // Thread, Size and Precision constrains
            static constexpr bool valid_size_for_thread_fp16 = is_valid_size_for_thread<__half>();
            static constexpr bool valid_size_for_thread_fp32 = is_valid_size_for_thread<float>();
            static constexpr bool valid_size_for_thread_fp64 = is_valid_size_for_thread<double>();

            static_assert(valid_size_for_thread_fp16,
                          "Thread execution in fp16 precision supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_THREAD_FP16) "]");
            static_assert(valid_size_for_thread_fp32,
                          "Thread execution in fp32 precision supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_THREAD_FP32) "]");
            static_assert(valid_size_for_thread_fp64,
                          "Thread execution in fp64 precision supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_THREAD_FP64) "]");

            inline static constexpr unsigned int get_storage_size() {
                static_assert(base_type::is_complete, "Can't calculate storage_size, FFT description is not complete");
                using fft_implementation_t       = check_and_get_fft_implementation_t<this_type>;
                constexpr bool is_folded = real_fft_mode_of<base_type>::value == real_mode::folded;
                if constexpr (is_folded) {
                    constexpr auto realopt = real_fft_layout_of<base_type>::value;
                    if constexpr (realopt == complex_layout::natural) {
                        return fft_implementation_t::storage_size + 1;
                    } else if constexpr (realopt == complex_layout::packed) {
                        return fft_implementation_t::storage_size;
                    } else if constexpr (realopt == complex_layout::full) {
                        return 2 * fft_implementation_t::storage_size;
                    }
                } else {
                    return fft_implementation_t::storage_size;
                }

                return 0;
            }


        public:
            using value_type     = typename make_complex_type<this_fft_precision_t>::cufftdx_type;
            using workspace_type = typename host_workspace_type::device_handle;
            static_assert(CUFFTDX_STD::is_same<host_workspace_type, cufftdx::detail::empty_workspace>::value,
                          "Internal cuFFTDx error, thread FFT should never require non-empty workspace");


            inline __device__ void execute(value_type* input) {
                static_assert(base_type::is_complete, "Can't execute, FFT description is not complete");

                using fft_implementation_t        = check_and_get_fft_implementation_t<this_type>;
                static constexpr auto function_id = fft_implementation_t::function_id;

                preprocess<this_type>(input);
                using scalar_type = typename value_type::value_type;
                database::detail::cufftdx_private_function_wrapper<function_id, scalar_type, 1>(input, nullptr);
                postprocess<this_type>(input);
            }

            // T - can be any type if it's alignment and size are the same as those of ::value_type
            template<class T /* TODO = typename make_vector_type<make_scalar_type<value_type>, 2>::type */>
            inline __device__ auto execute(T* input) //
                -> CUFFTDX_STD::enable_if_t<!CUFFTDX_STD::is_void<T>::value && (sizeof(T) == sizeof(value_type)) &&
                                            (alignof(T) == alignof(value_type))> {
                return execute(reinterpret_cast<value_type*>(input));
            }

            template<class T>
            inline __device__ auto execute(T* input) //
                -> CUFFTDX_STD::enable_if_t<CUFFTDX_STD::is_void<T>::value || (sizeof(T) != sizeof(value_type)) ||
                                            (alignof(T) != alignof(value_type))> {
                static constexpr bool condition =
                    CUFFTDX_STD::is_void<T>::value || (sizeof(T) != sizeof(value_type)) || (alignof(T) != alignof(value_type));
                static_assert(not condition, "Incorrect value type is used, try using ::value_type");
            }

            template<class T>
            inline __device__ auto execute(T* input, workspace_type& /* workspace */) //
                -> CUFFTDX_STD::enable_if_t<!CUFFTDX_STD::is_void<T>::value && (sizeof(T) == sizeof(value_type)) &&
                                            (alignof(T) == alignof(value_type))> {
                return execute(reinterpret_cast<value_type*>(input));
            }

            template<class T>
            inline __device__ auto execute(T* /* input */, workspace_type& /* workspace */) //
                -> CUFFTDX_STD::enable_if_t<CUFFTDX_STD::is_void<T>::value || (sizeof(T) != sizeof(value_type)) ||
                                            (alignof(T) != alignof(value_type))> {
                static constexpr bool condition =
                    CUFFTDX_STD::is_void<T>::value || (sizeof(T) != sizeof(value_type)) || (alignof(T) != alignof(value_type));
                static_assert(not condition, "Incorrect value type is used, try using ::value_type");
            }

            static constexpr unsigned int elements_per_thread           = check_and_get_trait<fft_operator::elements_per_thread, base_type>::value;
            static constexpr unsigned int effective_elements_per_thread = check_and_get_trait<fft_operator::elements_per_thread, base_type>::effective;
            static constexpr unsigned int stride                        = 1;
            static constexpr unsigned int storage_size                  = get_storage_size();

            static constexpr unsigned int input_length  = get_input_length(base_type::this_fft_type_v, base_type::this_fft_complex_layout_v, base_type::this_fft_size_v, fft_execution_size);
            static constexpr unsigned int output_length = get_output_length(base_type::this_fft_type_v, base_type::this_fft_complex_layout_v, base_type::this_fft_size_v, fft_execution_size);

            static constexpr unsigned int implicit_type_batching =
                CUFFTDX_STD::is_same<this_fft_precision_t, __half>::value ? 2 : 1;

            using input_type                        = CUFFTDX_STD::conditional_t<(!is_folded) && (base_type::this_fft_type_v == fft_type::r2c),
                                                         typename value_type::value_type,
                                                         value_type>;
            static constexpr unsigned int input_ept = get_input_ept(base_type::this_fft_type_v,
                                                                  base_type::this_fft_complex_layout_v,
                                                                  elements_per_thread,
                                                                  effective_elements_per_thread);

            using output_type                        = CUFFTDX_STD::conditional_t<(!is_folded) && (base_type::this_fft_type_v == fft_type::c2r),
                                                          typename value_type::value_type,
                                                          value_type>;
            static constexpr unsigned int output_ept = get_output_ept(base_type::this_fft_type_v,
                                                                    base_type::this_fft_complex_layout_v,
                                                                    elements_per_thread,
                                                                    effective_elements_per_thread);
        };

#if defined(__CUDACC_RTC__) || defined(_MSC_VER)
        template<class... Operators>
        class fft_block_execution_partial: public fft_execution<Operators...>
        {
            using base_type = fft_execution<Operators...>;
            using typename base_type::this_fft_precision_t;

        public:
            using value_type  = typename make_complex_type<this_fft_precision_t>::cufftdx_type;
        };
#endif

        template<class... Operators>
        class fft_block_execution: public fft_execution<Operators...>
        {
            using this_type = fft_block_execution<Operators...>;
            using base_type = fft_execution<Operators...>;
            using typename base_type::this_fft_precision_t;

        public:
            using value_type = typename make_complex_type<this_fft_precision_t>::cufftdx_type;

        private:
#ifndef __CUDACC_RTC__
            template<class FFT>
            friend typename FFT::host_workspace_type cufftdx::make_workspace(cudaError_t&, cudaStream_t) noexcept;

            template<class FFT>
            friend typename FFT::host_workspace_type cufftdx::make_workspace(cudaStream_t);
#endif // __CUDACC_RTC__
            static constexpr bool is_folded = real_fft_mode_of<base_type>::value == real_mode::folded;

            // Size of the equivalent C2C FFT that will be executed
            // (if fold optimization is turned on the size will be halved)
            static constexpr auto fft_execution_size = get_fft_execution_size(base_type::this_fft_real_mode_v,
                                                                              base_type::this_fft_complex_layout_v,
                                                                              base_type::this_fft_size_v);

            // Size of the equivalent C2C FFT that will be stored
            // (full layout doubles it and limits max size)
            static constexpr auto fft_memory_size = get_fft_memory_size(base_type::this_fft_real_mode_v,
                                                                        base_type::this_fft_complex_layout_v,
                                                                        base_type::this_fft_size_v);

            static constexpr auto effective_fft_type_v = is_folded ? fft_type::c2c : base_type::this_fft_type_v;

            static constexpr auto max_block_size = is_supported<this_type::this_fft_precision_t, fft_memory_size, base_type::this_fft_sm_v, true>::effective_max_block_size;

            // Max block size is set to a size that is possible to be performed only by R2C/C2R by effectively halving it.
            // If a description has execution size set to that value it may mean 2 things:
            // - The description is not complete (RealFFTOptions has not been passed yet)
            // - The user is trying to construct an unsupported FFT
            // To allow for the first case we need to detect this behavior (hence the boolean flag)
            // and wait with assertions for check_and_get_trait call.
            //
            // Whenever workspace_selector is passed a size that's not in the execution database it assumes that bluestein
            // execution will be required and constructs appropriate types (otherwise empty_workspace is returned). Since
            // there is no R2C_MAX_SIZE config in the database (because C2C performed is of size R2C_MAX_SIZE / 2) what
            // we need to do is give this assertion some leeway if the main FFT type is underdefined and max size is chosen.
            static constexpr bool is_this_fft_max_block_size = fft_execution_size == max_block_size;
            using host_workspace_type                        = typename workspace_selector<is_this_fft_max_block_size ? fft_execution_size / 2 : fft_execution_size,
                                                                    this_fft_precision_t,
                                                                    value_type,
                                                                    base_type::this_fft_direction_v,
                                                                    effective_fft_type_v,
                                                                    base_type::this_fft_sm_v,
                                                                    base_type::is_complete>::type;


            // Return false if fft's precision matches 'Precision', sm matches 'SM'
            // and test can not be executed
            template<class Precision, unsigned SM>
            static constexpr bool is_valid_size_for_block() {
                // We can't trivially pass size here, since R2C and C2R executions allow for doubling the
                // maximal execution size (by halving it with simulated C2C execution). We need to expect
                // type underconstruction and this flag is responsible for this detection. If we suspect
                // that the user may want to pass additional operators which will halve the execution size
                // we need to provision for this possibility by not making default assumptions.
                //
                // If underconstructed FFT type is tried for execution it will fail at check_and_get_traits
                constexpr bool relax_max_size = !this_type::has_type || (this_type::this_fft_type_v != fft_type::c2c && !this_type::has_real_fft_options);

                using is_size_supported = is_supported<Precision, fft_memory_size, base_type::this_fft_sm_v, relax_max_size>;
                return !base_type::has_size || // Size<> was not defined
                       !(base_type::this_fft_sm_v == SM && base_type::has_sm) ||
                       !(CUFFTDX_STD::is_same<this_fft_precision_t, Precision>::value) ||
                       is_size_supported::value;
            }

            // Check requirements for Bluestein size
            // If we need Bluestein, we need to generate chirp using FP64 FFT of size next-power_of_2(2*N - 1)
            template<unsigned SM>
            static constexpr bool is_valid_size_for_bluestein() {
                // host_workspace_type is empty_workspace when bluestein is not used for execution
                // and if so other conditions are not evaluated.
                constexpr bool this_fft_is_bluestein_required =
                    CUFFTDX_STD::is_same<
                        host_workspace_type,
                        detail::bluestein_workspace<fft_execution_size,
                                                    value_type,
                                                    base_type::this_fft_sm_v>>::value;

                return !this_fft_is_bluestein_required ||
                       !base_type::has_size || // Size<> was not defined
                       !(base_type::this_fft_sm_v == SM && base_type::has_sm) ||
                       is_supported<double, fft_memory_size, base_type::this_fft_sm_v>::blue_block_value;
            }

        protected:
            // Block, Size and Precision constrains

            // SM70
            static constexpr bool valid_size_for_block_fp16_sm70 = is_valid_size_for_block<__half, 700>();
            static constexpr bool valid_size_for_block_fp32_sm70 = is_valid_size_for_block<float, 700>();
            static constexpr bool valid_size_for_block_fp64_sm70 = is_valid_size_for_block<double, 700>();
            static_assert(valid_size_for_block_fp16_sm70,
                          "Block execution in fp16 precision on SM70 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM700_FP16_MAX) "]");
            static_assert(valid_size_for_block_fp32_sm70,
                          "Block execution in fp32 precision on SM70 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM700_FP32_MAX) "]");
            static_assert(valid_size_for_block_fp64_sm70,
                          "Block execution in fp64 precision on SM70 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM700_FP64_MAX) "]");

            static constexpr bool valid_size_for_bluestein_sm70 = is_valid_size_for_bluestein<700>();
            static_assert(valid_size_for_bluestein_sm70, "Block execution for this size is not supported");

            // SM72
            static constexpr bool valid_size_for_block_fp16_sm72 = is_valid_size_for_block<__half, 720>();
            static constexpr bool valid_size_for_block_fp32_sm72 = is_valid_size_for_block<float, 720>();
            static constexpr bool valid_size_for_block_fp64_sm72 = is_valid_size_for_block<double, 720>();
            static_assert(valid_size_for_block_fp16_sm72,
                          "Block execution in fp16 precision on SM72 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM720_FP16_MAX) "]");
            static_assert(valid_size_for_block_fp32_sm72,
                          "Block execution in fp32 precision on SM72 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM720_FP32_MAX) "]");
            static_assert(valid_size_for_block_fp64_sm72,
                          "Block execution in fp64 precision on SM72 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM720_FP64_MAX) "]");

            static constexpr bool valid_size_for_bluestein_sm72 = is_valid_size_for_bluestein<720>();
            static_assert(valid_size_for_bluestein_sm72, "Block execution for this size is not supported");

            // SM75
            static constexpr bool valid_size_for_block_fp16_sm75 = is_valid_size_for_block<__half, 750>();
            static constexpr bool valid_size_for_block_fp32_sm75 = is_valid_size_for_block<float, 750>();
            static constexpr bool valid_size_for_block_fp64_sm75 = is_valid_size_for_block<double, 750>();
            static_assert(valid_size_for_block_fp16_sm75,
                          "Block execution in fp16 precision on SM75 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM750_FP16_MAX) "]");
            static_assert(valid_size_for_block_fp32_sm75,
                          "Block execution in fp32 precision on SM75 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM750_FP32_MAX) "]");
            static_assert(valid_size_for_block_fp64_sm75,
                          "Block execution in fp64 precision on SM75 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM750_FP64_MAX) "]");

            static constexpr bool valid_size_for_bluestein_sm75 = is_valid_size_for_bluestein<750>();
            static_assert(valid_size_for_bluestein_sm75, "Block execution for this size is not supported");

            // SM80
            static constexpr bool valid_size_for_block_fp16_sm80 = is_valid_size_for_block<__half, 800>();
            static constexpr bool valid_size_for_block_fp32_sm80 = is_valid_size_for_block<float, 800>();
            static constexpr bool valid_size_for_block_fp64_sm80 = is_valid_size_for_block<double, 800>();
            static_assert(valid_size_for_block_fp16_sm80,
                          "Block execution in fp16 precision on SM80 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM800_FP16_MAX) "]");
            static_assert(valid_size_for_block_fp32_sm80,
                          "Block execution in fp32 precision on SM80 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM800_FP32_MAX) "]");
            static_assert(valid_size_for_block_fp64_sm80,
                          "Block execution in fp64 precision on SM80 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM800_FP64_MAX) "]");

            static constexpr bool valid_size_for_bluestein_sm80 = is_valid_size_for_bluestein<800>();
            static_assert(valid_size_for_bluestein_sm80, "Block execution for this size is not supported");

            // SM86
            static constexpr bool valid_size_for_block_fp16_sm86 = is_valid_size_for_block<__half, 860>();
            static constexpr bool valid_size_for_block_fp32_sm86 = is_valid_size_for_block<float, 860>();
            static constexpr bool valid_size_for_block_fp64_sm86 = is_valid_size_for_block<double, 860>();
            static_assert(valid_size_for_block_fp16_sm86,
                          "Block execution in fp16 precision on SM86 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM860_FP16_MAX) "]");
            static_assert(valid_size_for_block_fp32_sm86,
                          "Block execution in fp32 precision on SM86 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM860_FP32_MAX) "]");
            static_assert(valid_size_for_block_fp64_sm86,
                          "Block execution in fp64 precision on SM86 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM860_FP64_MAX) "]");

            static constexpr bool valid_size_for_bluestein_sm86 = is_valid_size_for_bluestein<860>();
            static_assert(valid_size_for_bluestein_sm86, "Block execution for this size is not supported");

            // SM87
            static constexpr bool valid_size_for_block_fp16_sm87 = is_valid_size_for_block<__half, 870>();
            static constexpr bool valid_size_for_block_fp32_sm87 = is_valid_size_for_block<float, 870>();
            static constexpr bool valid_size_for_block_fp64_sm87 = is_valid_size_for_block<double, 870>();
            static_assert(valid_size_for_block_fp16_sm87,
                          "Block execution in fp16 precision on SM87 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM870_FP16_MAX) "]");
            static_assert(valid_size_for_block_fp32_sm87,
                          "Block execution in fp32 precision on SM87 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM870_FP32_MAX) "]");
            static_assert(valid_size_for_block_fp64_sm87,
                          "Block execution in fp64 precision on SM87 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM870_FP64_MAX) "]");

            static constexpr bool valid_size_for_bluestein_sm87 = is_valid_size_for_bluestein<870>();
            static_assert(valid_size_for_bluestein_sm87, "Block execution for this size is not supported");

            // SM89
            static constexpr bool valid_size_for_block_fp16_sm89 = is_valid_size_for_block<__half, 890>();
            static constexpr bool valid_size_for_block_fp32_sm89 = is_valid_size_for_block<float, 890>();
            static constexpr bool valid_size_for_block_fp64_sm89 = is_valid_size_for_block<double, 890>();
            static_assert(valid_size_for_block_fp16_sm89,
                          "Block execution in fp16 precision on SM89 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM890_FP16_MAX) "]");
            static_assert(valid_size_for_block_fp32_sm89,
                          "Block execution in fp32 precision on SM89 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM890_FP32_MAX) "]");
            static_assert(valid_size_for_block_fp64_sm89,
                          "Block execution in fp64 precision on SM89 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM890_FP64_MAX) "]");

            static constexpr bool valid_size_for_bluestein_sm89 = is_valid_size_for_bluestein<890>();
            static_assert(valid_size_for_bluestein_sm89, "Block execution for this size is not supported");

            // SM90
            static constexpr bool valid_size_for_block_fp16_sm90 = is_valid_size_for_block<__half, 900>();
            static constexpr bool valid_size_for_block_fp32_sm90 = is_valid_size_for_block<float, 900>();
            static constexpr bool valid_size_for_block_fp64_sm90 = is_valid_size_for_block<double, 900>();
            static_assert(valid_size_for_block_fp16_sm90,
                          "Block execution in fp16 precision on SM90 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM900_FP16_MAX) "]");
            static_assert(valid_size_for_block_fp32_sm90,
                          "Block execution in fp32 precision on SM90 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM900_FP32_MAX) "]");
            static_assert(valid_size_for_block_fp64_sm90,
                          "Block execution in fp64 precision on SM90 supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM900_FP64_MAX) "]");

            static constexpr bool valid_size_for_bluestein_sm90 = is_valid_size_for_bluestein<900>();
            static_assert(valid_size_for_bluestein_sm90, "Block execution for this size is not supported");

            // MAX (No SM must be defined)
            static constexpr bool valid_size_for_block_fp16_max = is_valid_size_for_block<__half, unsigned(-1)>();
            static constexpr bool valid_size_for_block_fp32_max = is_valid_size_for_block<float, unsigned(-1)>();
            static constexpr bool valid_size_for_block_fp64_max = is_valid_size_for_block<double, unsigned(-1)>();
            static_assert(valid_size_for_block_fp16_max,
                          "Block execution in fp16 precision supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM800_FP16_MAX) "]");
            static_assert(valid_size_for_block_fp32_max,
                          "Block execution in fp32 precision supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM800_FP32_MAX) "]");
            static_assert(valid_size_for_block_fp64_max,
                          "Block execution in fp64 precision supports sizes in range [2; " STRINGIFY(CUFFTDX_DETAIL_SM800_FP64_MAX) "]");

            // MAX (No SM must be defined)
            // Check requirements for Bluestein size
            // If we need Bluestein, we need to generate chirp using FP64 FFT of size next-power_of_2(2*N - 1)
            static constexpr bool valid_size_for_bluestein_max = is_valid_size_for_bluestein<unsigned(-1)>();
            static_assert(valid_size_for_bluestein_max, "Block execution for this size is not supported");

        public:
            using workspace_type = typename host_workspace_type::device_handle;

            template<class T>
            inline __device__ void execute(T* shared_memory_input) {
#if !defined(NDEBUG) && defined(CUFFTDX_ENABLE_RUNTIME_ASSERTS) && !defined(__CUDACC_RTC__)
                const bool block_dimension_x_is_correct = (blockDim.x == block_dim.x);
                assert(block_dimension_x_is_correct);
                const bool block_dimension_y_is_correct = (blockDim.y == block_dim.y);
                assert(block_dimension_y_is_correct);
#endif
                static_assert(base_type::is_complete, "Can't execute, FFT description is not complete");
                static_assert(!requires_workspace, "This FFT configuration requires workspace");

                static constexpr bool use_bluestein = check_and_get_fft_implementation<this_type>::use_bluestein;

                value_type thread_data[storage_size];
                shared_to_registers<use_bluestein>(shared_memory_input, thread_data);

                block_preprocess_shared_api<this_type, use_bluestein>(thread_data, reinterpret_cast<value_type*>(shared_memory_input));
                workspace_type dummy_workspace;
                internal_execute<use_bluestein>(thread_data, shared_memory_input, dummy_workspace);
                block_postprocess<this_type, use_bluestein>(thread_data, reinterpret_cast<value_type*>(shared_memory_input));

                registers_to_shared<use_bluestein>(thread_data, shared_memory_input);
            }

            template<class T>
            inline __device__ void execute(T* shared_memory_input, workspace_type& workspace) {
#if !defined(NDEBUG) && defined(CUFFTDX_ENABLE_RUNTIME_ASSERTS) && !defined(__CUDACC_RTC__)
                const bool block_dimension_x_is_correct = (blockDim.x == block_dim.x);
                assert(block_dimension_x_is_correct);
                const bool block_dimension_y_is_correct = (blockDim.y == block_dim.y);
                assert(block_dimension_y_is_correct);
#endif
                static_assert(base_type::is_complete, "Can't execute, FFT description is not complete");

                value_type            thread_data[storage_size];
                static constexpr bool use_bluestein = check_and_get_fft_implementation<this_type>::use_bluestein;

                shared_to_registers<use_bluestein>(shared_memory_input, thread_data);

                block_preprocess_shared_api<this_type, use_bluestein>(thread_data, reinterpret_cast<value_type*>(shared_memory_input));
                internal_execute<use_bluestein>(thread_data, shared_memory_input, workspace);
                block_postprocess<this_type, use_bluestein>(thread_data, reinterpret_cast<value_type*>(shared_memory_input));

                registers_to_shared<use_bluestein>(thread_data, shared_memory_input);
            }

            inline __device__ void execute(value_type* input, void* shared_memory) {
#if !defined(NDEBUG) && defined(CUFFTDX_ENABLE_RUNTIME_ASSERTS) && !defined(__CUDACC_RTC__)
                const bool block_dimension_x_is_correct = (blockDim.x == block_dim.x);
                assert(block_dimension_x_is_correct);
                const bool block_dimension_y_is_correct = (blockDim.y == block_dim.y);
                assert(block_dimension_y_is_correct);
#endif

                static_assert(base_type::is_complete, "Can't execute, FFT description is not complete");
                static_assert(!requires_workspace, "This FFT configuration requires workspace");

                static constexpr bool use_bluestein = check_and_get_fft_implementation<this_type>::use_bluestein;
                block_preprocess<this_type, use_bluestein>(input, reinterpret_cast<value_type*>(shared_memory));
                workspace_type dummy_workspace;
                internal_execute<use_bluestein>(input, shared_memory, dummy_workspace);
                block_postprocess<this_type, use_bluestein>(input, reinterpret_cast<value_type*>(shared_memory));
            }

            // T - can be any type if its alignment and size are the same as those of ::value_type
            template<class T /* TODO = typename make_vector_type<make_scalar_type<value_type>, 2>::type */>
            inline __device__ auto execute(T* input, void* shared_memory) //
                -> CUFFTDX_STD::enable_if_t<!CUFFTDX_STD::is_void<T>::value && (sizeof(T) == sizeof(value_type)) &&
                                            (alignof(T) == alignof(value_type))> {
                return execute(reinterpret_cast<value_type*>(input), shared_memory);
            }

            template<class T>
            inline __device__ auto execute(T* /* input */, void* /* shared_memory */) //
                -> CUFFTDX_STD::enable_if_t<CUFFTDX_STD::is_void<T>::value || (sizeof(T) != sizeof(value_type)) ||
                                            (alignof(T) != alignof(value_type))> {
                static constexpr bool condition =
                    CUFFTDX_STD::is_void<T>::value || (sizeof(T) != sizeof(value_type)) || (alignof(T) != alignof(value_type));
                static_assert(not condition, "Incorrect value type is used, try using ::value_type");
            }

            inline __device__ void execute(value_type* input, void* shared_memory, workspace_type& workspace) {
#if !defined(NDEBUG) && defined(CUFFTDX_ENABLE_RUNTIME_ASSERTS) && !defined(__CUDACC_RTC__)
                const bool block_dimension_x_is_correct = (blockDim.x == block_dim.x);
                assert(block_dimension_x_is_correct);
                const bool block_dimension_y_is_correct = (blockDim.y == block_dim.y);
                assert(block_dimension_y_is_correct);
#endif
                static_assert(base_type::is_complete, "Can't execute, FFT description is not complete");

                static constexpr bool use_bluestein = check_and_get_fft_implementation<this_type>::use_bluestein;
                block_preprocess<this_type, use_bluestein>(input, reinterpret_cast<value_type*>(shared_memory));
                internal_execute<use_bluestein>(input, shared_memory, workspace);
                block_postprocess<this_type, use_bluestein>(input, reinterpret_cast<value_type*>(shared_memory));
            }

            // T - can be any type if its alignment and size are the same as those of ::value_type
            template<class T>
            inline __device__ auto execute(T* input, void* shared_memory, workspace_type& workspace) //
                -> CUFFTDX_STD::enable_if_t<!CUFFTDX_STD::is_void<T>::value && (sizeof(T) == sizeof(value_type)) &&
                                            (alignof(T) == alignof(value_type))> {
                return execute(reinterpret_cast<value_type*>(input), shared_memory, workspace);
            }

            template<class T>
            inline __device__ auto execute(T* /* input */, void* /* shared_memory */, workspace_type&
                                           /* workspace */) //
                -> CUFFTDX_STD::enable_if_t<CUFFTDX_STD::is_void<T>::value || (sizeof(T) != sizeof(value_type)) ||
                                            (alignof(T) != alignof(value_type))> {
                static constexpr bool condition =
                    CUFFTDX_STD::is_void<T>::value || (sizeof(T) != sizeof(value_type)) || (alignof(T) != alignof(value_type));
                static_assert(not condition, "Incorrect value type is used, try using ::value_type");
            }

        private:
            template<unsigned int N, class T>
            inline __device__ void shared_to_registers_impl(T* shared_memory, T* thread_data) {
                const unsigned int batch_offset = threadIdx.y * N;
                unsigned int       index        = threadIdx.x;
                for (unsigned int i = 0; i < input_ept; i++) {
                    if (index < N) {
                        thread_data[i] = shared_memory[batch_offset + index];
                    }
                    index += stride;
                }
            }

            template<unsigned int N, class T>
            inline __device__ void registers_to_shared_impl(T* thread_data, T* shared_memory) {
                const unsigned int batch_offset = threadIdx.y * N;
                unsigned int       index        = threadIdx.x;

                for (unsigned int i = 0; i < output_ept; i++) {
                    if (index < N) {
                        shared_memory[batch_offset + index] = thread_data[i];
                    }
                    index += stride;
                }
            }

            template<bool Bluestein, class V>
            inline __device__ void shared_to_registers(void* shared_memory, V* thread_data) {
                using input_t = typename this_type::input_type;
                shared_to_registers_impl<this_type::input_length>(
                    reinterpret_cast<input_t*>(shared_memory),
                    reinterpret_cast<input_t*>(thread_data));
                __syncthreads();
            }

            template<bool Bluestein, class V>
            inline __device__ void registers_to_shared(V* thread_data, void* shared_memory) {
                __syncthreads();
                using output_t = typename this_type::output_type;
                registers_to_shared_impl<this_type::output_length>(
                    reinterpret_cast<output_t*>(thread_data),
                    reinterpret_cast<output_t*>(shared_memory));
            }

            // Cooley-Tukey execution
            template<bool UseBluestein>
            inline __device__ auto internal_execute(value_type* input,
                                                    void*       shared_memory,
                                                    workspace_type& /* workspace */,
                                                    const unsigned int /* fft_id */ = threadIdx.x) //
                -> CUFFTDX_STD::enable_if_t<!UseBluestein> {
                using fft_implementation_t        = check_and_get_fft_implementation_t<this_type>;
                static constexpr auto function_id = fft_implementation_t::function_id;
                using scalar_type                 = typename value_type::value_type;
                database::detail::cufftdx_private_function_wrapper<function_id, scalar_type, 1 /* dynamic */>(input,
                                                                                                              shared_memory);
            }

            // Bluestein execution
            // Assumptions:
            // * fft_id is threadIdx.x -> user must use our block dimension
            template<bool UseBluestein>
            inline __device__ auto internal_execute(value_type*        input,
                                                    void*              shared_memory,
                                                    workspace_type&    workspace,
                                                    const unsigned int fft_id = threadIdx.x) //
                -> CUFFTDX_STD::enable_if_t<UseBluestein> {
#if !defined(NDEBUG) && defined(CUFFTDX_ENABLE_RUNTIME_ASSERTS) && !defined(__CUDACC_RTC__)
                const bool workspace_valid = workspace.valid();
                assert(workspace_valid && "Workspace is invalid, check if workspace was created successfully before passing it to kernel");
#endif

                using scalar_type = typename value_type::value_type;
                using ldg_type    = typename ldg_type<value_type>::type;

                using fft_implementation_t          = check_and_get_fft_implementation_t<this_type>;
                static constexpr auto function_id   = fft_implementation_t::function_id;
                static constexpr auto fft_blue_size = get_bluestein_size(base_type::this_fft_size_v);

                unsigned int                  index  = fft_id;
                static constexpr unsigned int stride = fft_blue_size / elements_per_thread;

                // Only first fft_size values are meaningful, others should be zero.
                static constexpr unsigned int max_meaningful_ept = (base_type::this_fft_size_v + (stride - 1)) / stride;
                // In this case user is expected to zero-padded input.
                // for (unsigned int i = 0; i < max_meaningful_ept; ++i) {
                //     auto v = __ldg((ldg_type*)workspace.w_time + index);
                //     input[i] *= *(reinterpret_cast<value_type*>(&v));
                //     index += stride;
                // }
                // This zeroes the padding.
                for (unsigned int i = 0; i < elements_per_thread; ++i) {
                    // Make swap real<->imag for inverse FFT
                    if (base_type::this_fft_direction_v == fft_direction::inverse) {
                        const auto tmp = input[i].x;
                        input[i].x     = input[i].y;
                        input[i].y     = tmp;
                    }

                    if ((i * stride + fft_id) < base_type::this_fft_size_v) {
                        auto v = __ldg((ldg_type*)workspace.w_time + index);
                        // For half precision we're loading float2 in ldg, so we need
                        // to reinterpret in to complex<__half2> in order to have correct
                        // multiplication performed.
                        input[i] *= *(reinterpret_cast<value_type*>(&v));
                    } else {
                        input[i] = value_type(0., 0.);
                    }
                    index += stride;
                }

                database::detail::cufftdx_private_function_wrapper<function_id, scalar_type, 1 /* dynamic */>(input,
                                                                                                              shared_memory);
                index = fft_id;
                for (unsigned int i = 0; i < elements_per_thread; ++i) {
                    auto v = __ldg((ldg_type*)workspace.w_freq + index);
                    input[i] *= *(reinterpret_cast<value_type*>(&v));
                    input[i].y = -input[i].y; // conjugate
                    index += stride;
                }

                database::detail::cufftdx_private_function_wrapper<function_id, scalar_type, 1 /* dynamic */>(input,
                                                                                                              shared_memory);

                // We can limit the last loop to just max_meaningful_ept, other values are not needed.
                index = fft_id;
                for (unsigned int i = 0; i < max_meaningful_ept; ++i) {
                    input[i].y = -input[i].y; // conjugate
                    // normalize; input[i] /= fft_blue_size; // divide by xsize, for ifft
                    normalize<fft_blue_size>(input[i]);
                    auto v = __ldg((ldg_type*)workspace.w_time + index);
                    input[i] *= *(reinterpret_cast<value_type*>(&v));
                    index += stride;

                    // Make swap real<->imag for inverse FFT
                    if (base_type::this_fft_direction_v == fft_direction::inverse) {
                        const auto tmp = input[i].x;
                        input[i].x     = input[i].y;
                        input[i].y     = tmp;
                    }
                }
            }

            inline static constexpr unsigned int get_shared_memory_size() {
                static_assert(base_type::is_complete, "Can't calculate shared memory, FFT description is not complete");
                using fft_implementation_t = check_and_get_fft_implementation_t<base_type>;
                return fft_implementation_t::shared_memory_size * ffts_per_block;
            }

            inline static constexpr unsigned int get_storage_size() {
                static_assert(base_type::is_complete, "Can't calculate storage_size, FFT description is not complete");
                using fft_implementation_t       = check_and_get_fft_implementation_t<this_type>;
                constexpr bool is_folded = real_fft_mode_of<base_type>::value == real_mode::folded;
                if constexpr (is_folded) {
                    constexpr auto realopt = real_fft_layout_of<base_type>::value;
                    if constexpr (realopt == complex_layout::natural) {
                        return fft_implementation_t::storage_size + 1;
                    } else if constexpr (realopt == complex_layout::packed) {
                        return fft_implementation_t::storage_size;
                    } else if constexpr (realopt == complex_layout::full) {
                        return 2 * fft_implementation_t::storage_size;
                    }
                } else {
                    return fft_implementation_t::storage_size;
                }

                return 0;
            }


        public:
            static constexpr dim3         block_dim = check_and_get_trait<fft_operator::block_dim, base_type>::value;
            static constexpr unsigned int ffts_per_block =
                check_and_get_trait<fft_operator::ffts_per_block, base_type>::value;
            static constexpr unsigned int elements_per_thread           = check_and_get_trait<fft_operator::elements_per_thread, base_type>::value;
            static constexpr unsigned int effective_elements_per_thread = check_and_get_trait<fft_operator::elements_per_thread, base_type>::effective;
            static constexpr unsigned int stride                        = block_dim.x;

            static constexpr unsigned int suggested_ffts_per_block =
                check_and_get_trait<fft_operator::ffts_per_block, base_type>::suggested;
            static constexpr unsigned int suggested_elements_per_thread = check_and_get_trait<fft_operator::elements_per_thread, base_type>::suggested;

            static constexpr unsigned int storage_size       = get_storage_size();
            static constexpr unsigned int shared_memory_size = get_shared_memory_size();

            static constexpr unsigned int max_threads_per_block = block_dim.x * block_dim.y * block_dim.z;

            static constexpr unsigned int implicit_type_batching =
                CUFFTDX_STD::is_same<this_fft_precision_t, __half>::value ? 2 : 1;

            static constexpr bool         requires_workspace = check_and_get_fft_implementation<base_type>::requires_workspace;
            static constexpr unsigned int workspace_size     = check_and_get_fft_implementation<base_type>::workspace_size;

            static constexpr unsigned int input_length  = get_input_length(base_type::this_fft_type_v, base_type::this_fft_complex_layout_v, base_type::this_fft_size_v, fft_execution_size);
            static constexpr unsigned int output_length = get_output_length(base_type::this_fft_type_v, base_type::this_fft_complex_layout_v, base_type::this_fft_size_v, fft_execution_size);

            using input_type                        = CUFFTDX_STD::conditional_t<(!is_folded) && (base_type::this_fft_type_v == fft_type::r2c),
                                                         typename value_type::value_type,
                                                         value_type>;
            static constexpr unsigned int input_ept = get_input_ept(base_type::this_fft_type_v,
                                                                  base_type::this_fft_complex_layout_v,
                                                                  elements_per_thread,
                                                                  effective_elements_per_thread);

            using output_type                        = CUFFTDX_STD::conditional_t<(!is_folded) && (base_type::this_fft_type_v == fft_type::c2r),
                                                          typename value_type::value_type,
                                                          value_type>;
            static constexpr unsigned int output_ept = get_output_ept(base_type::this_fft_type_v,
                                                                    base_type::this_fft_complex_layout_v,
                                                                    elements_per_thread,
                                                                    effective_elements_per_thread);
        };


        // [NOTE] Idea for testing static assert.
        //
        // Switch (macro) which changes behaviour from going to static_asserts
        // to returning description_error type in operator+(). That would required more indirection
        // in creating fft_description and fft_execution types.

        template<class... Operators>
        struct make_description {
        private:
            static constexpr bool has_block_operator =
                has_operator<fft_operator::block, fft_execution<Operators...>>::value;
            static constexpr bool has_thread_operator =
                has_operator<fft_operator::thread, fft_execution<Operators...>>::value;
            static constexpr bool has_execution_operator = has_block_operator || has_thread_operator;

            // Workaround (NVRTC/MSVC)
            //
            // For NVRTC we need to utilize a in-between class called fft_block_execution_partial, otherwise
            // we run into a complation error if Block() is added to description before FFT description is
            // complete, example:
            //
            // Fails on NVRTC:
            //     Size<...>() + Direction<...>() + Type<...>() + Precision<...>() + Block() + SM<700>()
            // Works on NVRTC:
            //     Size<...>() + Direction<...>() + Type<...>() + Precision<...>() + SM<700>() + Block()
            //
            // This workaround disables some useful diagnostics based on static_asserts.
#if defined(__CUDACC_RTC__) || defined(_MSC_VER)
            using operator_wrapper_type = fft_operator_wrapper<Operators...>;
            using fft_block_execution_type =
                CUFFTDX_STD::conditional_t<is_complete_fft<operator_wrapper_type>::value,
                                           fft_block_execution<Operators...>,
                                           fft_block_execution_partial<Operators...>>;
            using fft_thread_execution_type =
                CUFFTDX_STD::conditional_t<is_complete_fft<operator_wrapper_type>::value,
                                           fft_thread_execution<Operators...>,
                                           fft_thread_execution_partial<Operators...>>;
#else
            using fft_block_execution_type  = fft_block_execution<Operators...>;
            using fft_thread_execution_type = fft_thread_execution<Operators...>;
#endif


            using description_type = fft_description<Operators...>;
            using execution_type   = CUFFTDX_STD::conditional_t<has_block_operator,
                                                              fft_block_execution_type,
                                                              fft_thread_execution_type>;

        public:
            using type = CUFFTDX_STD::conditional_t<has_execution_operator, execution_type, description_type>;
        };

        template<class... Operators>
        using make_description_t = typename make_description<Operators...>::type;
    } // namespace detail

    template<class Operator1, class Operator2>
    __host__ __device__ __forceinline__ auto operator+(const Operator1&, const Operator2&) //
        -> CUFFTDX_STD::enable_if_t<commondx::detail::are_operator_expressions<Operator1, Operator2>::value,
                                    detail::make_description_t<Operator1, Operator2>> {
        return detail::make_description_t<Operator1, Operator2>();
    }

    template<class... Operators1, class Operator2>
    __host__ __device__ __forceinline__ auto operator+(const detail::fft_description<Operators1...>&,
                                                       const Operator2&) //
        -> CUFFTDX_STD::enable_if_t<commondx::detail::is_operator_expression<Operator2>::value,
                                    detail::make_description_t<Operators1..., Operator2>> {
        return detail::make_description_t<Operators1..., Operator2>();
    }

    template<class Operator1, class... Operators2>
    __host__ __device__ __forceinline__ auto operator+(const Operator1&,
                                                       const detail::fft_description<Operators2...>&) //
        -> CUFFTDX_STD::enable_if_t<commondx::detail::is_operator_expression<Operator1>::value,
                                    detail::make_description_t<Operator1, Operators2...>> {
        return detail::make_description_t<Operator1, Operators2...>();
    }

    template<class... Operators1, class... Operators2>
    __host__ __device__ __forceinline__ auto operator+(const detail::fft_description<Operators1...>&,
                                                       const detail::fft_description<Operators2...>&) //
        -> detail::make_description_t<Operators1..., Operators2...> {
        return detail::make_description_t<Operators1..., Operators2...>();
    }
} // namespace cufftdx

#undef STRINGIFY
#undef XSTRINGIFY

#endif // CUFFTDX_DETAIL_FFT_EXECUTION_HPP
