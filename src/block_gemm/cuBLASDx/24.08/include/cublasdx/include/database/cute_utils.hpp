// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_CUTE_UTILS_HPP
#define CUBLASDX_CUTE_UTILS_HPP

#include "cute_tensor.hpp"

namespace cublasdx {
    struct conjugate {
        // For non-complex types (ie. real type) conjugate does nothing
        template<class T>
        __device__ __host__ __forceinline__
        T operator()(T value) const {
            return cutlass::conj(value);
        }

        template<class T>
        __device__ __host__ __forceinline__
        complex<T> operator()(complex<T> value) const {
            return complex<T>{value.real(), -value.imag()};
        }
    };

    using cute::identity;

    namespace detail {
        template<class F1, class F2>
        struct composed_functor {
            template<class T>
            __device__ __forceinline__ T operator()(T value) const {
                return f2(f1(value));
            }

            const F1 f1;
            const F2 f2;
        };

        template<class F1, class F2>
        __device__ __host__ __forceinline__
        auto compose_functors(const F1& f1, const F2& f2) {
            if constexpr(COMMONDX_STL_NAMESPACE::is_same_v<F1, identity> && COMMONDX_STL_NAMESPACE::is_same_v<F2, identity>) {
                return f1;
            } else if constexpr(COMMONDX_STL_NAMESPACE::is_same_v<F1, identity>) {
                return f2;
            } else if constexpr(COMMONDX_STL_NAMESPACE::is_same_v<F2, identity>) {
                return f1;
            } else {
                return composed_functor<F1, F2> {f1, f2};
            }
        }


    enum class matrix {
        A, B, C
    };

    template<matrix M, typename ArgA, typename ArgB, typename ArgC>
    constexpr __device__ __forceinline__
    auto choose(const ArgA& a, const ArgB& b, const ArgC& c) {
        if constexpr(M == matrix::A) {
            return a;
        } else if constexpr(M == matrix::B) {
            return b;
        } else {
            return c;
        }
    }
        namespace cute_backend {
            // CuTe backend details
            using cute::Int;
            using cute::Layout;
            using cute::Shape;
            using cute::Tensor;

            template<cublasdx::arrangement A, typename D1, typename D2, typename LD>
            constexpr auto make_layout_from_arrangement(D1, D2, LD ld) {
                static_assert(cute::is_integral<D1>::value && cute::is_integral<D2>::value &&
                              cute::is_integral<LD>::value);
                static_assert(cute::is_static_v<D1> && cute::is_static_v<D2>);

                constexpr bool is_rm = (A == cublasdx::row_major);

                using shape_t = Shape<D1, D2>;

                if constexpr (cute::is_static_v<LD>) {
                    constexpr auto ldval = cute::get<0>(ld);
                    using stride_t       = cute::Stride<Int<is_rm ? ldval : 1>, Int<is_rm ? 1 : ldval>>;

                    return Layout<shape_t, stride_t> {};
                } else {
                    const auto stride = cute::make_stride(is_rm ? ld : 1u, is_rm ? 1u : ld);
                    return cute::make_layout(shape_t {}, stride);
                }
            }

            template<typename Layout>
            __forceinline__ constexpr __device__ __host__ auto
            swap_layout_modes(const Layout& l) {
                return cute::select<1, 0>(l);
            }

            template<typename Swizzle, typename Offset, typename Layout>
            __forceinline__ constexpr __device__ __host__ auto
            swap_layout_modes(const cute::ComposedLayout<Swizzle, Offset, Layout>& l) {
                return cute::composition(l.layout_a(),
                    l.offset(),
                    cute::select<1, 0>(l.layout_b()));
            }

            template<typename T, typename Layout>
            __forceinline__ constexpr __device__ __host__ auto
            swap_tensor_modes(const cute::Tensor<T, Layout>& t) {
                return cute::make_tensor(t.data(), swap_layout_modes(t.layout()));
            }

            template<transpose_mode TM>
            constexpr auto get_load_op_from_transpose() {
                if constexpr (TM == transpose_mode::conj_transposed) {
                    return conjugate {};
                } else {
                    return identity {};
                }
            }
        } // namespace cute_backend
    }     // namespace detail
} // namespace cublasdx

#endif // CUBLASDX_CUTE_UTILS_HPP
