// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DETAIL_TENSOR_HPP
#define CUBLASDX_DETAIL_TENSOR_HPP

#include "commondx/detail/stl/type_traits.hpp"
#include "commondx/detail/stl/tuple.hpp"

#include "../database/cute.hpp"

namespace cublasdx {
    namespace detail {
        struct gmem_tag {};
        struct smem_tag {};
        struct rmem_tag {};

        template<class PointerTag, class Layout>
        struct pointer_layout {
            PointerTag pointer_tag;
            Layout layout;

        private:
            template<class T, class K>
            static constexpr auto make_ptr_impl(K* ptr) {
                if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<PointerTag, gmem_tag>) {
                    return cute::make_gmem_ptr<T>(ptr);
                } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<PointerTag, smem_tag>) {
                    return cute::make_smem_ptr<T>(ptr);
                } else if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<PointerTag, rmem_tag>) {
                    return cute::make_rmem_ptr<T>(ptr);
                }
            }

        public:
            constexpr pointer_layout(PointerTag pt, Layout l) : pointer_tag(pt), layout(l) {}

            template<class T>
            static constexpr auto make_ptr(T* ptr) {
                return make_ptr_impl<T, T>(ptr);
            }

            template<class T>
            static constexpr auto make_ptr(void* ptr) {
                return make_ptr_impl<T, void>(ptr);
            }

            template<class T>
            static constexpr auto make_ptr(const void* ptr) {
                return make_ptr_impl<T, const void>(ptr);
            }
        };
    } // namespace detail

    template<class Engine, class Layout>
    using tensor = cute::Tensor<Engine, Layout>;

    template<class Iterator, class... Args>
    __device__ __host__ constexpr auto make_tensor(const Iterator& iter, Args const&... args) {
        return cute::make_tensor(iter, args...);
    }

    template<class T, class PointerTag, class Layout>
    __device__ __host__ constexpr auto make_tensor(T* ptr, const detail::pointer_layout<PointerTag, Layout>& pl) {
        using pl_t = typename detail::pointer_layout<PointerTag, Layout>;
        return cute::make_tensor(pl_t::make_ptr(ptr), pl.layout);
    }

    template<class T, class PointerTag, class Layout>
    __device__ __host__ constexpr auto make_tensor(void* ptr, const detail::pointer_layout<PointerTag, Layout>& pl) {
        using pl_t = typename detail::pointer_layout<PointerTag, Layout>;
        return cute::make_tensor(pl_t::template make_ptr<T>(ptr), pl.layout);
    }

    template<class T, class PointerTag, class Layout>
    __device__ __host__ constexpr auto make_tensor(const void*                                        ptr,
                                                   const detail::pointer_layout<PointerTag, Layout>& pl) {
        using pl_t = typename detail::pointer_layout<PointerTag, Layout>;
        return cute::make_tensor(pl_t::template make_ptr<T>(ptr), pl.layout);
    }
} // namespace cublasdx

#endif // CUBLASDX_DETAIL_TENSOR_HPP
