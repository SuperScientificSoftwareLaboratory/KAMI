// Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUFFTDX_TRAITS_FFT_TRAITS_HPP
#define CUFFTDX_TRAITS_FFT_TRAITS_HPP

#include "../detail/fft_description_fd.hpp"

#include "../operators.hpp"

#include "detail/description_traits.hpp"
#include "detail/make_complex_type.hpp"

#include "commondx/traits/dx_traits.hpp"

namespace cufftdx {
    template<class Description>
    struct size_of {
    private:
        static constexpr bool has_size = detail::has_operator<fft_operator::size, Description>::value;
        static_assert(has_size, "Description does not have size defined");

    public:
        using value_type                  = unsigned int;
        static constexpr value_type value = detail::get_t<fft_operator::size, Description>::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    constexpr unsigned int size_of<Description>::value;

    // sm_of
    template<class Description>
    using sm_of = commondx::sm_of<fft_operator, Description>;

    template<class Description>
    inline constexpr unsigned int sm_of_v = sm_of<Description>::value;

    // block_dim_of
    template<class Description>
    using block_dim_of = commondx::block_dim_of<fft_operator, Description>;

    template<class Description>
    inline constexpr dim3 block_dim_of_v = block_dim_of<Description>::value;

    template<class Description>
    struct type_of {
        using value_type = fft_type;
        static constexpr value_type value =
            detail::get_or_default_t<fft_operator::type, Description, Type<fft_type::c2c>>::value;
        constexpr operator value_type() const noexcept { return value; }
    };

    template<class Description>
    constexpr fft_type type_of<Description>::value;

    template<class Description>
    struct real_fft_layout_of {
        using value_type = complex_layout;
        static constexpr value_type value =
            detail::get_or_default_t<fft_operator::real_fft_options, Description, RealFFTOptions<complex_layout::natural, real_mode::normal>>::layout;
        constexpr operator value_type() const noexcept { return value; }
    };

    template<class Description>
    constexpr complex_layout real_fft_layout_of<Description>::value;

    template<class Description>
    struct real_fft_mode_of {
        using value_type = real_mode;
        static constexpr value_type value =
            detail::get_or_default_t<fft_operator::real_fft_options, Description, RealFFTOptions<complex_layout::natural, real_mode::normal>>::mode;
        constexpr operator value_type() const noexcept { return value; }
    };

    template<class Description>
    constexpr real_mode real_fft_mode_of<Description>::value;

    template<class Description>
    struct direction_of {
    private:
        using deduced_fft_direction = detail::deduce_direction_type_t<Type<type_of<Description>::value>>;
        using this_fft_direction =
            detail::get_or_default_t<fft_operator::direction, Description, deduced_fft_direction>;

        static_assert(!CUFFTDX_STD::is_void<this_fft_direction>::value,
                      "Description has neither direction defined, nor it can be deduced from its type");

    public:
        using value_type                  = fft_direction;
        static constexpr value_type value = this_fft_direction::value;
        constexpr                   operator value_type() const noexcept { return value; }
    };

    template<class Description>
    constexpr fft_direction direction_of<Description>::value;

    template<class Description>
    using precision_of = commondx::precision_of<fft_operator, Description, detail::default_fft_precision_operator>;

    template<class Description>
    using precision_of_t = typename precision_of<Description>::type;

    template<class Description>
    using is_fft = commondx::is_dx_expression<Description>;

    template<class Description>
    using is_fft_execution = commondx::is_dx_execution_expression<fft_operator, Description>;

    template<class Description>
    using is_complete_fft = commondx::is_complete_dx_expression<Description, detail::is_complete_description>;

    template<class Description>
    using is_complete_fft_execution =
        commondx::is_complete_dx_execution_expression<fft_operator, Description, detail::is_complete_description>;

    template<class Description>
    using extract_fft_description = commondx::extract_dx_description<detail::fft_description, Description, fft_operator>;

    template<class Description>
    using extract_fft_description_t = typename extract_fft_description<Description>::type;

    namespace detail {
        template<class Description>
        struct convert_to_fft_description {
            using type = void;
        };

        template<template<class...> class Description, class... Types>
        struct convert_to_fft_description<Description<Types...>> {
            using type = typename detail::fft_description<Types...>;
        };
    } // namespace detail
} // namespace cufftdx

#endif // CUFFTDX_TRAITS_FFT_TRAITS_HPP
