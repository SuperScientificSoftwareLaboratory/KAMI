// Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// NVIDIA CORPORATION and its licensors retain all intellectual property
// and proprietary rights in and to this software, related documentation
// and any modifications thereto.  Any use, reproduction, disclosure or
// distribution of this software and related documentation without an express
// license agreement from NVIDIA CORPORATION is strictly prohibited.

#ifndef CUBLASDX_DATABASE_CUTE_EXTENSION_HPP
#define CUBLASDX_DATABASE_CUTE_EXTENSION_HPP

#include "../detail/system_checks.hpp"
#include "cute/config.hpp"

namespace cublasdx {
    namespace detail {

        using cute::Layout;
        using cute::Shape;
        using cute::Stride;
        using cute::Int;
        using cute::Tensor;

        // General FMA
        template <class A, class B, class C>
        CUTE_HOST_DEVICE constexpr
        void
        fma(C& d, A const& a, B const& b, C const& c)
        {
            if constexpr (COMMONDX_STL_NAMESPACE::is_same_v<COMMONDX_STL_NAMESPACE::tuple<A, B, C>, COMMONDX_STL_NAMESPACE::tuple<float_e4m3_t, float_e4m3_t, double>> ||
                          COMMONDX_STL_NAMESPACE::is_same_v<COMMONDX_STL_NAMESPACE::tuple<A, B, C>, COMMONDX_STL_NAMESPACE::tuple<float_e5m2_t, float_e5m2_t, double>> ||
                          COMMONDX_STL_NAMESPACE::is_same_v<COMMONDX_STL_NAMESPACE::tuple<A, B, C>, COMMONDX_STL_NAMESPACE::tuple<float_e4m3_t, float_e4m3_t, float>> ||
                          COMMONDX_STL_NAMESPACE::is_same_v<COMMONDX_STL_NAMESPACE::tuple<A, B, C>, COMMONDX_STL_NAMESPACE::tuple<float_e5m2_t, float_e5m2_t, float>> ||
                          COMMONDX_STL_NAMESPACE::is_same_v<COMMONDX_STL_NAMESPACE::tuple<A, B, C>, COMMONDX_STL_NAMESPACE::tuple<float_e4m3_t, float_e4m3_t, half_t>> ||
                          COMMONDX_STL_NAMESPACE::is_same_v<COMMONDX_STL_NAMESPACE::tuple<A, B, C>, COMMONDX_STL_NAMESPACE::tuple<float_e5m2_t, float_e5m2_t, half_t>> ||
                          COMMONDX_STL_NAMESPACE::is_same_v<COMMONDX_STL_NAMESPACE::tuple<A, B, C>, COMMONDX_STL_NAMESPACE::tuple<float_e4m3_t, float_e4m3_t, bfloat16_t>> ||
                          COMMONDX_STL_NAMESPACE::is_same_v<COMMONDX_STL_NAMESPACE::tuple<A, B, C>, COMMONDX_STL_NAMESPACE::tuple<float_e5m2_t, float_e5m2_t, bfloat16_t>>) {
                d = static_cast<C>(static_cast<float>(a) * static_cast<float>(b) + c);
            } else {
                using cute::fma;
                fma(d, a, b, c);
            }
        }

        template <class A, class B, class C>
        CUTE_HOST_DEVICE constexpr
        void
        fma(cutlass::complex<C>      & d,
            cutlass::complex<A> const& a,
            cutlass::complex<B> const& b,
            cutlass::complex<C> const& c)
        {
            fma(d.real(),  a.real(), b.real(), c.real());
            fma(d.imag(),  a.real(), b.imag(), c.imag());
            fma(d.real(), -a.imag(), b.imag(), d.real());
            fma(d.imag(),  a.imag(), b.real(), d.imag());
        }

        // Universal FMA
        template <class A, class B, class C>
        struct UniversalFMA
        {
            using DRegisters = C[1];
            using ARegisters = A[1];
            using BRegisters = B[1];
            using CRegisters = C[1];

            CUTE_HOST_DEVICE static constexpr void
            fma(C      & d,
                A const& a,
                B const& b,
                C const& c)
            {
                using cublasdx::detail::fma;
                fma(d, a, b, c);
            }
        };

        // SM75 MMAs

        // F16 = F16 * F16 + F16
        struct SM75_16x8x8_F16F16F16F16_TN
        {
            using DRegisters = uint32_t[2];
            using ARegisters = uint32_t[2];
            using BRegisters = uint32_t[1];
            using CRegisters = uint32_t[2];

            CUTE_HOST_DEVICE static void
            fma(uint32_t      & d0, uint32_t      & d1,
                uint32_t const& a0, uint32_t const& a1,
                uint32_t const& b0,
                uint32_t const& c0, uint32_t const& c1)
            {
            #if defined(CUTE_ARCH_MMA_SM75_ENABLED)
                asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                "{%0, %1},"
                "{%2, %3},"
                "{%4},"
                "{%5, %6};\n"
                : "=r"(d0), "=r"(d1)
                :  "r"(a0),  "r"(a1),
                    "r"(b0),
                    "r"(c0),  "r"(c1));
            #else
                CUTE_INVALID_CONTROL_PATH("Attempting to use SM75_16x8x8_F16F16F16F16_TN without CUTE_ARCH_MMA_SM75_ENABLED");
            #endif
            }
        };

        // F32 = F16 * F16 + F32
        struct SM75_16x8x8_F32F16F16F32_TN
        {
            using DRegisters = float[4];
            using ARegisters = uint32_t[2];
            using BRegisters = uint32_t[1];
            using CRegisters = float[4];

            CUTE_HOST_DEVICE static void
            fma(float         & d0, float         & d1, float         & d2, float         & d3,
                uint32_t const& a0, uint32_t const& a1,
                uint32_t const& b0,
                float const   & c0, float const   & c1, float const   & c2, float const   & c3)
            {
            #if defined(CUTE_ARCH_MMA_SM75_ENABLED)
                asm volatile(
                "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
                "{%0,  %1,  %2,  %3},"
                "{%4,  %5},"
                "{%6},"
                "{%7,  %8,  %9,  %10};\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                :  "r"(a0),  "r"(a1),
                    "r"(b0),
                    "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3));
            #else
                CUTE_INVALID_CONTROL_PATH("Attempting to use SM75_16x8x8_F32F16F16F32_TN without CUTE_ARCH_MMA_SM75_ENABLED");
            #endif
            }
        };

        struct SM89_16x8x32_F32F8F8F32_TN
        {
            using DRegisters = float[4];
            using ARegisters = uint32_t[4];
            using BRegisters = uint32_t[2];
            using CRegisters = float[4];

            // fma() defined in derived class
        };

        // SM89 F32 = F8 * F8 + F32 MMA's

        // F32 = fe4m3 * fe4m3 + F32

        struct SM89_16x8x32_F32E4M3E4M3F32_TN : public SM89_16x8x32_F32F8F8F32_TN
        {
            CUTE_HOST_DEVICE static void
            fma(float         & d0, float         & d1, float         & d2, float         & d3,
                uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
                uint32_t const& b0, uint32_t const& b1,
                float    const& c0, float    const& c1, float    const& c2, float    const& c3)
            {
            #if defined(CUBLASDX_ARCH_MMA_SM89_ENABLED)
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)    // float
                    :
                    "r"(a0),  "r"(a1),  "r"(a2),  "r"(a3),    // uint32_t
                    "r"(b0),  "r"(b1),                        // uint32_t
                    "f"(c0),  "f"(c1),  "f"(c2),  "f"(c3)     // float
                );
            #else
                CUTE_INVALID_CONTROL_PATH("Attempting to use SM89_16x8x32_F32E4M3E4M3F32_TN without SM89 ENABLED and NVCC 12.4 or above");
            #endif
            }
        };

        // F32 = fe4m3 * fe5m2 + F32

        struct SM89_16x8x32_F32E4M3E5M2F32_TN : public SM89_16x8x32_F32F8F8F32_TN
        {
            CUTE_HOST_DEVICE static void
            fma(float         & d0, float         & d1, float         & d2, float         & d3,
                uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
                uint32_t const& b0, uint32_t const& b1,
                float    const& c0, float    const& c1, float    const& c2, float    const& c3)
            {
            #if defined(CUBLASDX_ARCH_MMA_SM89_ENABLED)
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    :
                        "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                        "r"(b0), "r"(b1),
                        "f"(c0), "f"(c1), "f"(c2), "f"(c3)
                );
            #else
                CUTE_INVALID_CONTROL_PATH("Attempting to use SM89_16x8x32_F32E4M3E5M2F32_TN without SM89 ENABLED and NVCC 12.4 or above");
            #endif
            }
        };

        // F32 = fe5m2 * fe4m3 + F32

        struct SM89_16x8x32_F32E5M2E4M3F32_TN : public SM89_16x8x32_F32F8F8F32_TN
        {
            CUTE_HOST_DEVICE static void
            fma(float         & d0, float         & d1, float         & d2, float         & d3,
                uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
                uint32_t const& b0, uint32_t const& b1,
                float    const& c0, float    const& c1, float    const& c2, float    const& c3)
            {
            #if defined(CUBLASDX_ARCH_MMA_SM89_ENABLED)
                asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e4m3.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                :
                    "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                    "r"(b0), "r"(b1),
                    "f"(c0), "f"(c1), "f"(c2), "f"(c3)
                );
            #else
                CUTE_INVALID_CONTROL_PATH("Attempting to use SM89_16x8x32_F32E5M2E4M3F32_TN without SM89 ENABLED and NVCC 12.4 or above");
            #endif
            }
        };

        // FP32 = fe5m2 * fe5m2 + F32
        struct SM89_16x8x32_F32E5M2E5M2F32_TN : public SM89_16x8x32_F32F8F8F32_TN
        {
            CUTE_HOST_DEVICE static void
            fma(float         & d0, float         & d1, float         & d2, float         & d3,
                uint32_t const& a0, uint32_t const& a1, uint32_t const& a2, uint32_t const& a3,
                uint32_t const& b0, uint32_t const& b1,
                float    const& c0, float    const& c1, float    const& c2, float    const& c3)
            {
            #if defined(CUBLASDX_ARCH_MMA_SM89_ENABLED)
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                    :
                        "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                        "r"(b0), "r"(b1),
                        "f"(c0), "f"(c1), "f"(c2), "f"(c3)
                );
            #else
                CUTE_INVALID_CONTROL_PATH("Attempting to use SM89_16x8x32_F32E5M2E5M2F32_TN without SM89 ENABLED and NVCC 12.4 or above");
            #endif
            }
        };

        using SM89_FP8_A = Layout<Shape <Shape < cute::_4,cute::_8>,Shape < cute::_4,cute::_2,cute::_2  >>,
                                  Stride<Stride<cute::_64,cute::_1>,Stride<cute::_16,cute::_8,cute::_256>>>;
        using SM89_FP8_B = Layout<Shape <Shape < cute::_4,cute::_8>,Shape <cute::_4,cute::_2  >>,
                                  Stride<Stride<cute::_32,cute::_1>,Stride<cute::_8,cute::_128>>>;
        using SM89_FP8_C = Layout<Shape <Shape < cute::_4,cute::_8>,Shape < cute::_2,cute::_2>>,
                                  Stride<Stride<cute::_32,cute::_1>,Stride<cute::_16,cute::_8>>>;

        template<typename A, typename B>
        struct MMA_Traits_SM89_16x8x32_F32F8F8F32_TN
        {
            using ValTypeD = float;
            using ValTypeA = A;
            using ValTypeB = B;
            using ValTypeC = float;

            using Shape_MNK = Shape<cute::_16,cute::_8,cute::_32>;
            using ThrID   = Layout<cute::_32>;
            using ALayout = SM89_FP8_A;
            using BLayout = SM89_FP8_B;
            using CLayout = SM89_FP8_C;
        };

        // Complex MMAs
        template<class T, class Layout, class X>
        CUTE_HOST_DEVICE void riri_to_rrii(Tensor<T, Layout> const& complex_in, X* real_out, X* imag_out) {
            CUTE_UNROLL
            for (unsigned i = 0; i < size(complex_in); i++) {
                real_out[i] = complex_in[i].real();
                imag_out[i] = complex_in[i].imag();
            }
        }

        template<class T, class Layout, class X>
        CUTE_HOST_DEVICE void rrii_to_riri(Tensor<T, Layout>& complex_out, X* const real_in, X* const imag_in) {
            CUTE_UNROLL
            for (unsigned i = 0; i < size(complex_out); i++) {
                complex_out[i].real(real_in[i]);
                complex_out[i].imag(imag_in[i]);
            }
        }

        template<class X>
        CUTE_HOST_DEVICE void negate(X* a, unsigned nelem) {
            CUTE_UNROLL
            for(unsigned i = 0 ; i < nelem; i++) {
                a[i] = -a[i];
            }
        }

        using cute::MMA_Traits;
        template<class RealMMA,
                 typename d_value_type = typename MMA_Traits<RealMMA>::ValTypeD,
                 typename a_value_type = typename MMA_Traits<RealMMA>::ValTypeA,
                 typename b_value_type = typename MMA_Traits<RealMMA>::ValTypeB,
                 typename c_value_type = typename MMA_Traits<RealMMA>::ValTypeC>

        struct ComplexMMA
        {
            using DRegisters = complex<d_value_type>[(2 * sizeof(typename RealMMA::DRegisters)) / sizeof(complex<d_value_type>)]; // Number of registers are picked so we have
                                                                                                                                  // 2 (one for real and one for imaginary) x sizeof(RealMMA::A/B/C/D Registers) = sizeof(ComplexMMA::A/B/C/D Registers)
            using ARegisters = complex<a_value_type>[(2 * sizeof(typename RealMMA::ARegisters)) / sizeof(complex<a_value_type>)];
            using BRegisters = complex<b_value_type>[(2 * sizeof(typename RealMMA::BRegisters)) / sizeof(complex<b_value_type>)];
            using CRegisters = complex<c_value_type>[(2 * sizeof(typename RealMMA::CRegisters)) / sizeof(complex<c_value_type>)];

            template <class TD, class DLayout, class TA, class ALayout, class TB, class BLayout, class TC, class CLayout>
            CUTE_HOST_DEVICE static void
            fma(Tensor<TD, DLayout>& rD, Tensor<TA, ALayout> const& rA, Tensor<TB, BLayout> const& rB, Tensor<TC, CLayout> const& rC)
            {
                typename RealMMA::DRegisters real_d, imag_d;
                typename RealMMA::ARegisters real_a, imag_a;
                typename RealMMA::BRegisters real_b, imag_b;
                typename RealMMA::CRegisters real_c, imag_c;

                constexpr int RealMMARegNumD = cute::extent<typename RealMMA::DRegisters>::value;
                constexpr int RealMMARegNumA = cute::extent<typename RealMMA::ARegisters>::value;
                constexpr int RealMMARegNumB = cute::extent<typename RealMMA::BRegisters>::value;
                constexpr int RealMMARegNumC = cute::extent<typename RealMMA::CRegisters>::value;

                riri_to_rrii(rA, reinterpret_cast<a_value_type*>(real_a), reinterpret_cast<a_value_type*>(imag_a));
                riri_to_rrii(rB, reinterpret_cast<b_value_type*>(real_b), reinterpret_cast<b_value_type*>(imag_b));
                riri_to_rrii(rC, reinterpret_cast<c_value_type*>(real_c), reinterpret_cast<c_value_type*>(imag_c));

                // d.real() =  a.real() * b.real() + c.real();
                cute::detail::explode(RealMMA::fma,
                                      real_d, cute::make_int_sequence<RealMMARegNumD>{},
                                      real_a, cute::make_int_sequence<RealMMARegNumA>{},
                                      real_b, cute::make_int_sequence<RealMMARegNumB>{},
                                      real_c, cute::make_int_sequence<RealMMARegNumC>{});

                // d.imag() =  a.imag() * b.real() + c.imag();
                cute::detail::explode(RealMMA::fma,
                                      imag_d, cute::make_int_sequence<RealMMARegNumD>{},
                                      imag_a, cute::make_int_sequence<RealMMARegNumA>{},
                                      real_b, cute::make_int_sequence<RealMMARegNumB>{},
                                      imag_c, cute::make_int_sequence<RealMMARegNumC>{});

                // d.real() = -a.imag() * b.imag() + d.real();
                negate(reinterpret_cast<a_value_type*>(imag_a), sizeof(typename RealMMA::ARegisters) / sizeof(a_value_type));
                cute::detail::explode(RealMMA::fma,
                                      real_d, cute::make_int_sequence<RealMMARegNumD>{},
                                      imag_a, cute::make_int_sequence<RealMMARegNumA>{},
                                      imag_b, cute::make_int_sequence<RealMMARegNumB>{},
                                      real_d, cute::make_int_sequence<RealMMARegNumD>{});

                // d.imag() =  a.real() * b.imag() + d.imag();
                cute::detail::explode(RealMMA::fma,
                                      imag_d, cute::make_int_sequence<RealMMARegNumD>{},
                                      real_a, cute::make_int_sequence<RealMMARegNumA>{},
                                      imag_b, cute::make_int_sequence<RealMMARegNumB>{},
                                      imag_d, cute::make_int_sequence<RealMMARegNumD>{});

                rrii_to_riri(rD, reinterpret_cast<d_value_type*>(real_d), reinterpret_cast<d_value_type*>(imag_d));
            }
        };
    } // namespace detail
} // cublasdx

namespace cute {

    template <class A, class B, class C>
    struct MMA_Traits<cublasdx::detail::UniversalFMA<A, B, C>> : MMA_Traits<UniversalFMA<C, A, B, C>> {};

    template <>
    struct MMA_Traits<cublasdx::detail::SM75_16x8x8_F16F16F16F16_TN>
    {
        using ValTypeD = half_t;
        using ValTypeA = half_t;
        using ValTypeB = half_t;
        using ValTypeC = half_t;

        using Shape_MNK = Shape<_16,_8,_8>;
        using ThrID   = Layout<_32>;
        using ALayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                               Stride<Stride<_32,_1>,Stride<_16,_8>>>;
        using BLayout = Layout<Shape< Shape < _4,_8>,_2>,
                               Stride<Stride<_16,_1>,_8>>;
        using CLayout = Layout<Shape <Shape < _4,_8>,Shape < _2,_2>>,
                               Stride<Stride<_32,_1>,Stride<_16,_8>>>;
    };

    template <>
    struct MMA_Traits<cublasdx::detail::SM75_16x8x8_F32F16F16F32_TN> : MMA_Traits<cublasdx::detail::SM75_16x8x8_F16F16F16F16_TN>
    {
        using ValTypeD = float;
        using ValTypeA = half_t;
        using ValTypeB = half_t;
        using ValTypeC = float;
    };

    template <> struct MMA_Traits<cublasdx::detail::SM89_16x8x32_F32E4M3E4M3F32_TN> : public cublasdx::detail::MMA_Traits_SM89_16x8x32_F32F8F8F32_TN<cutlass::float_e4m3_t, cutlass::float_e4m3_t> {};
    template <> struct MMA_Traits<cublasdx::detail::SM89_16x8x32_F32E4M3E5M2F32_TN> : public cublasdx::detail::MMA_Traits_SM89_16x8x32_F32F8F8F32_TN<cutlass::float_e4m3_t, cutlass::float_e5m2_t> {};
    template <> struct MMA_Traits<cublasdx::detail::SM89_16x8x32_F32E5M2E4M3F32_TN> : public cublasdx::detail::MMA_Traits_SM89_16x8x32_F32F8F8F32_TN<cutlass::float_e5m2_t, cutlass::float_e4m3_t> {};
    template <> struct MMA_Traits<cublasdx::detail::SM89_16x8x32_F32E5M2E5M2F32_TN> : public cublasdx::detail::MMA_Traits_SM89_16x8x32_F32F8F8F32_TN<cutlass::float_e5m2_t, cutlass::float_e5m2_t> {};

    template <class RealMMA>
    struct MMA_Traits<cublasdx::detail::ComplexMMA<RealMMA>> : MMA_Traits<RealMMA>
    {
        using ValTypeD = complex<typename MMA_Traits<RealMMA>::ValTypeD>;
        using ValTypeA = complex<typename MMA_Traits<RealMMA>::ValTypeA>;
        using ValTypeB = complex<typename MMA_Traits<RealMMA>::ValTypeB>;
        using ValTypeC = complex<typename MMA_Traits<RealMMA>::ValTypeC>;

        template <class TD, class DLayout,
                  class TA, class ALayout,
                  class TB, class BLayout,
                  class TC, class CLayout>
        CUTE_HOST_DEVICE constexpr friend
        void
        mma_unpack(MMA_Traits          const& traits,
                   Tensor<TD, DLayout>      & D,
                   Tensor<TA, ALayout> const& A,
                   Tensor<TB, BLayout> const& B,
                   Tensor<TC, CLayout> const& C)
        {
            static_assert(is_rmem<TD>::value, "Expected registers in MMA_Atom::call");
            static_assert(is_rmem<TA>::value, "Expected registers in MMA_Atom::call");
            static_assert(is_rmem<TB>::value, "Expected registers in MMA_Atom::call");
            static_assert(is_rmem<TC>::value, "Expected registers in MMA_Atom::call");

            using complex_mma = cublasdx::detail::ComplexMMA<RealMMA>;

            // Register value types from the MMA_Operation register arrays
            using RegTypeD = typename remove_extent<typename complex_mma::DRegisters>::type;
            using RegTypeA = typename remove_extent<typename complex_mma::ARegisters>::type;
            using RegTypeB = typename remove_extent<typename complex_mma::BRegisters>::type;
            using RegTypeC = typename remove_extent<typename complex_mma::CRegisters>::type;

            constexpr int RegNumD = extent<typename complex_mma::DRegisters>::value;
            constexpr int RegNumA = extent<typename complex_mma::ARegisters>::value;
            constexpr int RegNumB = extent<typename complex_mma::BRegisters>::value;
            constexpr int RegNumC = extent<typename complex_mma::CRegisters>::value;

            Tensor rA = recast<RegTypeA>(A);
            Tensor rB = recast<RegTypeB>(B);
            Tensor rD = recast<RegTypeD>(D);
            Tensor rC = recast<RegTypeC>(C);

            CUTE_STATIC_ASSERT_V(size(rA) == Int<RegNumA>{});
            CUTE_STATIC_ASSERT_V(size(rB) == Int<RegNumB>{});
            CUTE_STATIC_ASSERT_V(size(rD) == Int<RegNumD>{});
            CUTE_STATIC_ASSERT_V(size(rC) == Int<RegNumC>{});

            complex_mma::fma(rD, rA, rB, rC);
        }
    };
} // namespace cute

namespace cublasdx {
    namespace detail {
        using SM70_8x8x4_C16C16C16C16_TN       = ComplexMMA<cute::SM70_8x8x4_F16F16F16F16_TN>;
        using SM70_8x8x4_C32C16C16C32_TN       = ComplexMMA<cute::SM70_8x8x4_F32F16F16F32_TN>;
        using SM75_16x8x8_C16C16C16C16_TN      = ComplexMMA<SM75_16x8x8_F16F16F16F16_TN>;
        using SM80_16x8x16_C16C16C16C16_TN     = ComplexMMA<cute::SM80_16x8x16_F16F16F16F16_TN>;
        using SM75_16x8x8_C32C16C16C32_TN      = ComplexMMA<SM75_16x8x8_F32F16F16F32_TN>;
        using SM80_16x8x16_C32C16C16C32_TN     = ComplexMMA<cute::SM80_16x8x16_F32F16F16F32_TN>;
        using SM80_16x8x8_C32BC16BC16C32_TN    = ComplexMMA<cute::SM80_16x8x8_F32BF16BF16F32_TN>;
        using SM80_16x8x16_C32BC16BC16C32_TN   = ComplexMMA<cute::SM80_16x8x16_F32BF16BF16F32_TN>;
        using SM80_16x8x4_C32TC32TC32C32_TN    = ComplexMMA<cute::SM80_16x8x4_F32TF32TF32F32_TN>;
        using SM80_16x8x8_C32TC32TC32C32_TN    = ComplexMMA<cute::SM80_16x8x8_F32TF32TF32F32_TN>;
        using SM80_8x8x4_C64C64C64C64_TN       = ComplexMMA<cute::SM80_8x8x4_F64F64F64F64_TN>;
        using SM89_16x8x32_C32CE4M3CE4M3C32_TN = ComplexMMA<SM89_16x8x32_F32E4M3E4M3F32_TN>;
        using SM89_16x8x32_C32CE4M3CE5M2C32_TN = ComplexMMA<SM89_16x8x32_F32E4M3E5M2F32_TN>;
        using SM89_16x8x32_C32CE5M2CE4M3C32_TN = ComplexMMA<SM89_16x8x32_F32E5M2E4M3F32_TN>;
        using SM89_16x8x32_C32CE5M2CE5M2C32_TN = ComplexMMA<SM89_16x8x32_F32E5M2E5M2F32_TN>;
    }
} // namespace cublasdx

#endif // CUBLASDX_DATABASE_CUTE_EXTENSION_HPP
