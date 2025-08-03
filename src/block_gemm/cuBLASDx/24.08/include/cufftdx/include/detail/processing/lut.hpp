#ifndef CUFFTDX_DETAIL_PROCESSING_LUT_HPP
#define CUFFTDX_DETAIL_PROCESSING_LUT_HPP

#include <cuda_fp16.h>

namespace cufftdx {

    namespace detail {

        constexpr double fp64_sincos_coeffs[15] = {
            0x1p+0,
            0x1p+0,
            -0x1p-1,
            -0x1.5555555555548p-3,
            0x1.5555555555551p-5,
            0x1.111111110f7dp-7,
            -0x1.6c16c16c15d69p-10,
            -0x1.a01a019bfdf03p-13,
            0x1.a01a019ddec33p-16,
            0x1.71de3567d4896p-19,
            -0x1.27e4f8e26b8e3p-22,
            -0x1.ae5e5a9291691p-26,
            0x1.1eea7d67fad92p-29,
            0x1.5d8fd1fcf0ec1p-33,
            -0x1.8ff8d5a8f03dbp-37};

        constexpr double fp64_range_coeffs[3] = {
            0x1.921fb54442d18p+0,
            0x1.1a62633145cp-54,
            0x1.b839a252049cp-104};

        constexpr float fp32_sincos_coeffs[9] = {
            0x1p+0f,
            0x1p+0f,
            -0x1p-1f,
            -0x1.555546p-3f,
            0x1.55554ap-5f,
            0x1.11073cp-7f,
            -0x1.6c0c34p-10f,
            -0x1.9943f2p-13f,
            0x1.99eb9cp-16f};

        constexpr float fp32_range_coeffs[2] = {
            0x1.921fb4p+0f,
            0x1.4442d2p-24f};

        template<typename Precision>
        struct numeric {
            static_assert(CUFFTDX_STD::is_same_v<Precision, float> ||
                              CUFFTDX_STD::is_same_v<Precision, double>,
                          "The only precomputed precision are float and double");

            static constexpr bool is_float = CUFFTDX_STD::is_same_v<Precision, float>;

            static constexpr Precision pi  = 3.14159265358979323846264;
            static constexpr Precision tau = 2 * pi;

            static __device__ __forceinline__ constexpr Precision get_sincos_coeff(unsigned int idx) {
                if constexpr (is_float) {
                    return fp32_sincos_coeffs[idx];
                } else {
                    return fp64_sincos_coeffs[idx];
                }

                // For compilers not fully compliant with the standard.
                return Precision {};
            }

            static __device__ __forceinline__ constexpr Precision get_range_coeff(unsigned int idx) {
                if constexpr (is_float) {
                    return fp32_range_coeffs[idx];
                } else {
                    return fp64_range_coeffs[idx];
                }

                // For compilers not fully compliant with the standard.
                return Precision {};
            }

            static __device__ __forceinline__
                Precision
                fma(const Precision a, const Precision b, const Precision c) {
                if constexpr (is_float) {
                    return __fmaf_rn(a, b, c);
                } else {
                    return __fma_rn(a, b, c);
                }

                // For compilers not fully complying with the standard.
                return Precision {};
            }

            static __device__ __forceinline__ int to_int(const Precision v) {
                if constexpr (is_float) {
                    return __float2int_rn(v);
                } else {
                    return __double2int_rn(v);
                }

                // For compilers not fully complying with the standard.
                return 0;
            }
        };

        // This fold expression helps in calculating sine and cosine values,
        // by repeatedly multiplying the input value with a constant, and then
        // adding a precomputed coefficient. It does so with a use of a
        // precision specific FMA instruction. As a compile-time input this takes
        // indices of coefficients in the precision related table. It's used
        // purely for code brevity, and has no performance effect
        template<typename T, int FirstIndex, int... Indices>
        inline __device__
            T
            fma_sincos_repeat(const T a) {
            using numerics = detail::numeric<T>;
            const T a_s    = a * a;
            T       c1     = numerics::get_sincos_coeff(FirstIndex);
            ([&] {
                c1 = numerics::fma(c1, a_s, numerics::get_sincos_coeff(Indices));
            }(),
             ...);

            return c1;
        }

        // This fold expression helps in reducing the twiddle range to
        // [-PI/4, PI/4]. It repeatedly adds the product of a pre-computed
        // coefficient and an input constant to the output value. It does so
        // with a use of a precision specific FMA instruction. As a compile-time
        // input this takes indices of coefficients in the precision related table.
        // It's used purely for code brevity, and has no performance effect
        template<typename T, int... Indices>
        inline __device__
            T
            fma_range_repeat(T a, const T q) {
            using numerics = detail::numeric<T>;
            ([&] {
                a = numerics::fma(q, numerics::get_range_coeff(Indices), a);
            }(),
             ...);

            return a;
        }

        // This function approximates the value of (cos, sin) with use
        // of precomputed constants, limited to [-PI/4, PI/4] on input
        template<typename Precision>
        __forceinline__ __device__ cufftdx::detail::complex<Precision>
                                   approx_twiddle(const Precision a) {
            using complex_type = cufftdx::detail::complex<Precision>;
            if constexpr (CUFFTDX_STD::is_same_v<Precision, float>) {
                return complex_type {
                    fma_sincos_repeat<float, 8, 6, 4, 2, 0>(a), // cosine
                    a * fma_sincos_repeat<float, 7, 5, 3, 1>(a) // sine
                };
            } else if constexpr (CUFFTDX_STD::is_same_v<Precision, double>) {
                return complex_type {
                    fma_sincos_repeat<double, 14, 12, 10, 8, 6, 4, 2, 0>(a), // cosine
                    a * fma_sincos_repeat<double, 13, 11, 9, 7, 5, 3, 1>(a)  // sine
                };
            }

            // For compilers not fully compliant with the standard.
            return {};
        }

        // Reduce the range to [-PI/4, PI/4]
        template<typename Precision>
        __forceinline__ __device__ Precision
        limit_range(const Precision a, const Precision q) {
            if constexpr (CUFFTDX_STD::is_same_v<Precision, float>) {
                return fma_range_repeat<Precision, 0, 1>(a, q);
            } else if constexpr (CUFFTDX_STD::is_same_v<Precision, double>) {
                return fma_range_repeat<Precision, 0, 1, 2>(a, q);
            }
            // For compilers not fully compliant with the standard.
            return Precision {};
        }


        // Approximate ALU-based twiddle factor computation for single precision.
        template<typename Precision>
        __forceinline__ __device__
            CUFFTDX_STD::enable_if_t<CUFFTDX_STD::is_same_v<Precision, float> || CUFFTDX_STD::is_same_v<Precision, double>,
                                     cufftdx::detail::complex<Precision>>
            compute_twiddle(const unsigned k, const unsigned N) {
            using complex_type = detail::complex<Precision>;
            using numerics     = detail::numeric<Precision>;

            // Transform twiddle index to angle
            const Precision a = k * (-numerics::tau / N);
            const unsigned  q = numerics::to_int(a * (static_cast<Precision>(2.0) / numerics::pi));

            // Approximate sin and cos values
            complex_type twiddle = detail::approx_twiddle(detail::limit_range(a, static_cast<Precision>(-q)));

            // Handle ranges other than [-PI/4, PI/4]
            if (q & 1) {
                Precision T = twiddle.x;
                twiddle.x   = twiddle.y;
                twiddle.y   = T;
            }
            if (q & 2) {
                twiddle.y = -twiddle.y;
            }

            if ((q & 1) && !(q & 2)) {
                twiddle.x = -twiddle.x;
            }

            // Return the result
            return twiddle;
        }

        // RRII half2 overload
        template<typename Precision>
        __forceinline__ __device__
            CUFFTDX_STD::enable_if_t<CUFFTDX_STD::is_same_v<Precision, __half2>,
                                     cufftdx::detail::complex<__half2>>
            compute_twiddle(const unsigned k, const unsigned N) {
            using complex_type = detail::complex<float>;
            using numerics     = detail::numeric<float>;

            // Transform twiddle index to angle
            const float a = k * (-numerics::tau / N);
            // Use SFU instead of precomputed approximation
            complex_type fp32_twiddle;
            sincosf(a, &fp32_twiddle.y, &fp32_twiddle.x);
            // Convert and duplicate
            return {__float2half2_rn(fp32_twiddle.x), __float2half2_rn(fp32_twiddle.y)};
        }
    } // namespace detail
} // namespace cufftdx

#endif
