#include <cufftdx.hpp>

namespace cufftdx {
    namespace database {
        namespace detail {

            using lut_fp32_type = commondx::complex<float>;
            using lut_fp64_type = commondx::complex<double>;

#ifndef CUFFTDX_DETAIL_LUT_LINKAGE
#define CUFFTDX_DETAIL_LUT_LINKAGE extern
#endif // CUFFTDX_DETAIL_LUT_LINKAGE
            #include "database/lut_fp32.hpp.inc"
            #include "database/lut_fp64.hpp.inc"
#ifdef CUFFTDX_DETAIL_LUT_LINKAGE
#undef CUFFTDX_DETAIL_LUT_LINKAGE
#endif // CUFFTDX_DETAIL_LUT_LINKAGE

        }
    }
}
