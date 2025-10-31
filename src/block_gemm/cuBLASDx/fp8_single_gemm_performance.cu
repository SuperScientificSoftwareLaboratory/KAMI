#include "single_gemm_performance.hpp"

// This is an example of testing performance of cuBLASDx device function executing a general matrix multiply (GEMM)
//
//              C = alpha * A * B + beta * C
//
// A, B, and C are matrices. Mixed precisions are supported.
// Note that alpha and beta are expected to have the same precision and type (real or complex value)
// as the matrix C elements.
//
// Input data is generated on host using random number generators, and later copied to
// the global memory. Next, kernel with GEMM is executed, and then the matrix C (the result)
// is copied back to host memory, and verified against cuBLAS.
//
// The measured operation runs multiple times and the average speed is reported.

#ifndef INPUT_M
#define INPUT_M 64
#endif
#ifndef BLOCKSIZE
#define BLOCKSIZE 0
#endif
#ifndef USE_SUGGESTED_LD
#define USE_SUGGESTED_LD true
#endif

template<unsigned int Arch>
int single_gemm_performance() {
    using namespace cublasdx;

    // Parameters m, n, k define the dimensions of matrices A, B, and C.
    constexpr unsigned int m = INPUT_M;
    constexpr unsigned int n = INPUT_M;
    constexpr unsigned int k = INPUT_M;

    // Choose block size, or set to 0 to use library-suggested value.
    constexpr unsigned int BlockSize = BLOCKSIZE;

    // Flag to use library-suggested leading dimension (potential performance improvement).
    constexpr bool UseSuggestedLD = USE_SUGGESTED_LD;

    using PA = __nv_fp8_e5m2;
    using PB = __nv_fp8_e5m2;
    using PC = float;
    constexpr auto type = cublasdx::type::real;

    // Choose arrangement for A, B, C: row-major or column-major
    constexpr auto a_arrangement = cublasdx::row_major;
    constexpr auto b_arrangement = cublasdx::col_major;
    constexpr auto c_arrangement = cublasdx::row_major;

    // Define the matrix multiplication operation.
    using GEMM = decltype(cublasdx::Size<m, n, k>() +
                          cublasdx::Precision<PA, PB, PC>() +
                          cublasdx::Type<type>() +
                          cublasdx::Function<cublasdx::function::MM>() +
                          cublasdx::Arrangement<a_arrangement, b_arrangement, c_arrangement>() +
                          cublasdx::MaxAlignment() +
                          cublasdx::Block() +
                          cublasdx::SM<Arch>());

    bool verbose = true;
    cudaStream_t stream;
    CUDA_CHECK_AND_EXIT(cudaStreamCreate(&stream))
    int status = benchmark_mixed_precision_gemm<GEMM, Arch, BlockSize, UseSuggestedLD>(stream, verbose);
    CUDA_CHECK_AND_EXIT(cudaStreamDestroy(stream));

    return status;
}

template<unsigned int Arch>
struct single_gemm_performance_functor {
    int operator()() { return single_gemm_performance<Arch>(); }
};

int main(int, char**) {
    return example::sm_runner<single_gemm_performance_functor>();
}
