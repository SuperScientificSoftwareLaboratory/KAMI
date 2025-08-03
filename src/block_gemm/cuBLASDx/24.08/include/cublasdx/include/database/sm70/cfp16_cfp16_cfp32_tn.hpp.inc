
template<>
struct generated_config<cutlass::complex<half_t>, cutlass::complex<half_t>, cutlass::complex<float>, 16, 16, 16, SM< 700 >> : generated_config_list<
    
    generated_config_impl<32, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<64, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 2, 4>,
    generated_config_impl<128, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 2, 8>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<512, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 32, 2>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>
> {};

template<>
struct generated_config<cutlass::complex<half_t>, cutlass::complex<half_t>, cutlass::complex<float>, 32, 32, 32, SM< 700 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 4, 4>,
    generated_config_impl<64, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<256, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 4, 8>,
    generated_config_impl<32, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>
> {};

template<>
struct generated_config<cutlass::complex<half_t>, cutlass::complex<half_t>, cutlass::complex<float>, 48, 48, 48, SM< 700 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 8, 2>,
    generated_config_impl<256, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 2, 16>,
    generated_config_impl<64, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 32, 2>,
    generated_config_impl<32, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<1024, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 32, 4>
> {};

template<>
struct generated_config<cutlass::complex<half_t>, cutlass::complex<half_t>, cutlass::complex<float>, 64, 64, 64, SM< 700 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 8, 2>,
    generated_config_impl<256, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 8, 4>,
    generated_config_impl<512, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 8, 8>,
    generated_config_impl<64, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<1024, mma_atom::universal_fma, 64, 16>,
    generated_config_impl<32, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 2, 2>
> {};

template<>
struct generated_config<cutlass::complex<half_t>, cutlass::complex<half_t>, cutlass::complex<float>, 78, 78, 78, SM< 700 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 16, 2>,
    generated_config_impl<512, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 16, 4>,
    generated_config_impl<128, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 16, 8>,
    generated_config_impl<32, mma_atom::SM70_8x8x4_C32C16C16C32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>
> {};
