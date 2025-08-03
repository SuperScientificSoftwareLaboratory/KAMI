
template<>
struct generated_config<cutlass::complex<tfloat32_t>, cutlass::complex<tfloat32_t>, cutlass::complex<float>, 16, 16, 16, SM< 890 >> : generated_config_list<
    
    generated_config_impl<64, mma_atom::SM80_16x8x4_C32TC32TC32C32_TN_CUBLASDX, 1, 2>,
    generated_config_impl<128, mma_atom::SM80_16x8x4_C32TC32TC32C32_TN_CUBLASDX, 1, 4>,
    generated_config_impl<256, mma_atom::SM80_16x8x8_C32TC32TC32C32_TN_CUBLASDX, 2, 4>,
    generated_config_impl<32, mma_atom::SM80_16x8x4_C32TC32TC32C32_TN_CUBLASDX, 1, 1>,
    generated_config_impl<512, mma_atom::SM80_16x8x8_C32TC32TC32C32_TN_CUBLASDX, 8, 2>,
    generated_config_impl<1024, mma_atom::SM80_16x8x8_C32TC32TC32C32_TN_CUBLASDX, 16, 2>
> {};

template<>
struct generated_config<cutlass::complex<tfloat32_t>, cutlass::complex<tfloat32_t>, cutlass::complex<float>, 32, 32, 32, SM< 890 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM80_16x8x4_C32TC32TC32C32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_16x8x8_C32TC32TC32C32_TN_CUBLASDX, 2, 4>,
    generated_config_impl<64, mma_atom::SM80_16x8x4_C32TC32TC32C32_TN_CUBLASDX, 2, 1>,
    generated_config_impl<512, mma_atom::SM80_16x8x4_C32TC32TC32C32_TN_CUBLASDX, 2, 8>,
    generated_config_impl<1024, mma_atom::SM80_16x8x8_C32TC32TC32C32_TN_CUBLASDX, 8, 4>,
    generated_config_impl<32, mma_atom::SM80_16x8x4_C32TC32TC32C32_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<cutlass::complex<tfloat32_t>, cutlass::complex<tfloat32_t>, cutlass::complex<float>, 48, 48, 48, SM< 890 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM80_16x8x8_C32TC32TC32C32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<128, mma_atom::SM80_16x8x4_C32TC32TC32C32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<512, mma_atom::SM80_16x8x8_C32TC32TC32C32_TN_CUBLASDX, 4, 4>,
    generated_config_impl<64, mma_atom::SM80_16x8x4_C32TC32TC32C32_TN_CUBLASDX, 1, 2>,
    generated_config_impl<1024, mma_atom::SM80_16x8x4_C32TC32TC32C32_TN_CUBLASDX, 4, 8>,
    generated_config_impl<32, mma_atom::SM80_16x8x4_C32TC32TC32C32_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<cutlass::complex<tfloat32_t>, cutlass::complex<tfloat32_t>, cutlass::complex<float>, 64, 64, 64, SM< 890 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM80_16x8x8_C32TC32TC32C32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_16x8x4_C32TC32TC32C32_TN_CUBLASDX, 4, 4>,
    generated_config_impl<128, mma_atom::SM80_16x8x4_C32TC32TC32C32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<1024, mma_atom::SM80_16x8x8_C32TC32TC32C32_TN_CUBLASDX, 4, 8>,
    generated_config_impl<64, mma_atom::SM80_16x8x4_C32TC32TC32C32_TN_CUBLASDX, 2, 1>,
    generated_config_impl<32, mma_atom::SM80_16x8x8_C32TC32TC32C32_TN_CUBLASDX, 1, 1>
> {};
