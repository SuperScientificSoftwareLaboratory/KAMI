
template<>
struct generated_config<cutlass::complex<double>, cutlass::complex<double>, cutlass::complex<double>, 16, 16, 16, SM< 890 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<32, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 1, 1>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>
> {};

template<>
struct generated_config<cutlass::complex<double>, cutlass::complex<double>, cutlass::complex<double>, 32, 32, 32, SM< 890 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<256, mma_atom::universal_fma, 32, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<32, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 1, 1>
> {};

template<>
struct generated_config<cutlass::complex<double>, cutlass::complex<double>, cutlass::complex<double>, 45, 45, 45, SM< 890 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>,
    generated_config_impl<32, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 1, 1>
> {};
