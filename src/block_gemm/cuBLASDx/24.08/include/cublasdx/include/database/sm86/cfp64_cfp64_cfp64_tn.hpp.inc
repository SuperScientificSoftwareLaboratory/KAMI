
template<>
struct generated_config<cutlass::complex<double>, cutlass::complex<double>, cutlass::complex<double>, 16, 16, 16, SM< 860 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 4, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>
> {};

template<>
struct generated_config<cutlass::complex<double>, cutlass::complex<double>, cutlass::complex<double>, 32, 32, 32, SM< 860 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<128, mma_atom::universal_fma, 32, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<256, mma_atom::universal_fma, 8, 32>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<32, mma_atom::universal_fma, 8, 4>
> {};

template<>
struct generated_config<cutlass::complex<double>, cutlass::complex<double>, cutlass::complex<double>, 45, 45, 45, SM< 860 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>,
    generated_config_impl<32, mma_atom::universal_fma, 16, 2>
> {};
