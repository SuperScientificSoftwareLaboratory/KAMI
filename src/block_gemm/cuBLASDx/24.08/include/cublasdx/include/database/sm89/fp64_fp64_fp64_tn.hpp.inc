
template<>
struct generated_config<double, double, double, 16, 16, 16, SM< 890 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<32, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 1, 1>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 64, 16>
> {};

template<>
struct generated_config<double, double, double, 32, 32, 32, SM< 890 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::universal_fma, 32, 8>,
    generated_config_impl<128, mma_atom::universal_fma, 4, 32>,
    generated_config_impl<64, mma_atom::universal_fma, 4, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<32, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 1, 1>
> {};

template<>
struct generated_config<double, double, double, 48, 48, 48, SM< 890 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 4, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<32, mma_atom::universal_fma, 4, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 64, 16>
> {};

template<>
struct generated_config<double, double, double, 64, 64, 64, SM< 890 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 32, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 4, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 64, 8>,
    generated_config_impl<1024, mma_atom::universal_fma, 64, 16>,
    generated_config_impl<32, mma_atom::SM80_8x8x4_F64F64F64F64_TN, 1, 1>
> {};
