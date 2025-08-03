
template<>
struct generated_config<double, double, double, 16, 16, 16, SM< 700 >> : generated_config_list<
    
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 4, 8>,
    generated_config_impl<256, mma_atom::universal_fma, 64, 4>,
    generated_config_impl<512, mma_atom::universal_fma, 128, 4>,
    generated_config_impl<1024, mma_atom::universal_fma, 128, 8>
> {};

template<>
struct generated_config<double, double, double, 32, 32, 32, SM< 700 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::universal_fma, 32, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 32, 2>,
    generated_config_impl<256, mma_atom::universal_fma, 32, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 4, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 128, 8>
> {};

template<>
struct generated_config<double, double, double, 48, 48, 48, SM< 700 >> : generated_config_list<
    
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>,
    generated_config_impl<128, mma_atom::universal_fma, 32, 4>,
    generated_config_impl<256, mma_atom::universal_fma, 64, 4>,
    generated_config_impl<32, mma_atom::universal_fma, 4, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 128, 4>,
    generated_config_impl<1024, mma_atom::universal_fma, 128, 8>
> {};

template<>
struct generated_config<double, double, double, 64, 64, 64, SM< 700 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::universal_fma, 64, 2>,
    generated_config_impl<256, mma_atom::universal_fma, 64, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 4, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 64, 8>,
    generated_config_impl<1024, mma_atom::universal_fma, 64, 16>,
    generated_config_impl<32, mma_atom::universal_fma, 4, 8>
> {};
