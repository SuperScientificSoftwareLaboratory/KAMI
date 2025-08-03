
template<>
struct generated_config<float_e5m2_t, float_e5m2_t, float, 16, 16, 16, SM< 750 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<1024, mma_atom::universal_fma, 16, 64>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 8, 4>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, float, 32, 32, 32, SM< 750 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>,
    generated_config_impl<32, mma_atom::universal_fma, 8, 4>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, float, 48, 48, 48, SM< 750 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 64, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>,
    generated_config_impl<32, mma_atom::universal_fma, 16, 2>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, float, 64, 64, 64, SM< 750 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 32, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>,
    generated_config_impl<32, mma_atom::universal_fma, 16, 2>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, float, 80, 80, 80, SM< 750 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 4, 16>,
    generated_config_impl<32, mma_atom::universal_fma, 4, 8>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, float, 96, 96, 96, SM< 750 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 8, 4>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, float, 104, 104, 104, SM< 750 >> : generated_config_list<
    
    generated_config_impl<1024, mma_atom::universal_fma, 8, 128>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 8, 4>
> {};
