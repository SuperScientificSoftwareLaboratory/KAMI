
template<>
struct generated_config<float, float, float, 16, 16, 16, SM< 900 >> : generated_config_list<
    
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 4, 8>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 16, 32>,
    generated_config_impl<1024, mma_atom::universal_fma, 128, 8>
> {};

template<>
struct generated_config<float, float, float, 32, 32, 32, SM< 900 >> : generated_config_list<
    
    generated_config_impl<64, mma_atom::universal_fma, 4, 16>,
    generated_config_impl<32, mma_atom::universal_fma, 4, 8>,
    generated_config_impl<128, mma_atom::universal_fma, 32, 4>,
    generated_config_impl<256, mma_atom::universal_fma, 32, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>
> {};

template<>
struct generated_config<float, float, float, 48, 48, 48, SM< 900 >> : generated_config_list<
    
    generated_config_impl<64, mma_atom::universal_fma, 4, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<32, mma_atom::universal_fma, 4, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 128, 4>,
    generated_config_impl<1024, mma_atom::universal_fma, 128, 8>
> {};

template<>
struct generated_config<float, float, float, 64, 64, 64, SM< 900 >> : generated_config_list<
    
    generated_config_impl<64, mma_atom::universal_fma, 4, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 32, 4>,
    generated_config_impl<256, mma_atom::universal_fma, 64, 4>,
    generated_config_impl<512, mma_atom::universal_fma, 64, 8>,
    generated_config_impl<1024, mma_atom::universal_fma, 64, 16>,
    generated_config_impl<32, mma_atom::universal_fma, 4, 8>
> {};

template<>
struct generated_config<float, float, float, 80, 80, 80, SM< 900 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<256, mma_atom::universal_fma, 64, 4>,
    generated_config_impl<512, mma_atom::universal_fma, 64, 8>,
    generated_config_impl<1024, mma_atom::universal_fma, 64, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 8, 4>
> {};

template<>
struct generated_config<float, float, float, 96, 96, 96, SM< 900 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<256, mma_atom::universal_fma, 32, 8>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 4, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<32, mma_atom::universal_fma, 8, 4>
> {};

template<>
struct generated_config<float, float, float, 112, 112, 112, SM< 900 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<512, mma_atom::universal_fma, 64, 8>,
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 64, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 4, 8>
> {};

template<>
struct generated_config<float, float, float, 128, 128, 128, SM< 900 >> : generated_config_list<
    
    generated_config_impl<512, mma_atom::universal_fma, 64, 8>,
    generated_config_impl<256, mma_atom::universal_fma, 64, 4>,
    generated_config_impl<1024, mma_atom::universal_fma, 128, 8>,
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 8, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 16, 2>
> {};

template<>
struct generated_config<float, float, float, 139, 139, 139, SM< 900 >> : generated_config_list<
    
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<1024, mma_atom::universal_fma, 32, 32>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<128, mma_atom::universal_fma, 16, 8>,
    generated_config_impl<32, mma_atom::universal_fma, 8, 4>,
    generated_config_impl<64, mma_atom::universal_fma, 4, 16>
> {};
