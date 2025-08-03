
template<>
struct generated_config<half_t, half_t, float, 16, 16, 16, SM< 700 >> : generated_config_list<
    
    generated_config_impl<32, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 2, 2>,
    generated_config_impl<128, mma_atom::universal_fma, 8, 16>,
    generated_config_impl<64, mma_atom::universal_fma, 16, 4>,
    generated_config_impl<256, mma_atom::universal_fma, 16, 16>,
    generated_config_impl<512, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 32, 2>,
    generated_config_impl<1024, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 64, 2>
> {};

template<>
struct generated_config<half_t, half_t, float, 32, 32, 32, SM< 700 >> : generated_config_list<
    
    generated_config_impl<64, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 2, 4>,
    generated_config_impl<128, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 4, 4>,
    generated_config_impl<32, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 1, 4>,
    generated_config_impl<256, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 4, 8>,
    generated_config_impl<512, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 4, 16>,
    generated_config_impl<1024, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 32, 4>
> {};

template<>
struct generated_config<half_t, half_t, float, 48, 48, 48, SM< 700 >> : generated_config_list<
    
    generated_config_impl<64, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 4, 2>,
    generated_config_impl<32, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 2, 2>,
    generated_config_impl<128, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 2, 8>,
    generated_config_impl<256, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 2, 16>,
    generated_config_impl<512, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 32, 2>,
    generated_config_impl<1024, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 64, 2>
> {};

template<>
struct generated_config<half_t, half_t, float, 64, 64, 64, SM< 700 >> : generated_config_list<
    
    generated_config_impl<64, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 1, 8>,
    generated_config_impl<128, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 4, 4>,
    generated_config_impl<256, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 8, 4>,
    generated_config_impl<32, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 1, 4>,
    generated_config_impl<512, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 8, 8>,
    generated_config_impl<1024, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 8, 16>
> {};

template<>
struct generated_config<half_t, half_t, float, 80, 80, 80, SM< 700 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 2, 8>,
    generated_config_impl<256, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 8, 4>,
    generated_config_impl<64, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 2, 4>,
    generated_config_impl<512, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 8, 8>,
    generated_config_impl<32, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 2, 2>,
    generated_config_impl<1024, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 8, 16>
> {};

template<>
struct generated_config<half_t, half_t, float, 96, 96, 96, SM< 700 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 4, 4>,
    generated_config_impl<64, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 2, 4>,
    generated_config_impl<256, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 4, 8>,
    generated_config_impl<512, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 4, 16>,
    generated_config_impl<1024, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 4, 32>,
    generated_config_impl<32, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 2, 2>
> {};

template<>
struct generated_config<half_t, half_t, float, 110, 110, 110, SM< 700 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 4, 8>,
    generated_config_impl<512, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 8, 8>,
    generated_config_impl<128, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 2, 8>,
    generated_config_impl<1024, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 16, 8>,
    generated_config_impl<64, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 2, 4>,
    generated_config_impl<32, mma_atom::SM70_8x8x4_F32F16F16F32_TN, 2, 2>
> {};
