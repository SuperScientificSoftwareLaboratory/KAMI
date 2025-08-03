
template<>
struct generated_config<half_t, half_t, half_t, 16, 16, 16, SM< 800 >> : generated_config_list<
    
    generated_config_impl<64, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 1, 2>,
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 1, 4>,
    generated_config_impl<32, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 1, 1>,
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 1, 8>,
    generated_config_impl<512, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 8, 2>,
    generated_config_impl<1024, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 16, 2>
> {};

template<>
struct generated_config<half_t, half_t, half_t, 32, 32, 32, SM< 800 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<64, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 1, 2>,
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 2, 4>,
    generated_config_impl<32, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 1, 1>,
    generated_config_impl<512, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 2, 8>,
    generated_config_impl<1024, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 8, 4>
> {};

template<>
struct generated_config<half_t, half_t, half_t, 48, 48, 48, SM< 800 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<64, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 2, 1>,
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 4, 2>,
    generated_config_impl<32, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 1, 1>,
    generated_config_impl<512, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 8, 2>,
    generated_config_impl<1024, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 16, 2>
> {};

template<>
struct generated_config<half_t, half_t, half_t, 64, 64, 64, SM< 800 >> : generated_config_list<
    
    generated_config_impl<64, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 1, 2>,
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 2, 2>,
    generated_config_impl<256, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 4, 4>,
    generated_config_impl<32, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 1, 1>,
    generated_config_impl<1024, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 4, 8>
> {};

template<>
struct generated_config<half_t, half_t, half_t, 80, 80, 80, SM< 800 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 1, 4>,
    generated_config_impl<64, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 2, 1>,
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 8, 2>,
    generated_config_impl<1024, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 8, 4>,
    generated_config_impl<32, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 1, 1>
> {};

template<>
struct generated_config<half_t, half_t, half_t, 96, 96, 96, SM< 800 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<64, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 2, 1>,
    generated_config_impl<256, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 2, 4>,
    generated_config_impl<512, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<32, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 1, 1>,
    generated_config_impl<1024, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 8, 4>
> {};

template<>
struct generated_config<half_t, half_t, half_t, 112, 112, 112, SM< 800 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 4, 2>,
    generated_config_impl<64, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 2, 1>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 4, 4>,
    generated_config_impl<1024, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 8, 4>,
    generated_config_impl<32, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<half_t, half_t, half_t, 128, 128, 128, SM< 800 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<64, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 1, 2>,
    generated_config_impl<256, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 2, 4>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 4, 4>,
    generated_config_impl<1024, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 8, 4>,
    generated_config_impl<32, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 1, 1>
> {};

template<>
struct generated_config<half_t, half_t, half_t, 144, 144, 144, SM< 800 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 4, 2>,
    generated_config_impl<64, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 2, 1>,
    generated_config_impl<512, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 8, 4>,
    generated_config_impl<32, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<half_t, half_t, half_t, 160, 160, 160, SM< 800 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 2, 2>,
    generated_config_impl<256, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 2, 4>,
    generated_config_impl<512, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 4, 4>,
    generated_config_impl<1024, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 8, 4>,
    generated_config_impl<64, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 1, 2>,
    generated_config_impl<32, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<half_t, half_t, half_t, 166, 166, 166, SM< 800 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 2, 4>,
    generated_config_impl<512, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM75_16x8x8_F16F16F16F16_TN_CUBLASDX, 8, 4>,
    generated_config_impl<32, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 1, 1>,
    generated_config_impl<64, mma_atom::SM80_16x8x16_F16F16F16F16_TN, 2, 1>
> {};
