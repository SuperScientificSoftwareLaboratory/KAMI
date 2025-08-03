
template<>
struct generated_config<float_e5m2_t, float_e5m2_t, float, 16, 16, 16, SM< 900 >> : generated_config_list<
    
    generated_config_impl<64, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 2>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 1>,
    generated_config_impl<128, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 4>,
    generated_config_impl<256, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 16>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 32>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, float, 32, 32, 32, SM< 900 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 2, 4>,
    generated_config_impl<128, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 4>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 2>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 1>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 2, 8>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 8, 4>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, float, 48, 48, 48, SM< 900 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 2>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 1>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 8, 2>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 4, 8>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, float, 64, 64, 64, SM< 900 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 2, 4>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 2>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 4, 8>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, float, 80, 80, 80, SM< 900 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 2>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 8, 2>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 8, 4>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, float, 96, 96, 96, SM< 900 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 2, 4>,
    generated_config_impl<128, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 8, 2>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 2, 1>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 8, 4>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, float, 112, 112, 112, SM< 900 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<128, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 4, 8>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 2>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, float, 128, 128, 128, SM< 900 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 8, 4>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 2, 1>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, float, 144, 144, 144, SM< 900 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 4, 4>,
    generated_config_impl<128, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 16, 2>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 2>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, float, 160, 160, 160, SM< 900 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 2, 4>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 4, 4>,
    generated_config_impl<128, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 16, 2>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 2, 1>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, float, 176, 176, 176, SM< 900 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 4, 8>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 2>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 1>,
    generated_config_impl<128, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 4>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, float, 192, 192, 192, SM< 900 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 4, 8>,
    generated_config_impl<128, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 2, 1>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<float_e5m2_t, float_e5m2_t, float, 196, 196, 196, SM< 900 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 8, 2>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 16, 2>,
    generated_config_impl<128, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 1>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_F32E5M2E5M2F32_TN_CUBLASDX, 1, 2>
> {};
