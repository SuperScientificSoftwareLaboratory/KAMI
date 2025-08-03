
template<>
struct generated_config<cutlass::complex<float_e5m2_t>, cutlass::complex<float_e5m2_t>, cutlass::complex<float>, 16, 16, 16, SM< 900 >> : generated_config_list<
    
    generated_config_impl<64, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 1, 2>,
    generated_config_impl<128, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 1, 1>,
    generated_config_impl<256, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 1, 8>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 1, 16>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 1, 32>
> {};

template<>
struct generated_config<cutlass::complex<float_e5m2_t>, cutlass::complex<float_e5m2_t>, cutlass::complex<float>, 32, 32, 32, SM< 900 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 2, 4>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 2, 1>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 2, 8>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 1, 1>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 8, 4>
> {};

template<>
struct generated_config<cutlass::complex<float_e5m2_t>, cutlass::complex<float_e5m2_t>, cutlass::complex<float>, 48, 48, 48, SM< 900 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 4, 4>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 1, 2>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 4, 8>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<cutlass::complex<float_e5m2_t>, cutlass::complex<float_e5m2_t>, cutlass::complex<float>, 64, 64, 64, SM< 900 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<128, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 4, 4>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 4, 8>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 2, 1>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<cutlass::complex<float_e5m2_t>, cutlass::complex<float_e5m2_t>, cutlass::complex<float>, 80, 80, 80, SM< 900 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<256, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 8, 2>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 8, 4>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 2, 1>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<cutlass::complex<float_e5m2_t>, cutlass::complex<float_e5m2_t>, cutlass::complex<float>, 96, 96, 96, SM< 900 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<128, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 8, 2>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 8, 4>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 1, 2>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<cutlass::complex<float_e5m2_t>, cutlass::complex<float_e5m2_t>, cutlass::complex<float>, 112, 112, 112, SM< 900 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 8, 2>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 8, 4>,
    generated_config_impl<128, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 2, 1>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<cutlass::complex<float_e5m2_t>, cutlass::complex<float_e5m2_t>, cutlass::complex<float>, 128, 128, 128, SM< 900 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<512, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 8, 2>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 8, 4>,
    generated_config_impl<128, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 2, 1>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 1, 1>
> {};

template<>
struct generated_config<cutlass::complex<float_e5m2_t>, cutlass::complex<float_e5m2_t>, cutlass::complex<float>, 139, 139, 139, SM< 900 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 4, 2>,
    generated_config_impl<1024, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 16, 2>,
    generated_config_impl<512, mma_atom::universal_fma, 32, 16>,
    generated_config_impl<128, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 2, 2>,
    generated_config_impl<32, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 1, 1>,
    generated_config_impl<64, mma_atom::SM89_16x8x32_C32CE5M2CE5M2C32_TN_CUBLASDX, 1, 2>
> {};
