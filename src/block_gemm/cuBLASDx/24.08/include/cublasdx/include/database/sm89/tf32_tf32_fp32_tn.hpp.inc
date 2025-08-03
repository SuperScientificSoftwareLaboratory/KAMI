
template<>
struct generated_config<tfloat32_t, tfloat32_t, float, 16, 16, 16, SM< 890 >> : generated_config_list<
    
    generated_config_impl<64, mma_atom::SM80_16x8x8_F32TF32TF32F32_TN, 1, 2>,
    generated_config_impl<128, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 2, 2>,
    generated_config_impl<32, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 1, 1>,
    generated_config_impl<256, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_16x8x8_F32TF32TF32F32_TN, 8, 2>,
    generated_config_impl<1024, mma_atom::SM80_16x8x8_F32TF32TF32F32_TN, 16, 2>
> {};

template<>
struct generated_config<tfloat32_t, tfloat32_t, float, 32, 32, 32, SM< 890 >> : generated_config_list<
    
    generated_config_impl<64, mma_atom::SM80_16x8x8_F32TF32TF32F32_TN, 2, 1>,
    generated_config_impl<128, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 2, 2>,
    generated_config_impl<32, mma_atom::SM80_16x8x8_F32TF32TF32F32_TN, 1, 1>,
    generated_config_impl<256, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 2, 4>,
    generated_config_impl<512, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 4, 4>,
    generated_config_impl<1024, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 8, 4>
> {};

template<>
struct generated_config<tfloat32_t, tfloat32_t, float, 48, 48, 48, SM< 890 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 2, 2>,
    generated_config_impl<64, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 1, 2>,
    generated_config_impl<256, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_16x8x8_F32TF32TF32F32_TN, 4, 4>,
    generated_config_impl<32, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 1, 1>,
    generated_config_impl<1024, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 4, 8>
> {};

template<>
struct generated_config<tfloat32_t, tfloat32_t, float, 64, 64, 64, SM< 890 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 2, 2>,
    generated_config_impl<64, mma_atom::SM80_16x8x8_F32TF32TF32F32_TN, 2, 1>,
    generated_config_impl<256, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 4, 4>,
    generated_config_impl<32, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 1, 1>,
    generated_config_impl<1024, mma_atom::SM80_16x8x8_F32TF32TF32F32_TN, 4, 8>
> {};

template<>
struct generated_config<tfloat32_t, tfloat32_t, float, 80, 80, 80, SM< 890 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM80_16x8x8_F32TF32TF32F32_TN, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 2, 4>,
    generated_config_impl<512, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 4, 4>,
    generated_config_impl<64, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 1, 2>,
    generated_config_impl<1024, mma_atom::SM80_16x8x8_F32TF32TF32F32_TN, 8, 4>,
    generated_config_impl<32, mma_atom::SM80_16x8x8_F32TF32TF32F32_TN, 1, 1>
> {};

template<>
struct generated_config<tfloat32_t, tfloat32_t, float, 91, 91, 91, SM< 890 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 2, 4>,
    generated_config_impl<128, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 1, 4>,
    generated_config_impl<512, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 2, 8>,
    generated_config_impl<1024, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 8, 4>,
    generated_config_impl<64, mma_atom::SM80_16x8x4_F32TF32TF32F32_TN, 2, 1>,
    generated_config_impl<32, mma_atom::SM80_16x8x8_F32TF32TF32F32_TN, 1, 1>
> {};
