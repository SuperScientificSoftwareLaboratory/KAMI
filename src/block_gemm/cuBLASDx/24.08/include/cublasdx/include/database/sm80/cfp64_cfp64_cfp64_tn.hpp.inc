
template<>
struct generated_config<cutlass::complex<double>, cutlass::complex<double>, cutlass::complex<double>, 16, 16, 16, SM< 800 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 2, 4>,
    generated_config_impl<64, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 2, 1>,
    generated_config_impl<512, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 4, 4>,
    generated_config_impl<32, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 1, 1>,
    generated_config_impl<1024, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 16, 2>
> {};

template<>
struct generated_config<cutlass::complex<double>, cutlass::complex<double>, cutlass::complex<double>, 32, 32, 32, SM< 800 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 2, 4>,
    generated_config_impl<512, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 4, 4>,
    generated_config_impl<128, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 2, 2>,
    generated_config_impl<1024, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 8, 4>,
    generated_config_impl<64, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 2, 1>,
    generated_config_impl<32, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 1, 1>
> {};

template<>
struct generated_config<cutlass::complex<double>, cutlass::complex<double>, cutlass::complex<double>, 48, 48, 48, SM< 800 >> : generated_config_list<
    
    generated_config_impl<128, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 2, 2>,
    generated_config_impl<256, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 2, 4>,
    generated_config_impl<512, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 8, 2>,
    generated_config_impl<1024, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 8, 4>,
    generated_config_impl<64, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 2, 1>,
    generated_config_impl<32, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 1, 1>
> {};

template<>
struct generated_config<cutlass::complex<double>, cutlass::complex<double>, cutlass::complex<double>, 58, 58, 58, SM< 800 >> : generated_config_list<
    
    generated_config_impl<256, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 4, 2>,
    generated_config_impl<512, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 4, 4>,
    generated_config_impl<1024, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 4, 8>,
    generated_config_impl<128, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 2, 2>,
    generated_config_impl<32, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 1, 1>,
    generated_config_impl<64, mma_atom::SM80_8x8x4_C64C64C64C64_TN, 2, 1>
> {};
