//Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//NVIDIA CORPORATION and its licensors retain all intellectual property
//and proprietary rights in and to this software, related documentation
//and any modifications thereto.  Any use, reproduction, disclosure or
//distribution of this software and related documentation without an express
//license agreement from NVIDIA CORPORATION is strictly prohibited.
//


#ifndef CUFFTDX_FFT_3_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_3_FP16_FWD_PTX_HPP



#ifdef __CUDA_ARCH__
template<> __forceinline__ __device__ void cufftdx_private_function<975, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({.reg .f32 f<5>;.reg .b32 r<89>;.reg .b64 rd<2>;mov.f32 f2,0fBF000000;{.reg .f16 low, high;cvt.rn.f16.f32 low,f2;cvt.rn.f16.f32 high,f2;mov.b32 r1,{low, high};}mov.f32 f4,0fBF5DB3D7;{.reg .f16 low, high;cvt.rn.f16.f32 low,f4;cvt.rn.f16.f32 high,f4;mov.b32 r2,{low, high};}neg.f16x2 r3,r2;add.f16x2 r5,%6,%7;add.f16x2 %0,%8,r5;add.f16x2 r11,%9,%10;add.f16x2 %1,%11,r11;add.f16x2 r17,%6,%7;mul.f16x2 r20,r17,r1;add.f16x2 r23,%8,r20;sub.f16x2 r26,%9,%10;mul.f16x2 r29,r26,r3;add.f16x2 %2,r23,r29;add.f16x2 r35,%6,%7;mul.f16x2 r38,r35,r1;add.f16x2 r41,%8,r38;sub.f16x2 r44,%9,%10;mul.f16x2 r47,r44,r3;sub.f16x2 %4,r41,r47;add.f16x2 r53,%9,%10;mul.f16x2 r56,r53,r1;add.f16x2 r59,%11,r56;sub.f16x2 r62,%6,%7;mul.f16x2 r65,r62,r3;sub.f16x2 %3,r59,r65;add.f16x2 r71,%9,%10;mul.f16x2 r74,r71,r1;add.f16x2 r77,%11,r74;sub.f16x2 r80,%6,%7;mul.f16x2 r83,r80,r3;add.f16x2 %5,r77,r83;})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)): "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[0].y)));
};
#endif // __CUDA_ARCH__


#endif

