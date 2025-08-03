//Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//NVIDIA CORPORATION and its licensors retain all intellectual property
//and proprietary rights in and to this software, related documentation
//and any modifications thereto.  Any use, reproduction, disclosure or
//distribution of this software and related documentation without an express
//license agreement from NVIDIA CORPORATION is strictly prohibited.
//


#ifndef CUFFTDX_FFT_6_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_6_FP16_FWD_PTX_HPP



#ifdef __CUDA_ARCH__
template<> __forceinline__ __device__ void cufftdx_private_function<1042, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({.reg .f32 f<29>;.reg .b32 r<255>;.reg .b64 rd<2>;mov.f32 f14,0fBF000000;{.reg .f16 low, high;cvt.rn.f16.f32 low,f14;cvt.rn.f16.f32 high,f14;mov.b32 r1,{low, high};}mov.f32 f16,0fBF5DB3D7;{.reg .f16 low, high;cvt.rn.f16.f32 low,f16;cvt.rn.f16.f32 high,f16;mov.b32 r2,{low, high};}neg.f16x2 r3,r2;add.f16x2 r5,%17,%13;add.f16x2 r8,%18,r5;add.f16x2 r11,%19,%14;add.f16x2 r14,%20,r11;add.f16x2 r17,%17,%13;mul.f16x2 r20,r17,r1;add.f16x2 r23,%18,r20;sub.f16x2 r26,%19,%14;mul.f16x2 r29,r26,r3;add.f16x2 r32,r23,r29;add.f16x2 r35,%17,%13;mul.f16x2 r38,r35,r1;add.f16x2 r41,%18,r38;sub.f16x2 r44,%19,%14;mul.f16x2 r47,r44,r3;sub.f16x2 r50,r41,r47;add.f16x2 r53,%19,%14;mul.f16x2 r56,r53,r1;add.f16x2 r59,%20,r56;sub.f16x2 r62,%17,%13;mul.f16x2 r65,r62,r3;sub.f16x2 r68,r59,r65;add.f16x2 r71,%19,%14;mul.f16x2 r74,r71,r1;add.f16x2 r77,%20,r74;sub.f16x2 r80,%17,%13;mul.f16x2 r83,r80,r3;add.f16x2 r86,r77,r83;{.reg .f16 low, high;cvt.rn.f16.f32 low,f14;cvt.rn.f16.f32 high,f14;mov.b32 r89,{low, high};}{.reg .f16 low, high;cvt.rn.f16.f32 low,f16;cvt.rn.f16.f32 high,f16;mov.b32 r90,{low, high};}neg.f16x2 r91,r90;add.f16x2 r93,%21,%15;add.f16x2 r96,%22,r93;add.f16x2 r99,%12,%16;add.f16x2 r102,%23,r99;add.f16x2 r105,%21,%15;mul.f16x2 r108,r105,r89;add.f16x2 r111,%22,r108;sub.f16x2 r114,%12,%16;mul.f16x2 r117,r114,r91;add.f16x2 r120,r111,r117;add.f16x2 r123,%21,%15;mul.f16x2 r126,r123,r89;add.f16x2 r129,%22,r126;sub.f16x2 r132,%12,%16;mul.f16x2 r135,r132,r91;sub.f16x2 r138,r129,r135;add.f16x2 r141,%12,%16;mul.f16x2 r144,r141,r89;add.f16x2 r147,%23,r144;sub.f16x2 r150,%21,%15;mul.f16x2 r153,r150,r91;sub.f16x2 r156,r147,r153;add.f16x2 r159,%12,%16;mul.f16x2 r162,r159,r89;add.f16x2 r165,%23,r162;sub.f16x2 r168,%21,%15;mul.f16x2 r171,r168,r91;add.f16x2 r174,r165,r171;mov.f32 f10,0f3F000000;{.reg .f16 low, high;cvt.rn.f16.f32 low,f10;cvt.rn.f16.f32 high,f10;mov.b32 r177,{low, high};}{.reg .f16 low, high;cvt.rn.f16.f32 low,f16;cvt.rn.f16.f32 high,f16;mov.b32 r178,{low, high};}{.reg .f16 low, high;cvt.rn.f16.f32 low,f14;cvt.rn.f16.f32 high,f14;mov.b32 r179,{low, high};}{.reg .f16 low, high;cvt.rn.f16.f32 low,f16;cvt.rn.f16.f32 high,f16;mov.b32 r180,{low, high};}mul.f16x2 r187,r120,r177;mul.f16x2 r190,r156,r178;sub.f16x2 r193,r187,r190;mul.f16x2 r196,r120,r178;fma.rn.f16x2 r199,r156,r177,r196;mul.f16x2 r203,r138,r179;mul.f16x2 r206,r174,r180;sub.f16x2 r209,r203,r206;mul.f16x2 r212,r138,r180;fma.rn.f16x2 r215,r174,r179,r212;add.f16x2 %0,r8,r96;add.f16x2 %1,r14,r102;sub.f16x2 %6,r8,r96;sub.f16x2 %7,r14,r102;add.f16x2 %2,r32,r193;add.f16x2 %3,r68,r199;sub.f16x2 %8,r32,r193;sub.f16x2 %9,r68,r199;add.f16x2 %4,r50,r209;add.f16x2 %5,r86,r215;sub.f16x2 %10,r50,r209;sub.f16x2 %11,r86,r215;})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)): "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)));
};
#endif // __CUDA_ARCH__


#endif

