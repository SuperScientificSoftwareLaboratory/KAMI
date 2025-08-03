//Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//NVIDIA CORPORATION and its licensors retain all intellectual property
//and proprietary rights in and to this software, related documentation
//and any modifications thereto.  Any use, reproduction, disclosure or
//distribution of this software and related documentation without an express
//license agreement from NVIDIA CORPORATION is strictly prohibited.
//


#ifndef CUFFTDX_FFT_5_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_5_FP16_INV_PTX_HPP



#ifdef __CUDA_ARCH__
template<> __forceinline__ __device__ void cufftdx_private_function<1246, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({.reg .f32 f<13>;.reg .b32 r<323>;.reg .b64 rd<2>;mov.f32 f10,0f3E9E377A;{.reg .f16 low, high;cvt.rn.f16.f32 low,f10;cvt.rn.f16.f32 high,f10;mov.b32 r1,{low, high};}mov.f32 f12,0fBF737871;{.reg .f16 low, high;cvt.rn.f16.f32 low,f12;cvt.rn.f16.f32 high,f12;mov.b32 r2,{low, high};}neg.f16x2 r3,r2;mov.f32 f6,0fBF4F1BBD;{.reg .f16 low, high;cvt.rn.f16.f32 low,f6;cvt.rn.f16.f32 high,f6;mov.b32 r5,{low, high};}mov.f32 f8,0fBF167918;{.reg .f16 low, high;cvt.rn.f16.f32 low,f8;cvt.rn.f16.f32 high,f8;mov.b32 r6,{low, high};}neg.f16x2 r7,r6;{.reg .f16 low, high;cvt.rn.f16.f32 low,f10;cvt.rn.f16.f32 high,f10;mov.b32 r9,{low, high};}{.reg .f16 low, high;cvt.rn.f16.f32 low,f12;cvt.rn.f16.f32 high,f12;mov.b32 r10,{low, high};}add.f16x2 r11,%11,%12;add.f16x2 r14,%13,r11;add.f16x2 r17,%14,%15;add.f16x2 %0,r14,r17;add.f16x2 r23,%16,%10;add.f16x2 r26,%17,r23;add.f16x2 r29,%18,%19;add.f16x2 %1,r26,r29;add.f16x2 r35,%11,%12;mul.f16x2 r38,r35,r1;add.f16x2 r41,%13,r38;add.f16x2 r44,%14,%15;mul.f16x2 r47,r44,r5;add.f16x2 r50,r41,r47;sub.f16x2 r53,%16,%10;mul.f16x2 r56,r53,r3;sub.f16x2 r59,%18,%19;mul.f16x2 r62,r59,r7;add.f16x2 r65,r56,r62;sub.f16x2 %2,r50,r65;add.f16x2 r71,%11,%12;mul.f16x2 r74,r71,r1;add.f16x2 r77,%13,r74;add.f16x2 r80,%14,%15;mul.f16x2 r83,r80,r5;add.f16x2 r86,r77,r83;sub.f16x2 r89,%16,%10;mul.f16x2 r92,r89,r3;sub.f16x2 r95,%18,%19;mul.f16x2 r98,r95,r7;add.f16x2 r101,r92,r98;add.f16x2 %8,r86,r101;add.f16x2 r107,%11,%12;mul.f16x2 r110,r107,r5;add.f16x2 r113,%13,r110;add.f16x2 r116,%14,%15;mul.f16x2 r119,r116,r9;add.f16x2 r122,r113,r119;sub.f16x2 r125,%16,%10;mul.f16x2 r128,r125,r7;sub.f16x2 r131,%18,%19;mul.f16x2 r134,r131,r10;add.f16x2 r137,r128,r134;sub.f16x2 %4,r122,r137;add.f16x2 r143,%11,%12;mul.f16x2 r146,r143,r5;add.f16x2 r149,%13,r146;add.f16x2 r152,%14,%15;mul.f16x2 r155,r152,r9;add.f16x2 r158,r149,r155;sub.f16x2 r161,%16,%10;mul.f16x2 r164,r161,r7;sub.f16x2 r167,%18,%19;mul.f16x2 r170,r167,r10;add.f16x2 r173,r164,r170;add.f16x2 %6,r158,r173;add.f16x2 r179,%16,%10;mul.f16x2 r182,r179,r1;add.f16x2 r185,%17,r182;add.f16x2 r188,%18,%19;mul.f16x2 r191,r188,r5;add.f16x2 r194,r185,r191;sub.f16x2 r197,%11,%12;mul.f16x2 r200,r197,r3;sub.f16x2 r203,%14,%15;mul.f16x2 r206,r203,r7;add.f16x2 r209,r200,r206;add.f16x2 %3,r194,r209;add.f16x2 r215,%16,%10;mul.f16x2 r218,r215,r1;add.f16x2 r221,%17,r218;add.f16x2 r224,%18,%19;mul.f16x2 r227,r224,r5;add.f16x2 r230,r221,r227;sub.f16x2 r233,%11,%12;mul.f16x2 r236,r233,r3;sub.f16x2 r239,%14,%15;mul.f16x2 r242,r239,r7;add.f16x2 r245,r236,r242;sub.f16x2 %9,r230,r245;add.f16x2 r251,%16,%10;mul.f16x2 r254,r251,r5;add.f16x2 r257,%17,r254;add.f16x2 r260,%18,%19;mul.f16x2 r263,r260,r9;add.f16x2 r266,r257,r263;sub.f16x2 r269,%11,%12;mul.f16x2 r272,r269,r7;sub.f16x2 r275,%14,%15;mul.f16x2 r278,r275,r10;add.f16x2 r281,r272,r278;add.f16x2 %5,r266,r281;add.f16x2 r287,%16,%10;mul.f16x2 r290,r287,r5;add.f16x2 r293,%17,r290;add.f16x2 r296,%18,%19;mul.f16x2 r299,r296,r9;add.f16x2 r302,r293,r299;sub.f16x2 r305,%11,%12;mul.f16x2 r308,r305,r7;sub.f16x2 r311,%14,%15;mul.f16x2 r314,r311,r10;add.f16x2 r317,r308,r314;sub.f16x2 %7,r302,r317;})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)): "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].y)));
};
#endif // __CUDA_ARCH__


#endif

