//Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//NVIDIA CORPORATION and its licensors retain all intellectual property
//and proprietary rights in and to this software, related documentation
//and any modifications thereto.  Any use, reproduction, disclosure or
//distribution of this software and related documentation without an express
//license agreement from NVIDIA CORPORATION is strictly prohibited.
//


#ifndef CUFFTDX_FFT_5_FP16_FWD_PTX_HPP
#define CUFFTDX_FFT_5_FP16_FWD_PTX_HPP



#ifdef __CUDA_ARCH__
template<> __forceinline__ __device__ void cufftdx_private_function<1012, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({.reg .f32 f<13>;.reg .b32 r<321>;.reg .b64 rd<2>;mov.f32 f10,0f3E9E377A;{.reg .f16 low, high;cvt.rn.f16.f32 low,f10;cvt.rn.f16.f32 high,f10;mov.b32 r1,{low, high};}mov.f32 f12,0fBF737871;{.reg .f16 low, high;cvt.rn.f16.f32 low,f12;cvt.rn.f16.f32 high,f12;mov.b32 r2,{low, high};}mov.f32 f6,0fBF4F1BBD;{.reg .f16 low, high;cvt.rn.f16.f32 low,f6;cvt.rn.f16.f32 high,f6;mov.b32 r3,{low, high};}mov.f32 f8,0fBF167918;{.reg .f16 low, high;cvt.rn.f16.f32 low,f8;cvt.rn.f16.f32 high,f8;mov.b32 r4,{low, high};}{.reg .f16 low, high;cvt.rn.f16.f32 low,f10;cvt.rn.f16.f32 high,f10;mov.b32 r5,{low, high};}{.reg .f16 low, high;cvt.rn.f16.f32 low,f12;cvt.rn.f16.f32 high,f12;mov.b32 r6,{low, high};}neg.f16x2 r7,r6;add.f16x2 r9,%11,%12;add.f16x2 r12,%13,r9;add.f16x2 r15,%14,%15;add.f16x2 %0,r12,r15;add.f16x2 r21,%16,%10;add.f16x2 r24,%17,r21;add.f16x2 r27,%18,%19;add.f16x2 %1,r24,r27;add.f16x2 r33,%11,%12;mul.f16x2 r36,r33,r1;add.f16x2 r39,%13,r36;add.f16x2 r42,%14,%15;mul.f16x2 r45,r42,r3;add.f16x2 r48,r39,r45;sub.f16x2 r51,%16,%10;mul.f16x2 r54,r51,r2;sub.f16x2 r57,%18,%19;mul.f16x2 r60,r57,r4;add.f16x2 r63,r54,r60;sub.f16x2 %2,r48,r63;add.f16x2 r69,%11,%12;mul.f16x2 r72,r69,r1;add.f16x2 r75,%13,r72;add.f16x2 r78,%14,%15;mul.f16x2 r81,r78,r3;add.f16x2 r84,r75,r81;sub.f16x2 r87,%16,%10;mul.f16x2 r90,r87,r2;sub.f16x2 r93,%18,%19;mul.f16x2 r96,r93,r4;add.f16x2 r99,r90,r96;add.f16x2 %8,r84,r99;add.f16x2 r105,%11,%12;mul.f16x2 r108,r105,r3;add.f16x2 r111,%13,r108;add.f16x2 r114,%14,%15;mul.f16x2 r117,r114,r5;add.f16x2 r120,r111,r117;sub.f16x2 r123,%16,%10;mul.f16x2 r126,r123,r4;sub.f16x2 r129,%18,%19;mul.f16x2 r132,r129,r7;add.f16x2 r135,r126,r132;sub.f16x2 %4,r120,r135;add.f16x2 r141,%11,%12;mul.f16x2 r144,r141,r3;add.f16x2 r147,%13,r144;add.f16x2 r150,%14,%15;mul.f16x2 r153,r150,r5;add.f16x2 r156,r147,r153;sub.f16x2 r159,%16,%10;mul.f16x2 r162,r159,r4;sub.f16x2 r165,%18,%19;mul.f16x2 r168,r165,r7;add.f16x2 r171,r162,r168;add.f16x2 %6,r156,r171;add.f16x2 r177,%16,%10;mul.f16x2 r180,r177,r1;add.f16x2 r183,%17,r180;add.f16x2 r186,%18,%19;mul.f16x2 r189,r186,r3;add.f16x2 r192,r183,r189;sub.f16x2 r195,%11,%12;mul.f16x2 r198,r195,r2;sub.f16x2 r201,%14,%15;mul.f16x2 r204,r201,r4;add.f16x2 r207,r198,r204;add.f16x2 %3,r192,r207;add.f16x2 r213,%16,%10;mul.f16x2 r216,r213,r1;add.f16x2 r219,%17,r216;add.f16x2 r222,%18,%19;mul.f16x2 r225,r222,r3;add.f16x2 r228,r219,r225;sub.f16x2 r231,%11,%12;mul.f16x2 r234,r231,r2;sub.f16x2 r237,%14,%15;mul.f16x2 r240,r237,r4;add.f16x2 r243,r234,r240;sub.f16x2 %9,r228,r243;add.f16x2 r249,%16,%10;mul.f16x2 r252,r249,r3;add.f16x2 r255,%17,r252;add.f16x2 r258,%18,%19;mul.f16x2 r261,r258,r5;add.f16x2 r264,r255,r261;sub.f16x2 r267,%11,%12;mul.f16x2 r270,r267,r4;sub.f16x2 r273,%14,%15;mul.f16x2 r276,r273,r7;add.f16x2 r279,r270,r276;add.f16x2 %5,r264,r279;add.f16x2 r285,%16,%10;mul.f16x2 r288,r285,r3;add.f16x2 r291,%17,r288;add.f16x2 r294,%18,%19;mul.f16x2 r297,r294,r5;add.f16x2 r300,r291,r297;sub.f16x2 r303,%11,%12;mul.f16x2 r306,r303,r4;sub.f16x2 r309,%14,%15;mul.f16x2 r312,r309,r7;add.f16x2 r315,r306,r312;sub.f16x2 %7,r300,r315;})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)): "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[3].y)));
};
#endif // __CUDA_ARCH__


#endif

