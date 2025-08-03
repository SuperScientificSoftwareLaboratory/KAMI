//Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//NVIDIA CORPORATION and its licensors retain all intellectual property
//and proprietary rights in and to this software, related documentation
//and any modifications thereto.  Any use, reproduction, disclosure or
//distribution of this software and related documentation without an express
//license agreement from NVIDIA CORPORATION is strictly prohibited.
//


#ifndef CUFFTDX_FFT_6_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_6_FP16_INV_PTX_HPP



#ifdef __CUDA_ARCH__
template<> __forceinline__ __device__ void cufftdx_private_function<1276, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({.reg .f32 f<29>;.reg .b32 r<251>;.reg .b64 rd<2>;mov.f32 f14,0fBF000000;{.reg .f16 low, high;cvt.rn.f16.f32 low,f14;cvt.rn.f16.f32 high,f14;mov.b32 r1,{low, high};}mov.f32 f8,0fBF5DB3D7;{.reg .f16 low, high;cvt.rn.f16.f32 low,f8;cvt.rn.f16.f32 high,f8;mov.b32 r2,{low, high};}add.f16x2 r3,%17,%13;add.f16x2 r6,%18,r3;add.f16x2 r9,%19,%14;add.f16x2 r12,%20,r9;add.f16x2 r15,%17,%13;mul.f16x2 r18,r15,r1;add.f16x2 r21,%18,r18;sub.f16x2 r24,%19,%14;mul.f16x2 r27,r24,r2;add.f16x2 r30,r21,r27;add.f16x2 r33,%17,%13;mul.f16x2 r36,r33,r1;add.f16x2 r39,%18,r36;sub.f16x2 r42,%19,%14;mul.f16x2 r45,r42,r2;sub.f16x2 r48,r39,r45;add.f16x2 r51,%19,%14;mul.f16x2 r54,r51,r1;add.f16x2 r57,%20,r54;sub.f16x2 r60,%17,%13;mul.f16x2 r63,r60,r2;sub.f16x2 r66,r57,r63;add.f16x2 r69,%19,%14;mul.f16x2 r72,r69,r1;add.f16x2 r75,%20,r72;sub.f16x2 r78,%17,%13;mul.f16x2 r81,r78,r2;add.f16x2 r84,r75,r81;{.reg .f16 low, high;cvt.rn.f16.f32 low,f14;cvt.rn.f16.f32 high,f14;mov.b32 r87,{low, high};}{.reg .f16 low, high;cvt.rn.f16.f32 low,f8;cvt.rn.f16.f32 high,f8;mov.b32 r88,{low, high};}add.f16x2 r89,%21,%15;add.f16x2 r92,%22,r89;add.f16x2 r95,%12,%16;add.f16x2 r98,%23,r95;add.f16x2 r101,%21,%15;mul.f16x2 r104,r101,r87;add.f16x2 r107,%22,r104;sub.f16x2 r110,%12,%16;mul.f16x2 r113,r110,r88;add.f16x2 r116,r107,r113;add.f16x2 r119,%21,%15;mul.f16x2 r122,r119,r87;add.f16x2 r125,%22,r122;sub.f16x2 r128,%12,%16;mul.f16x2 r131,r128,r88;sub.f16x2 r134,r125,r131;add.f16x2 r137,%12,%16;mul.f16x2 r140,r137,r87;add.f16x2 r143,%23,r140;sub.f16x2 r146,%21,%15;mul.f16x2 r149,r146,r88;sub.f16x2 r152,r143,r149;add.f16x2 r155,%12,%16;mul.f16x2 r158,r155,r87;add.f16x2 r161,%23,r158;sub.f16x2 r164,%21,%15;mul.f16x2 r167,r164,r88;add.f16x2 r170,r161,r167;mov.f32 f10,0f3F000000;{.reg .f16 low, high;cvt.rn.f16.f32 low,f10;cvt.rn.f16.f32 high,f10;mov.b32 r173,{low, high};}mov.f32 f16,0f3F5DB3D7;{.reg .f16 low, high;cvt.rn.f16.f32 low,f16;cvt.rn.f16.f32 high,f16;mov.b32 r174,{low, high};}{.reg .f16 low, high;cvt.rn.f16.f32 low,f14;cvt.rn.f16.f32 high,f14;mov.b32 r175,{low, high};}{.reg .f16 low, high;cvt.rn.f16.f32 low,f16;cvt.rn.f16.f32 high,f16;mov.b32 r176,{low, high};}mul.f16x2 r183,r116,r173;mul.f16x2 r186,r152,r174;sub.f16x2 r189,r183,r186;mul.f16x2 r192,r116,r174;fma.rn.f16x2 r195,r152,r173,r192;mul.f16x2 r199,r134,r175;mul.f16x2 r202,r170,r176;sub.f16x2 r205,r199,r202;mul.f16x2 r208,r134,r176;fma.rn.f16x2 r211,r170,r175,r208;add.f16x2 %0,r6,r92;add.f16x2 %1,r12,r98;sub.f16x2 %6,r6,r92;sub.f16x2 %7,r12,r98;add.f16x2 %2,r30,r189;add.f16x2 %3,r66,r195;sub.f16x2 %8,r30,r189;sub.f16x2 %9,r66,r195;add.f16x2 %4,r48,r205;add.f16x2 %5,r84,r211;sub.f16x2 %10,r48,r205;sub.f16x2 %11,r84,r211;})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)), "=r"(__HALF2_TO_UI(rmem[3].x)), "=r"(__HALF2_TO_UI(rmem[3].y)), "=r"(__HALF2_TO_UI(rmem[4].x)), "=r"(__HALF2_TO_UI(rmem[4].y)), "=r"(__HALF2_TO_UI(rmem[5].x)), "=r"(__HALF2_TO_UI(rmem[5].y)): "r"(__HALF2_TO_UI(rmem[3].y)), "r"(__HALF2_TO_UI(rmem[4].x)), "r"(__HALF2_TO_UI(rmem[4].y)), "r"(__HALF2_TO_UI(rmem[5].x)), "r"(__HALF2_TO_UI(rmem[5].y)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[0].y)), "r"(__HALF2_TO_UI(rmem[3].x)), "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[1].y)));
};
#endif // __CUDA_ARCH__


#endif

