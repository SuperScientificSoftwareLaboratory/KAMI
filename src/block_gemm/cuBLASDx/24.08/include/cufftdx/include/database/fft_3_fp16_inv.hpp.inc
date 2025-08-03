//Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//NVIDIA CORPORATION and its licensors retain all intellectual property
//and proprietary rights in and to this software, related documentation
//and any modifications thereto.  Any use, reproduction, disclosure or
//distribution of this software and related documentation without an express
//license agreement from NVIDIA CORPORATION is strictly prohibited.
//


#ifndef CUFFTDX_FFT_3_FP16_INV_PTX_HPP
#define CUFFTDX_FFT_3_FP16_INV_PTX_HPP



#ifdef __CUDA_ARCH__
template<> __forceinline__ __device__ void cufftdx_private_function<1209, __half2, 1>(cufftdx::detail::complex<__half2> *rmem, unsigned smem){

asm volatile (R"({.reg .f32 f<5>;.reg .b32 r<87>;.reg .b64 rd<2>;mov.f32 f2,0fBF000000;{.reg .f16 low, high;cvt.rn.f16.f32 low,f2;cvt.rn.f16.f32 high,f2;mov.b32 r1,{low, high};}mov.f32 f4,0fBF5DB3D7;{.reg .f16 low, high;cvt.rn.f16.f32 low,f4;cvt.rn.f16.f32 high,f4;mov.b32 r2,{low, high};}add.f16x2 r3,%6,%7;add.f16x2 %0,%8,r3;add.f16x2 r9,%9,%10;add.f16x2 %1,%11,r9;add.f16x2 r15,%6,%7;mul.f16x2 r18,r15,r1;add.f16x2 r21,%8,r18;sub.f16x2 r24,%9,%10;mul.f16x2 r27,r24,r2;add.f16x2 %2,r21,r27;add.f16x2 r33,%6,%7;mul.f16x2 r36,r33,r1;add.f16x2 r39,%8,r36;sub.f16x2 r42,%9,%10;mul.f16x2 r45,r42,r2;sub.f16x2 %4,r39,r45;add.f16x2 r51,%9,%10;mul.f16x2 r54,r51,r1;add.f16x2 r57,%11,r54;sub.f16x2 r60,%6,%7;mul.f16x2 r63,r60,r2;sub.f16x2 %3,r57,r63;add.f16x2 r69,%9,%10;mul.f16x2 r72,r69,r1;add.f16x2 r75,%11,r72;sub.f16x2 r78,%6,%7;mul.f16x2 r81,r78,r2;add.f16x2 %5,r75,r81;})"
     : "=r"(__HALF2_TO_UI(rmem[0].x)), "=r"(__HALF2_TO_UI(rmem[0].y)), "=r"(__HALF2_TO_UI(rmem[1].x)), "=r"(__HALF2_TO_UI(rmem[1].y)), "=r"(__HALF2_TO_UI(rmem[2].x)), "=r"(__HALF2_TO_UI(rmem[2].y)): "r"(__HALF2_TO_UI(rmem[1].x)), "r"(__HALF2_TO_UI(rmem[2].x)), "r"(__HALF2_TO_UI(rmem[0].x)), "r"(__HALF2_TO_UI(rmem[1].y)), "r"(__HALF2_TO_UI(rmem[2].y)), "r"(__HALF2_TO_UI(rmem[0].y)));
};
#endif // __CUDA_ARCH__


#endif

