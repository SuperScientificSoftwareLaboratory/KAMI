//Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//NVIDIA CORPORATION and its licensors retain all intellectual property
//and proprietary rights in and to this software, related documentation
//and any modifications thereto.  Any use, reproduction, disclosure or
//distribution of this software and related documentation without an express
//license agreement from NVIDIA CORPORATION is strictly prohibited.
//


template<> struct block_fft_record<13, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<13,   13,    1,    32,     0,     0>
    >;
};

template<> struct block_fft_record<14, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<14,   14,    1,    32,     0,     1>
    >;
};

template<> struct block_fft_record<15, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<15,   15,    1,    32,     0,     2>
    >;
};

template<> struct block_fft_record<17, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<17,   17,    1,    32,     0,     3>
    >;
};

template<> struct block_fft_record<18, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<18,   18,    1,    32,     0,     4>
    >;
};

template<> struct block_fft_record<19, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<19,   19,    1,    32,     0,     5>
    >;
};

template<> struct block_fft_record<20, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<20,   20,    1,    32,     0,     6>
    >;
};

template<> struct block_fft_record<21, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<21,   21,    1,    32,     0,     7>
    >;
};

template<> struct block_fft_record<22, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<22,   22,    1,    32,     0,     8>
    >;
};

template<> struct block_fft_record<23, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<23,   23,    1,    32,     0,     9>
    >;
};

template<> struct block_fft_record<24, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<24,   24,    1,    32,     0,    10>
    >;
};

template<> struct block_fft_record<26, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<26,   26,    1,    32,     0,    11>
    >;
};

template<> struct block_fft_record<28, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<28,   28,    1,    32,     0,    12>
    >;
};

template<> struct block_fft_record<29, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<29,   29,    1,    32,     0,    13>
    >;
};

template<> struct block_fft_record<30, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<30,   30,    1,    32,     0,    14>
    >;
};

template<> struct block_fft_record<31, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<31,   31,    1,    32,     0,    15>
    >;
};

template<> struct block_fft_record<33, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<33,   33,    1,    32,     0,    16>
    >;
};

template<> struct block_fft_record<34, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<34,   34,    1,    32,     0,    17>
    >;
};

template<> struct block_fft_record<35, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<35,   35,    1,    32,     0,    18>
    >;
};

template<> struct block_fft_record<37, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<37,   37,    1,    32,     0,    19>
    >;
};

template<> struct block_fft_record<38, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<38,   38,    1,    32,     0,    20>
    >;
};

template<> struct block_fft_record<39, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<39,   39,    1,    32,     0,    21>
    >;
};

template<> struct block_fft_record<40, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<40,   40,    1,    32,     0,    22>
    >;
};

template<> struct block_fft_record<41, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<41,   41,    1,    32,     0,    23>
    >;
};

template<> struct block_fft_record<42, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<42,   42,    1,    32,     0,    24>
    >;
};

template<> struct block_fft_record<43, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<43,   43,    1,    32,     0,    25>
    >;
};

template<> struct block_fft_record<44, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<44,   44,    1,    32,     0,    26>
    >;
};

template<> struct block_fft_record<45, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<45,   45,    1,    32,     0,    27>
    >;
};

template<> struct block_fft_record<46, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<46,   46,    1,    32,     0,    28>
    >;
};

template<> struct block_fft_record<47, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<47,   47,    1,    32,     0,    29>
    >;
};

template<> struct block_fft_record<48, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<48,   48,    1,    32,     0,    30>
    >;
};

template<> struct block_fft_record<50, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<50,   50,    1,    32,     0,    31>
    >;
};

template<> struct block_fft_record<51, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<51,   51,    1,    32,     0,    32>
    >;
};

template<> struct block_fft_record<52, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<52,   52,    1,    32,     0,    33>
    >;
};

template<> struct block_fft_record<53, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<53,   53,    1,    32,     0,    34>
    >;
};

template<> struct block_fft_record<54, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<54,   54,    1,    32,     0,    35>
    >;
};

template<> struct block_fft_record<55, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<55,   55,    1,    32,     0,    36>
    >;
};

template<> struct block_fft_record<56, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<56,   56,    1,    32,     0,    37>
    >;
};

template<> struct block_fft_record<57, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<57,   57,    1,    32,     0,    38>
    >;
};

template<> struct block_fft_record<58, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<58,   58,    1,    32,     0,    39>
    >;
};

template<> struct block_fft_record<59, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<59,   59,    1,    32,     0,    40>
    >;
};

template<> struct block_fft_record<60, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<60,   60,    1,    32,     0,    41>
    >;
};

template<> struct block_fft_record<61, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<61,   61,    1,    32,     0,    42>
    >;
};

template<> struct block_fft_record<62, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<62,   62,    1,    32,     0,    43>
    >;
};

template<> struct block_fft_record<63, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<63,   63,    1,    32,     0,    44>
    >;
};

template<> struct block_fft_record<2, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 2,    2,    1,    64,     0,    45>
    >;
};

template<> struct block_fft_record<4, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 4,    4,    1,    64,     0,    46>,
        block_fft_implementation< 2,    2,    2,   128,    16,    47>,
        block_fft_implementation< 2,    2,    2,   128,    32,    48>
    >;
};

template<> struct block_fft_record<8, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,    1,    32,     0,    49>,
        block_fft_implementation< 4,    4,    2,   128,    32,    50>,
        block_fft_implementation< 4,    4,    2,    96,    64,    51>,
        block_fft_implementation< 2,    2,    4,    80,    32,    52>,
        block_fft_implementation< 2,    2,    4,   128,    64,    53>
    >;
};

template<> struct block_fft_record<16, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,    1,   112,     0,    54>,
        block_fft_implementation< 4,    4,    4,    64,    64,    55>,
        block_fft_implementation< 4,    4,    4,    96,   128,    56>,
        block_fft_implementation< 8,    8,    2,   128,    64,    57>,
        block_fft_implementation< 8,    8,    2,   112,   128,    58>,
        block_fft_implementation< 2,    2,    8,    64,    64,    59>,
        block_fft_implementation< 2,    2,    8,    64,   128,    60>
    >;
};

template<> struct block_fft_record<32, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<32,   32,    1,    32,     0,    61>,
        block_fft_implementation< 8,    8,    4,    48,   256,    62>,
        block_fft_implementation< 8,    8,    4,    96,   128,    63>,
        block_fft_implementation< 4,    4,    8,    48,   256,    64>,
        block_fft_implementation< 4,    4,    8,    64,   128,    65>,
        block_fft_implementation< 2,    2,   16,    20,   256,    66>,
        block_fft_implementation<16,   16,    2,    64,   256,    67>,
        block_fft_implementation< 2,    2,   16,    26,   128,    68>,
        block_fft_implementation<16,   16,    2,   128,   128,    69>
    >;
};

template<> struct block_fft_record<64, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<64,   64,    1,    32,     0,    70>,
        block_fft_implementation< 8,    8,    8,    24,   512,    71>,
        block_fft_implementation< 8,    8,    8,    13,   256,    72>,
        block_fft_implementation< 4,    4,   16,     9,   512,    73>,
        block_fft_implementation< 4,    4,   16,    15,   256,    74>,
        block_fft_implementation<16,   16,    4,    25,   512,    75>,
        block_fft_implementation<16,   16,    4,    29,   256,    76>,
        block_fft_implementation<32,   32,    2,    23,   512,    77>,
        block_fft_implementation< 2,    2,   32,    14,   512,    78>,
        block_fft_implementation< 2,    2,   32,     8,   256,    79>,
        block_fft_implementation<32,   32,    2,    64,   256,    80>
    >;
};

template<> struct block_fft_record<128, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,    8,    16,   512,    81>,
        block_fft_implementation< 8,    8,   16,     8,  1024,    82>,
        block_fft_implementation<16,   16,    8,    13,  1024,    83>,
        block_fft_implementation< 8,    8,   16,    16,   512,    84>,
        block_fft_implementation< 4,    4,   32,     4,  1024,    85>,
        block_fft_implementation< 4,    4,   32,     7,   512,    86>,
        block_fft_implementation<32,   32,    4,     4,  1024,    87>,
        block_fft_implementation<32,   32,    4,    31,   512,    88>,
        block_fft_implementation< 2,    2,   64,     6,  1024,    89>,
        block_fft_implementation< 2,    2,   64,     6,   512,    90>
    >;
};

template<> struct block_fft_record<256, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,   16,     2,  1024,    91>,
        block_fft_implementation< 8,    8,   32,     4,  2048,    92>,
        block_fft_implementation< 8,    8,   32,     4,  1024,    93>,
        block_fft_implementation<16,   16,   16,     3,  2048,    94>,
        block_fft_implementation<32,   32,    8,     4,  2048,    95>,
        block_fft_implementation< 4,    4,   64,     2,  2048,    96>,
        block_fft_implementation< 4,    4,   64,     4,  1024,    97>,
        block_fft_implementation<32,   32,    8,    16,  1024,    98>,
        block_fft_implementation< 2,    2,  128,     3,  2048,    99>,
        block_fft_implementation< 2,    2,  128,     4,  1024,   100>
    >;
};

template<> struct block_fft_record<512, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 8,    8,   64,     2,  4096,   101>,
        block_fft_implementation< 8,    8,   64,     2,  2048,   102>,
        block_fft_implementation<16,   16,   32,     1,  2048,   103>,
        block_fft_implementation<32,   32,   16,     2,  4096,   104>,
        block_fft_implementation<32,   32,   16,     2,  2048,   105>,
        block_fft_implementation<16,   16,   32,     4,  4096,   106>,
        block_fft_implementation< 4,    4,  128,     2,  4096,   107>,
        block_fft_implementation< 4,    4,  128,     2,  2048,   108>,
        block_fft_implementation< 2,    2,  256,     1,  4096,   109>,
        block_fft_implementation< 2,    2,  256,     2,  2048,   110>
    >;
};

template<> struct block_fft_record<1024, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,   64,     1,  4096,   111>,
        block_fft_implementation< 8,    8,  128,     1,  8192,   112>,
        block_fft_implementation<32,   32,   32,     4,  8192,   113>,
        block_fft_implementation< 8,    8,  128,     1,  4096,   114>,
        block_fft_implementation<32,   32,   32,     4,  4096,   115>,
        block_fft_implementation<16,   16,   64,     2,  8192,   116>,
        block_fft_implementation< 4,    4,  256,     1,  8192,   117>,
        block_fft_implementation< 4,    4,  256,     1,  4096,   118>,
        block_fft_implementation< 2,    2,  512,     1,  8192,   119>,
        block_fft_implementation< 2,    2,  512,     1,  4096,   120>
    >;
};

template<> struct block_fft_record<2048, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,  128,     1,  8192,   121>,
        block_fft_implementation< 8,    8,  256,     1, 16384,   122>,
        block_fft_implementation< 8,    8,  256,     1,  8192,   123>,
        block_fft_implementation<16,   16,  128,     1, 16384,   124>,
        block_fft_implementation<32,   32,   64,     2, 16384,   125>,
        block_fft_implementation<32,   32,   64,     2,  8192,   126>,
        block_fft_implementation< 4,    4,  512,     1, 16384,   127>,
        block_fft_implementation< 4,    4,  512,     1,  8192,   128>,
        block_fft_implementation< 2,    2, 1024,     1, 16384,   129>,
        block_fft_implementation< 2,    2, 1024,     1,  8192,   130>
    >;
};

template<> struct block_fft_record<4096, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,  256,     1, 16384,   131>,
        block_fft_implementation< 8,    8,  512,     1, 32768,   132>,
        block_fft_implementation< 8,    8,  512,     1, 16384,   133>,
        block_fft_implementation<16,   16,  256,     1, 32768,   134>,
        block_fft_implementation<32,   32,  128,     1, 32768,   135>,
        block_fft_implementation<32,   32,  128,     1, 16384,   136>,
        block_fft_implementation< 4,    4, 1024,     1, 32768,   137>,
        block_fft_implementation< 4,    4, 1024,     1, 16384,   138>
    >;
};

template<> struct block_fft_record<8192, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16,  512,     1, 32768,   139>,
        block_fft_implementation<32,   32,  256,     1, 65536,   140>,
        block_fft_implementation<32,   32,  256,     1, 32768,   141>,
        block_fft_implementation< 8,    8, 1024,     1, 65536,   142>,
        block_fft_implementation<16,   16,  512,     1, 65536,   143>,
        block_fft_implementation< 8,    8, 1024,     1, 32768,   144>
    >;
};

template<> struct block_fft_record<16384, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<16,   16, 1024,     1, 65536,   145>,
        block_fft_implementation<32,   32,  512,     1, 65536,   146>
    >;
};

template<> struct block_fft_record<3, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 3,    3,    1,    64,     0,   147>
    >;
};

template<> struct block_fft_record<9, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,    1,    64,     0,   148>,
        block_fft_implementation< 3,    3,    3,    64,    72,   149>,
        block_fft_implementation< 3,    3,    3,   128,    36,   150>
    >;
};

template<> struct block_fft_record<27, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,    1,    32,     0,   151>,
        block_fft_implementation< 9,    9,    3,    21,   108,   152>,
        block_fft_implementation< 9,    9,    3,    10,   216,   153>,
        block_fft_implementation< 3,    3,    9,    26,   216,   154>,
        block_fft_implementation< 3,    3,    9,    24,   108,   155>
    >;
};

template<> struct block_fft_record<81, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,    9,    14,   648,   156>,
        block_fft_implementation< 9,    9,    9,     7,   324,   157>,
        block_fft_implementation<27,   27,    3,    10,   324,   158>,
        block_fft_implementation<27,   27,    3,     9,   648,   159>,
        block_fft_implementation< 3,    3,   27,     7,   648,   160>,
        block_fft_implementation< 3,    3,   27,     9,   324,   161>
    >;
};

template<> struct block_fft_record<243, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,    9,    14,  1944,   162>,
        block_fft_implementation<27,   27,    9,    14,   972,   163>,
        block_fft_implementation< 9,    9,   27,     9,  1944,   164>,
        block_fft_implementation< 9,    9,   27,     9,   972,   165>,
        block_fft_implementation< 3,    3,   81,     3,  1944,   166>,
        block_fft_implementation< 3,    3,   81,     3,   972,   167>
    >;
};

template<> struct block_fft_record<729, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,   27,    14,  5832,   168>,
        block_fft_implementation<27,   27,   27,     9,  2916,   169>,
        block_fft_implementation< 9,    9,   81,     3,  5832,   170>,
        block_fft_implementation< 9,    9,   81,     3,  2916,   171>,
        block_fft_implementation< 3,    3,  243,     1,  5832,   172>,
        block_fft_implementation< 3,    3,  243,     1,  2916,   173>
    >;
};

template<> struct block_fft_record<2187, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 9,    9,  243,     1, 17496,   174>,
        block_fft_implementation<27,   27,   81,     3, 17496,   175>,
        block_fft_implementation< 9,    9,  243,     1,  8748,   176>,
        block_fft_implementation<27,   27,   81,     3,  8748,   177>,
        block_fft_implementation< 3,    3,  729,     1, 17496,   178>,
        block_fft_implementation< 3,    3,  729,     1,  8748,   179>
    >;
};

template<> struct block_fft_record<6561, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<27,   27,  243,     1, 26244,   180>,
        block_fft_implementation<27,   27,  243,     1, 52488,   181>,
        block_fft_implementation< 9,    9,  729,     1, 26244,   182>,
        block_fft_implementation< 9,    9,  729,     1, 52488,   183>
    >;
};

template<> struct block_fft_record<5, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 5,    5,    1,    32,     0,   184>
    >;
};

template<> struct block_fft_record<25, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,    1,    32,     0,   185>,
        block_fft_implementation< 5,    5,    5,   128,   200,   186>,
        block_fft_implementation< 5,    5,    5,    25,   100,   187>
    >;
};

template<> struct block_fft_record<125, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,    5,    25,   500,   188>,
        block_fft_implementation<25,   25,    5,    25,  1000,   189>,
        block_fft_implementation< 5,    5,   25,     5,  1000,   190>,
        block_fft_implementation< 5,    5,   25,     5,   500,   191>
    >;
};

template<> struct block_fft_record<625, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,   25,     5,  2500,   192>,
        block_fft_implementation<25,   25,   25,     5,  5000,   193>,
        block_fft_implementation< 5,    5,  125,     1,  5000,   194>,
        block_fft_implementation< 5,    5,  125,     1,  2500,   195>
    >;
};

template<> struct block_fft_record<3125, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,  125,     1, 12500,   196>,
        block_fft_implementation<25,   25,  125,     3, 25000,   197>,
        block_fft_implementation< 5,    5,  625,     1, 25000,   198>,
        block_fft_implementation< 5,    5,  625,     1, 12500,   199>
    >;
};

template<> struct block_fft_record<15625, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<25,   25,  625,     1, 62500,   200>
    >;
};

template<> struct block_fft_record<7, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,    1,    64,     0,   201>
    >;
};

template<> struct block_fft_record<49, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<49,   49,    1,    32,     0,   202>,
        block_fft_implementation< 7,    7,    7,     9,   392,   203>,
        block_fft_implementation< 7,    7,    7,     9,   196,   204>
    >;
};

template<> struct block_fft_record<343, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,   49,     5,  2744,   205>,
        block_fft_implementation< 7,    7,   49,     5,  1372,   206>
    >;
};

template<> struct block_fft_record<2401, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 7,    7,  343,     1, 19208,   207>,
        block_fft_implementation< 7,    7,  343,     1,  9604,   208>
    >;
};

template<> struct block_fft_record<11, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,    1,    96,     0,   209>
    >;
};

template<> struct block_fft_record<121, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,   11,    23,   968,   210>,
        block_fft_implementation<11,   11,   11,    23,   484,   211>
    >;
};

template<> struct block_fft_record<1331, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<11,   11,  121,     1,  5324,   212>,
        block_fft_implementation<11,   11,  121,     1, 10648,   213>
    >;
};

template<> struct block_fft_record<6, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    1,    32,     0,   214>
    >;
};

template<> struct block_fft_record<36, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,    6,    64,   144,   215>,
        block_fft_implementation< 6,    6,    6,    64,   288,   216>,
        block_fft_implementation<36,   36,    1,    36,     0,   217>
    >;
};

template<> struct block_fft_record<216, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,   36,     7,  1728,   218>,
        block_fft_implementation< 6,    6,   36,     7,   864,   219>
    >;
};

template<> struct block_fft_record<1296, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation< 6,    6,  216,     2, 10368,   220>,
        block_fft_implementation< 6,    6,  216,     1,  5184,   221>
    >;
};

template<> struct block_fft_record<10, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,    1,    32,     0,   222>
    >;
};

template<> struct block_fft_record<100, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,   10,    25,   400,   223>,
        block_fft_implementation<10,   10,   10,    25,   800,   224>
    >;
};

template<> struct block_fft_record<1000, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10,  100,     5,  8000,   225>,
        block_fft_implementation<10,   10,  100,     5,  4000,   226>
    >;
};

template<> struct block_fft_record<10000, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<10,   10, 1000,     1, 40000,   228>
    >;
};

template<> struct block_fft_record<12, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,    1,    64,     0,   229>
    >;
};

template<> struct block_fft_record<144, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,   12,    21,  1152,   230>,
        block_fft_implementation<12,   12,   12,    21,   576,   231>
    >;
};

template<> struct block_fft_record<1728, float, fft_type::c2c, fft_direction::forward, 750> {
    static constexpr bool defined = true;
    using blobs = type_list<
        block_fft_implementation<12,   12,  144,     1, 13824,   232>,
        block_fft_implementation<12,   12,  144,     1,  6912,   233>
    >;
};
