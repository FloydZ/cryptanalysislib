#include <immintrin.h>
// TODO docs 
// src: https://github.com/InstLatx64/InstLatX64_Demo/

/// NOTE: apparently this function is not in the normal list of intrinsics?
static inline 
__m512i _mm512_setr_epi8 (char __e63, char __e62, char __e61, char __e60, char __e59,            
                          char __e58, char __e57, char __e56, char __e55, char __e54, char __e53,
                          char __e52, char __e51, char __e50, char __e49, char __e48, char __e47,
                          char __e46, char __e45, char __e44, char __e43, char __e42, char __e41,
                          char __e40, char __e39, char __e38, char __e37, char __e36, char __e35,
                          char __e34, char __e33, char __e32, char __e31, char __e30, char __e29,
                          char __e28, char __e27, char __e26, char __e25, char __e24, char __e23,
                          char __e22, char __e21, char __e20, char __e19, char __e18, char __e17,
                          char __e16, char __e15, char __e14, char __e13, char __e12, char __e11,
                          char __e10, char __e9,  char __e8,  char __e7,  char __e6,  char __e5,
                          char __e4,  char __e3,  char __e2,  char __e1,  char __e0) noexcept {
  return __extension__ (__m512i)(__v64qi) {
    __e63, __e62, __e61, __e60, __e59,            
    __e58, __e57, __e56, __e55, __e54, __e53,
    __e52, __e51, __e50, __e49, __e48, __e47,
    __e46, __e45, __e44, __e43, __e42, __e41,
    __e40, __e39, __e38, __e37, __e36, __e35,
    __e34, __e33, __e32, __e31, __e30, __e29,
    __e28, __e27, __e26, __e25, __e24, __e23,
    __e22, __e21, __e20, __e19, __e18, __e17,
    __e16, __e15, __e14, __e13, __e12, __e11,
    __e10, __e9,  __e8,  __e7,  __e6,  __e5,
    __e4,  __e3,  __e2,  __e1,  __e0};
}


//c ? a : b
//imm8 = 0xE4
//
//a				1	1	1	1	0	0	0	0
//b				1	1	0	0	1	1	0	0
//c				1	0	1	0	1	0	1	0
//c ? a : b		1	1	1	0	0	1	0	0

// A signed addition overflow will happen if, and only if:
// - the signs of both inputs are the same, and
// - the sign of the sum (when added with wrap-around) is different from the input:
// sign(a) = sign(b),  sign(a + b) != sign(a)
//eq(a, b) & neq(a, c)
//imm8 = 0x42
//a							1	1	1	1	0	0	0	0
//b							1	1	0	0	1	1	0	0
//a + b						1	0	1	0	1	0	1	0
//eq(a, b) & neq(a, a + b)	0	1	0	0	0	0	1	0
static inline
__m512i _mm512_adds_epi32(const __m512i a,
                          const __m512i b) noexcept {
	__m512i add		= _mm512_add_epi32(a, b);

	__m512i sign	= _mm512_srai_epi32(add, 31);
	__m512i of		= _mm512_ternarylogic_epi32(a, b, add, 0x42);

	__m512i ofmask	= _mm512_srai_epi32(of, 31);
	__m512i ofvalue	= _mm512_xor_si512(_mm512_set1_epi32(LONG_MIN), sign);

	__m512i val		= _mm512_ternarylogic_epi32(ofvalue, add, ofmask, 0xe4);
	return	val;
}

static inline
__m512i _mm512_adds_epi64(const __m512i a,
                          const __m512i b) noexcept {
	__m512i add		= _mm512_add_epi64(a, b);

	__m512i sign	= _mm512_srai_epi64(add, 63);
	__m512i of		= _mm512_ternarylogic_epi64(a, b, add, 0x42);

	__m512i ofmask	= _mm512_srai_epi64(of, 63);
	__m512i ofvalue	= _mm512_xor_si512(_mm512_set1_epi64(LLONG_MIN), sign);

	__m512i val		= _mm512_ternarylogic_epi64(ofvalue, add, ofmask, 0xe4);
	return	val;
}

// A signed substraction overflow will happen if, and only if:
// - if the signs of the two inputs are different, and
// - the sign of the difference (when substracted with wrap-around) is different from the minuend:
// sign(a) != sign(b),  sign(a - b) != sign(a)
//imm8 = 0x18
//a							1	1	1	1	0	0	0	0
//b							1	1	0	0	1	1	0	0
//a - b						1	0	1	0	1	0	1	0
//neq(a, b) & neq(a - b, a)	0	0	0	1	1	0	0	0
static inline
__m512i _mm512_subs_epi32(const __m512i a,
                          const __m512i b) noexcept {
	__m512i sub		= _mm512_sub_epi32(a, b);

	__m512i sign	= _mm512_srai_epi32(sub, 31);
	__m512i of		= _mm512_ternarylogic_epi32(a, b, sub, 0x18);

	__m512i ofmask	= _mm512_srai_epi32(of, 31);
	__m512i ofvalue	= _mm512_xor_si512(_mm512_set1_epi32(LONG_MIN), sign);

	__m512i val		= _mm512_ternarylogic_epi32(ofvalue, sub, ofmask, 0xe4);
	return	val;
}

static inline
__m512i _mm512_subs_epi64(const __m512i a,
                          const __m512i b) noexcept {
	__m512i sub		= _mm512_sub_epi64(a, b);

	__m512i sign	= _mm512_srai_epi64(sub, 63);
	__m512i of		= _mm512_ternarylogic_epi64(a, b, sub, 0x18);

	__m512i ofmask	= _mm512_srai_epi64(of, 63);
	__m512i ofvalue	= _mm512_xor_si512(_mm512_set1_epi64(LLONG_MIN), sign);

	__m512i val		= _mm512_ternarylogic_epi64(ofvalue, sub, ofmask, 0xe4);
	return	val;
}


//~a
//imm8 = 0x0F
//
//a		1	1	1	1	0	0	0	0
//b		1	1	0	0	1	1	0	0
//c		1	0	1	0	1	0	1	0
//~a	0	0	0	0	1	1	1	1
static inline
__m512i _mm512_adds_epu32(const __m512i a,
                          const __m512i b) noexcept {
	return _mm512_add_epi32(_mm512_min_epu32(a, _mm512_ternarylogic_epi32(b, b, b, 0x0f)), b);
}

static inline
__m512i _mm512_adds_epu64(const __m512i a,
                          const __m512i b) {
	return _mm512_add_epi64(_mm512_min_epu64(a, _mm512_ternarylogic_epi64(b, b, b, 0x0f)), b);
}

static inline
__m512i _mm512_subs_epu32(const __m512i a,
                          const __m512i b) {
	return _mm512_sub_epi32(_mm512_max_epu32(a, b), b);
}

static inline
__m512i _mm512_subs_epu64(const __m512i a, 
                          const __m512i b) {
	return _mm512_sub_epi64(_mm512_max_epu64(a, b), b);
}



//the Zen4 versions consider the peculiarities of TERNLOG handling on AMD CPUs:
//https://uops.info/html-lat/ZEN4/VPTERNLOGD_ZMM_ZMM_ZMM_I8-Measurements.html
//Instruction: VPTERNLOGD ZMM0, ZMM1, ZMM2, 2
//Code:
//	vpternlogd zmm0,zmm1,zmm2,0x2
//	vandpd zmm0,zmm0,zmm0
//	vandpd zmm0,zmm0,zmm0
//	vandpd zmm0,zmm0,zmm0
//	vandpd zmm0,zmm0,zmm0
//	vandpd zmm0,zmm0,zmm0
//Results:
//MPERF: 5.17
//APERF: 6.0 -> 1 clk
//
//Instruction: VPTERNLOGD ZMM1, ZMM0, ZMM2, 2
//Code:
//	vpternlogd zmm1,zmm0,zmm2,0x2
//	vandpd zmm0,zmm1,zmm1
//	vandpd zmm0,zmm0,zmm0
//	vandpd zmm0,zmm0,zmm0
//	vandpd zmm0,zmm0,zmm0
//	vandpd zmm0,zmm0,zmm0
//Results:
//MPERF: 6.11
//APERF: 7.06 -> 2 clks
//
//Instruction: VPTERNLOGD ZMM1, ZMM2, ZMM0, 2
//Code:
//	vpternlogd zmm1,zmm2,zmm0,0x2
//	vandpd zmm0,zmm1,zmm1
//	vandpd zmm0,zmm0,zmm0
//	vandpd zmm0,zmm0,zmm0
//	vandpd zmm0,zmm0,zmm0
//	vandpd zmm0,zmm0,zmm0
//Results:
//MPERF: 6.19
//APERF: 7.0 -> 2 clks

//a ? b : c
//imm8 = 0xCA
//a				1	1	1	1	0	0	0	0
//b				1	1	0	0	1	1	0	0
//c				1	0	1	0	1	0	1	0
//a ? b : c		1	1	0	0	1	0	1	0

// A signed addition overflow will happen if, and only if:
// - if the signs of both inputs are the same, and 
// - the sign of the sum (when added with wrap-around) is different from the input:
// sign(a) = sign(b),  sign(a + b) != sign(a)
//imm8 = 0x18
//a + b						1	1	1	1	0	0	0	0
//a							1	1	0	0	1	1	0	0
//b							1	0	1	0	1	0	1	0
//eq(a, b) & neq(a+b, a)	0	0	0	1	1	0	0	0
static inline
__m512i _mm512_adds_Zen4_epi32(const __m512i a,
                               const __m512i b) noexcept {
	__m512i add		= _mm512_add_epi32(a, b);

	__m512i sign	= _mm512_srai_epi32(add, 31);
	__m512i of		= _mm512_ternarylogic_epi32(add, a, b, 0x18);

	__m512i ofmask	= _mm512_srai_epi32(of, 31);
	__m512i ofvalue	= _mm512_xor_si512(sign, _mm512_set1_epi32(LONG_MIN));

	__m512i val		= _mm512_ternarylogic_epi32(ofmask, ofvalue, add, 0xca);
	return	val;
}

static inline
__m512i _mm512_adds_Zen4_epi64(const __m512i a,
                               const __m512i b) noexcept {
	__m512i add		= _mm512_add_epi64(a, b);

	__m512i sign	= _mm512_srai_epi64(add, 63);
	__m512i of		= _mm512_ternarylogic_epi64(add, a, b, 0x18);

	__m512i ofmask	= _mm512_srai_epi64(of, 63);
	__m512i ofvalue	= _mm512_xor_si512(sign, _mm512_set1_epi64(LLONG_MIN));

	__m512i val		= _mm512_ternarylogic_epi64(ofmask, ofvalue, add, 0xca);
	return	val;
}

// A signed substraction overflow will happen if, and only if:
// - the signs of the two inputs are different, and 
// - the sign of the difference (when substracted with wrap-around) is different from the minuend:
// sign(a) != sign(b),  sign(a - b) != sign(a)
//imm8 = 0x24
//a	- b						1	1	1	1	0	0	0	0
//a							1	1	0	0	1	1	0	0
//b							1	0	1	0	1	0	1	0
//neq(a, b) & neq(a - b, a)	0	0	1	0	0	1	0	0
static inline
__m512i _mm512_subs_Zen4_epi32(const __m512i a,
                               const __m512i b) noexcept {
	__m512i sub		= _mm512_sub_epi32(a, b);

	__m512i sign	= _mm512_srai_epi32(sub, 31);
	__m512i of		= _mm512_ternarylogic_epi32(sub, a, b, 0x24);

	__m512i ofmask	= _mm512_srai_epi32(of, 31);
	__m512i ofvalue	= _mm512_xor_si512(sign, _mm512_set1_epi32(LONG_MIN));

	__m512i val		= _mm512_ternarylogic_epi32(ofmask, ofvalue, sub, 0xca);
	return	val;
}

static inline
__m512i _mm512_subs_Zen4_epi64(const __m512i a,
                               const __m512i b) noexcept {
	__m512i sub		= _mm512_sub_epi64(a, b);

	__m512i sign	= _mm512_srai_epi64(sub, 63);
	__m512i of		= _mm512_ternarylogic_epi64(sub, a, b, 0x24);

	__m512i ofmask	= _mm512_srai_epi64(of, 63);
	__m512i ofvalue	= _mm512_xor_si512(sign, _mm512_set1_epi64(LLONG_MIN));

	__m512i val		= _mm512_ternarylogic_epi64(ofmask, ofvalue, sub, 0xca);
	return	val;
}

//~a
//imm8 = 0x0F
//
//a		1	1	1	1	0	0	0	0
//b		1	1	0	0	1	1	0	0
//c		1	0	1	0	1	0	1	0
//~a	0	0	0	0	1	1	1	1
static inline
__m512i _mm512_adds_Zen4_epu32(const __m512i a,
                               const __m512i b) noexcept {
	__m512i u = _mm512_undefined_epi32();
	return _mm512_add_epi32(_mm512_min_epu32(a, _mm512_ternarylogic_epi32(b, u, u, 0x0f)), b);
}

static inline
__m512i _mm512_adds_Zen4_epu64(const __m512i a,
                               const __m512i b) noexcept {
	__m512i u = _mm512_undefined_epi32(); // there isn't _mm512_undefined_epi64();
	return _mm512_add_epi64(_mm512_min_epu64(a, _mm512_ternarylogic_epi64(b, u, u, 0x0f)), b);
}

static inline
__m512i _mm512_subs_Zen4_epu32(const __m512i a,
                               const __m512i b) noexcept {
	return _mm512_sub_epi32(_mm512_max_epu32(a, b), b);
}

static inline
__m512i _mm512_subs_Zen4_epu64(const __m512i a,
                               const __m512i b) noexcept {
	return _mm512_sub_epi64(_mm512_max_epu64(a, b), b);
}


static inline
__m256i _mm256_bsrli_epi256(const __m256i a,
                            const int b) noexcept {
    //left shift is correct here
	return _mm256_maskz_compress_epi8(~0UL << b, a); 
}

static inline
__m256i _mm256_bslli_epi256(const __m256i a, 
                            const int b) noexcept {
	return _mm256_maskz_expand_epi8(~0UL << b, a); 
}

static inline
__m256i _mm256_palignr_epi256(const __m256i a,
                              const __m256i b,
                              const int c) noexcept {
	const __m256i disp = _mm256_setr_epi8(
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
		0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f);
	__m256i idx = _mm256_add_epi8(disp, _mm256_set1_epi8(c));
	return _mm256_permutex2var_epi8(a, idx, b);
}

static inline
__m256i _mm256_palignl_epi256(const __m256i a,
                              const __m256i b,
                              const int c) noexcept {
	const __m256i disp = _mm256_setr_epi8(
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
		0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f);
	__m256i idx = _mm256_sub_epi8(disp, _mm256_set1_epi8(c));
	return _mm256_permutex2var_epi8(a, idx, b);
}

static inline
__m256i _mm256_rotater_epi256(const __m256i a,
                              const int c) {
	const __m256i disp = _mm256_setr_epi8(
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
		0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f);
	__m256i idx = _mm256_add_epi8(disp, _mm256_set1_epi8(c));
	return _mm256_permutexvar_epi8(idx, a);
}

static inline
__m256i _mm256_rotatel_epi256(const __m256i a,
                              const int c) noexcept {
	const __m256i disp = _mm256_setr_epi8(
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
		0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f);
	__m256i idx = _mm256_sub_epi8(disp, _mm256_set1_epi8(c));
	return _mm256_permutexvar_epi8(idx, a);
}

//BSRLI older
//vpbroadcastb	zmm1, rax					;P5
//mov			rcx, -1						;P0156B
//shlx			rcx, rcx, rax				;P06
//kmovq			k1, rcx						;P5
//vpaddb		zmm1, zmm1, [disp]			;P05+P23A
//vpermb		zmm0 {k1}{z}, zmm1, zmm0	;P5
//
//7 uops: P0156B+P06+3*P5+P23A+P05

//shorter:
//mov			rcx, -1						;P0156B
//shlx			rcx, rcx, rax				;P06
//kmovq			k1, rcx						;P5
//vpcompressb	zmm0 {k1}{z}, zmm0			;2*P5
// 
//5 uops: P0156B+P06+3*P5
static inline
__m512i _mm512_bsrli_epi512(const __m512i a,
                            const int b) noexcept {
	return _mm512_maskz_compress_epi8(~0ULL << b, a); //left shift is correct here
}

static inline
__m512i _mm512_bslli_epi512(const __m512i a,
                            const int b) {
	return _mm512_maskz_expand_epi8(~0ULL << b, a); 
}

static inline
__m512i _mm512_palignr_epi512(const __m512i a, 
                              const __m512i b,
                              const int c) {
	const __m512i disp = _mm512_setr_epi8(
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
		0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
		0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
		0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f);
	__m512i idx = _mm512_add_epi8(disp, _mm512_set1_epi8(c));
	return _mm512_permutex2var_epi8(a, idx, b);
}

static inline
__m512i _mm512_palignl_epi512(const __m512i a,
                              const __m512i b,
                              const int c) noexcept {
	const __m512i disp = _mm512_setr_epi8(
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
		0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
		0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
		0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f);
	__m512i idx = _mm512_sub_epi8(disp, _mm512_set1_epi8(c));
	return _mm512_permutex2var_epi8(a, idx, b);
}

static inline
__m512i _mm512_rotater_epi512(const __m512i a,
                              const int c) noexcept {
	const __m512i disp = _mm512_setr_epi8(
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
		0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
		0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
		0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f);
	__m512i idx = _mm512_add_epi8(disp, _mm512_set1_epi8(c));
	return _mm512_permutexvar_epi8(idx, a);
}

static inline
__m512i _mm512_rotatel_epi512(const __m512i a,
                              const int c) noexcept {
	const __m512i disp = _mm512_setr_epi8(
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
		0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
		0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
		0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f);
	__m512i idx = _mm512_sub_epi8(disp, _mm512_set1_epi8(c));
	return _mm512_permutexvar_epi8(idx, a);
}




//credit: @geofflangdale https://twitter.com/geofflangdale/status/1609575574946865154
static inline
uint32_t _mm512_reduce2_add_epu8(const __m512i z) noexcept {
	__m128i permb_collect	= _mm_setr_epi64((__m64)0x3830282018100800, (__m64)0x3931292119110901);
	__m512i sad				= _mm512_sad_epu8(_mm512_setzero_si512(), z);
	__m128i permb			= _mm512_castsi512_si128(_mm512_permutexvar_epi8(_mm512_zextsi128_si512(permb_collect), sad));
	__m128i sad2			= _mm_sad_epu8(_mm_setzero_si128(), permb);
	return _mm_cvtsi128_si32(_mm_add_epi32(sad2, _mm_srli_si128(sad2, 7)));
}

static inline 
uint32_t _mm512_reduce2_add_epu16(const __m512i z) noexcept {
	__m128i pshufb_collect	= _mm_setr_epi32(0x06040200, 0x0e0c0a08, 0x07050301, 0x0f0d0b09);
	__m512i pshufb			= _mm512_shuffle_epi8(z, _mm512_broadcast_i32x4(pshufb_collect));
	__m512i sad				= _mm512_sad_epu8(_mm512_setzero_si512(), pshufb);
	__m128i permb_collect	= _mm_setr_epi32(0x30201000, 0x31211101, 0x38281808, 0x39291909);
	__m128i permb			= _mm512_castsi512_si128(_mm512_permutexvar_epi8(_mm512_zextsi128_si512(permb_collect), sad));
	__m128i dbsad			= _mm_maskz_dbsad_epu8(0x55, permb, _mm_setzero_si128(), 0);
	__m128i add				= _mm_add_epi32(_mm_srli_epi64(dbsad, 24), dbsad);
	return _mm_cvtsi128_si32(_mm_add_epi32(add, _mm_srli_si128(add, 7)));
}

static inline
uint64_t _mm512_reduce2_add_epu32(const __m512i z) noexcept {
	__m128i transpose_4x4	= _mm_setr_epi32(0x0c080400, 0x0d090501, 0x0e0a0602, 0x0f0b0703);
	__m512i transpose		= _mm512_shuffle_epi8(z, _mm512_broadcast_i32x4(transpose_4x4));
	__m512i dbsad			= _mm512_maskz_dbsad_epu8(0x55555555, transpose, _mm512_setzero_si512(), 0);
	__m512i permb_collect	= _mm512_setr_epi32(0x30201000, 0x3f3f3f3f, 0x31211101, 0x34241404, 0x35251505, 0x38281808, 0x39291909, 0x3c2c1c0c, 0x3d2d1d0d, 0x3f3f3f3f, 0x3f3f3f3f, 0x3f3f3f3f, 0x3f3f3f3f, 0x3f3f3f3f, 0x3f3f3f3f, 0x3f3f3f3f);
	__m512i permb			= _mm512_permutexvar_epi8(permb_collect, dbsad);
	__m512i sad				= _mm512_sad_epu8(permb, _mm512_setzero_si512());
	__m128i permb2			= _mm512_castsi512_si128(_mm512_permutexvar_epi8(_mm512_zextsi128_si512(_mm_setr_epi64((__m64)0x3f3f3f2018100800, (__m64)0x3f3f21191109013f)), sad));
	return _mm_cvtsi128_si64(_mm_add_epi64(permb2, _mm_unpackhi_epi64(permb2, permb2)));
}

static inline
uint64_t _mm512_reduce2_add_epu64(const __m512i z) noexcept {
	__m512i transpose_8x8	= _mm512_setr_epi64(0x3830282018100800, 0x3931292119110901, 0x3a322a221a120a02, 0x3b332b231b130b03, 0x3c342c241c140c04, 0x3d352d251d150d05, 0x3e362e261e160e06, 0x3f372f271f170f07);
	__m512i transpose		= _mm512_permutexvar_epi8(transpose_8x8, z);
	__m512i sad				= _mm512_sad_epu8(transpose, _mm512_setzero_si512());
	__m128i collect			= _mm512_castsi512_si128(_mm512_permutexvar_epi8(_mm512_zextsi128_si512(_mm_setr_epi64((__m64)0x3830282018100800, (__m64)0x312921191109013f)), sad));
	return _mm_cvtsi128_si64(_mm_add_epi64(collect, _mm_unpackhi_epi64(collect, collect)));
}

static inline
uint64_t _mm512_reduce2_add_epu128(const __m512i z,
                                   uint64_t* hi) noexcept {
	__m512i transpose_8x8	= _mm512_setr_epi64(0x3830282018100800, 0x3931292119110901, 0x3a322a221a120a02, 0x3b332b231b130b03, 0x3c342c241c140c04, 0x3d352d251d150d05, 0x3e362e261e160e06, 0x3f372f271f170f07);
	__m512i transpose		= _mm512_permutexvar_epi8(transpose_8x8, z);
	__m512i sad				= _mm512_sad_epu8(transpose, _mm512_setzero_si512());
	__m128i collect0		= _mm512_castsi512_si128(_mm512_permutexvar_epi8(_mm512_zextsi128_si512(_mm_setr_epi64((__m64)0x3830282018100800, (__m64)0x3f38302820181008)), sad));
	__m128i collect1		= _mm512_castsi512_si128(_mm512_permutexvar_epi8(_mm512_zextsi128_si512(_mm_setr_epi64((__m64)0x312921191109013f, (__m64)0x3931292119110901)), sad));
	__m128i add				= _mm_add_epi64(collect0, collect1);
	*hi						= _mm_extract_epi64(add, 1) >> 56;
	return _mm_cvtsi128_si64(add);
}



/* Rotate right words by imm8 bits mod 16 */
#define _mm_ror_vbmi2_epi16(a, cnt)					_mm_shrdi_epi16(a, a, cnt)
#define _mm_mask_ror_vbmi2_epi16(a, k, b, cnt)		_mm_mask_shrdi_epi16(a, k, b, b, cnt)
#define _mm_maskz_ror_vbmi2_epi16(k, a, cnt)		_mm_maskz_shrdi_epi16(k, a, a, cnt)
#define _mm256_ror_vbmi2_epi16(a, cnt)				_mm256_shrdi_epi16(a, a, cnt)
#define _mm256_mask_ror_vbmi2_epi16(a, k, b, cnt)	_mm256_mask_shrdi_epi16(a, k, b, b, cnt)
#define _mm256_maskz_ror_vbmi2_epi16(k, a, cnt)		_mm256_maskz_shrdi_epi16(k, a, a, cnt)
#define _mm512_ror_vbmi2_epi16(a, cnt)				_mm512_shrdi_epi16(a, a, cnt)
#define _mm512_mask_ror_vbmi2_epi16(a, k, b, cnt)	_mm512_mask_shrdi_epi16(a, k, b, b, cnt)
#define _mm512_maskz_ror_vbmi2_epi16(k, a, cnt)		_mm512_maskz_shrdi_epi16(k, a, a, cnt)

/* Rotate left by words by imm8 bits mod 16 */
#define _mm_rol_vbmi2_epi16(a, cnt)					_mm_shldi_epi16(a, a, cnt)
#define _mm_mask_rol_vbmi2_epi16(a, k, b, cnt)		_mm_mask_shldi_epi16(a, k, b, b, cnt)
#define _mm_maskz_rol_vbmi2_epi16(k, a, cnt)		_mm_maskz_shldi_epi16(k, a, a, cnt)
#define _mm256_rol_vbmi2_epi16(a, cnt)				_mm256_shldi_epi16(a, a, cnt)
#define _mm256_mask_rol_vbmi2_epi16(a, k, b, cnt)	_mm256_mask_shldi_epi16(a, k, b, b, cnt)
#define _mm256_maskz_rol_vbmi2_epi16(k, a, cnt)		_mm256_maskz_shldi_epi16(k, a, a, cnt)
#define _mm512_rol_vbmi2_epi16(a, cnt)				_mm512_shldi_epi16(a, a, cnt)
#define _mm512_mask_rol_vbmi2_epi16(a, k, b, cnt)	_mm512_mask_shldi_epi16(a, k, b, b, cnt)
#define _mm512_maskz_rol_vbmi2_epi16(k, a, cnt)		_mm512_maskz_shldi_epi16(k, a, a, cnt)

/* Variable rotate right words by cnt bits, cnt mod 16 */
#define _mm_rorv_vbmi2_epi16(a, cnt)				_mm_shrdv_epi16(a, a, cnt)
#define _mm_mask_rorv_vbmi2_epi16(a, k, b, cnt)		_mm_mask_blend_epi16(k, a, _mm_shrdv_epi16(b, b, cnt))
#define _mm_maskz_rorv_vbmi2_epi16(k, a, cnt)		_mm_maskz_shrdv_epi16(k, a, a, cnt)
#define _mm256_rorv_vbmi2_epi16(a, cnt)				_mm256_shrdv_epi16(a, a, cnt)
#define _mm256_mask_rorv_vbmi2_epi16(a, k, b, cnt)	_mm256_mask_blend_epi16(k, a, _mm256_shrdv_epi16(b, b, cnt))
#define _mm256_maskz_rorv_vbmi2_epi16(k, a, cnt)	_mm256_maskz_shrdv_epi16(k, a, a, cnt)
#define _mm512_rorv_vbmi2_epi16(a, cnt)				_mm512_shrdv_epi16(a, a, cnt)
#define _mm512_mask_rorv_vbmi2_epi16(a, k, b, cnt)	_mm512_mask_blend_epi16(k, a, _mm512_shrdv_epi16(b, b, cnt))
#define _mm512_maskz_rorv_vbmi2_epi16(k, a, cnt)	_mm512_maskz_shrdv_epi16(k, a, a, cnt)

/* Variable rotate left words by cnt bits, cnt mod 16 */
#define _mm_rolv_vbmi2_epi16(a, cnt)				_mm_shldv_epi16(a, a, cnt)
#define _mm_mask_rolv_vbmi2_epi16(a, k, b, cnt)		_mm_mask_blend_epi16(k, a, _mm_shldv_epi16(b, b, cnt))
#define _mm_maskz_rolv_vbmi2_epi16(k, a, cnt)		_mm_maskz_shldv_epi16(k, a, a, cnt)
#define _mm256_rolv_vbmi2_epi16(a, cnt)				_mm256_shldv_epi16(a, a, cnt)
#define _mm256_mask_rolv_vbmi2_epi16(a, k, b, cnt)	_mm256_mask_blend_epi16(k, a, _mm256_shldv_epi16(b, b, cnt))
#define _mm256_maskz_rolv_vbmi2_epi16(k, a, cnt)	_mm256_maskz_shldv_epi16(k, a, a, cnt)
#define _mm512_rolv_vbmi2_epi16(a, cnt)				_mm512_shldv_epi16(a, a, cnt)
#define _mm512_mask_rolv_vbmi2_epi16(a, k, b, cnt)	_mm512_mask_blend_epi16(k, a, _mm512_shldv_epi16(b, b, cnt))
#define _mm512_maskz_rolv_vbmi2_epi16(k, a, cnt)	_mm512_maskz_shldv_epi16(k, a, a, cnt)

#define _mm_swaplh_epi8(a)							_mm_shldi_epi16(a, a, 8)
#define _mm_swaphl_epi8(a)							_mm_shrdi_epi16(a, a, 8)
#define _mm256_swaplh_epi8(a)						_mm256_shldi_epi16(a, a, 8)
#define _mm256_swaphl_epi8(a)						_mm256_shrdi_epi16(a, a, 8)
#define _mm512_swaplh_epi8(a)						_mm512_shldi_epi16(a, a, 8)
#define _mm512_swaphl_epi8(a)						_mm512_shrdi_epi16(a, a, 8)

#define _shift_cnt1(cnt)							min(cnt + 8, 255)
#define _shift_cnt2(cnt)							(cnt)

#define _rotate_cnt1(cnt)							(cnt | 0x8)
#define _rotate_cnt2(cnt)							(cnt & ~0x8)

//a & ~b | c
#define _rotate_vcnt_r(size, cnt)					_##size##_ternarylogic_epi32(cnt, _##size##_set1_epi32(0x00080008), _##size##_set1_epi32(0x08000800), 0xba)
// a | b &  ~c
#define _rotate_vcnt_l(size, cnt)					_##size##_ternarylogic_epi32(cnt, _##size##_set1_epi32(0x00080008), _##size##_set1_epi32(0x08000800), 0x54)

/* Logical shift right bytes by imm bits, if imm > 7, result is 0 */

#define _mm_srli_vbmi2_epi8(a, cnt)					_mm_shldi_epi16(_mm_srli_epi16(a, _shift_cnt1(cnt)), _mm_srli_epi16(_mm_swaphl_epi8(a), _shift_cnt2(cnt)), 8)
#define _mm_mask_srli_vbmi2_epi8(a, k, b, cnt)		_mm_mask_mov_epi8(a, k, _mm_srli_vbmi2_epi8(b, cnt))
#define _mm_maskz_srli_vbmi2_epi8(k, a, cnt)		_mm_maskz_mov_epi8(k, _mm_srli_vbmi2_epi8(a, cnt))
#define _mm256_srli_vbmi2_epi8(a, cnt)				_mm256_shldi_epi16(_mm256_srli_epi16(a, _shift_cnt1(cnt)), _mm256_srli_epi16(_mm256_swaphl_epi8(a), _shift_cnt2(cnt)), 8)
#define _mm256_mask_srli_vbmi2_epi8(a, k, b, cnt)	_mm256_mask_mov_epi8(a, k, _mm256_srli_vbmi2_epi8(b, cnt))
#define _mm256_maskz_srli_vbmi2_epi8(k, a, cnt)		_mm256_maskz_mov_epi8(k, _mm256_srli_vbmi2_epi8(a, cnt))
#define _mm512_srli_vbmi2_epi8(a, cnt)				_mm512_shldi_epi16(_mm512_srli_epi16(a, _shift_cnt1(cnt)), _mm512_srli_epi16(_mm512_swaphl_epi8(a), _shift_cnt2(cnt)), 8)
#define _mm512_mask_srli_vbmi2_epi8(a, k, b, cnt)	_mm512_mask_mov_epi8(a, k, _mm512_srli_vbmi2_epi8(b, cnt))
#define _mm512_maskz_srli_vbmi2_epi8(k, a, cnt)		_mm512_maskz_mov_epi8(k, _mm512_srli_vbmi2_epi8(a, cnt))

/* Rotate right bytes by imm8 bits mod 8 */

#define _mm_ror_vbmi2_epi8(a, cnt)					_mm_shldi_epi16(_mm_shrdi_epi16(a, _mm_swaphl_epi8(a), _rotate_cnt1(cnt)), _mm_shrdi_epi16(_mm_swaphl_epi8(a), a, _rotate_cnt2(cnt)), 8)
#define _mm_mask_ror_vbmi2_epi8(a, k, b, cnt)		_mm_mask_mov_epi8(a, k, _mm_ror_vbmi2_epi8(b, cnt))
#define _mm_maskz_ror_vbmi2_epi8(k, a, cnt)			_mm_maskz_mov_epi8(k, _mm_ror_vbmi2_epi8(a, cnt))
#define _mm256_ror_vbmi2_epi8(a, cnt)				_mm256_shldi_epi16(_mm256_shrdi_epi16(a, _mm256_swaphl_epi8(a), _rotate_cnt1(cnt)), _mm256_shrdi_epi16(_mm256_swaphl_epi8(a), a, _rotate_cnt2(cnt)), 8)
#define _mm256_mask_ror_vbmi2_epi8(a, k, b, cnt)	_mm256_mask_mov_epi8(a, k, _mm256_ror_vbmi2_epi8(b, cnt))
#define _mm256_maskz_ror_vbmi2_epi8(k, a, cnt)		_mm256_maskz_mov_epi8(k, _mm256_ror_vbmi2_epi8(a, cnt))
#define _mm512_ror_vbmi2_epi8(a, cnt)				_mm512_shldi_epi16(_mm512_shrdi_epi16(a, _mm512_swaphl_epi8(a), _rotate_cnt1(cnt)), _mm512_shrdi_epi16(_mm512_swaphl_epi8(a), a, _rotate_cnt2(cnt)), 8)
#define _mm512_mask_ror_vbmi2_epi8(a, k, b, cnt)	_mm512_mask_mov_epi8(a, k, _mm512_ror_vbmi2_epi8(b, cnt))
#define _mm512_maskz_ror_vbmi2_epi8(k, a, cnt)		_mm512_maskz_mov_epi8(k, _mm512_ror_vbmi2_epi8(a, cnt))

/* Variable logical shift right bytes by cnt bits, if cnt > 7, result is 0 */

#define _mm_srlv_vbmi2_epi8(a, cnt)					_mm_shldi_epi16(_mm_srlv_epi16(a, _mm_adds_epu8(_mm_set1_epi32(0x00080008), _mm_srli_epi16(cnt, 8))), _mm_srlv_epi16(_mm_swaphl_epi8(a), _mm_and_si128(_mm_set1_epi32(0x00ff00ff), cnt)), 8)
#define _mm_mask_srlv_vbmi2_epi8(a, k, b, cnt)		_mm_mask_mov_epi8(a, k, _mm_srlv_vbmi2_epi8(b, cnt))
#define _mm_maskz_srlv_vbmi2_epi8(k, a, cnt)		_mm_maskz_mov_epi8(k, _mm_srlv_vbmi2_epi8(a, cnt))
#define _mm256_srlv_vbmi2_epi8(a, cnt)				_mm256_shldi_epi16(_mm256_srlv_epi16(a, _mm256_adds_epu8(_mm256_set1_epi32(0x00080008), _mm256_srli_epi16(cnt, 8))), _mm256_srlv_epi16(_mm256_swaphl_epi8(a), _mm256_and_si256(_mm256_set1_epi32(0x00ff00ff), cnt)), 8)
#define _mm256_mask_srlv_vbmi2_epi8(a, k, b, cnt)	_mm256_mask_mov_epi8(a, k, _mm256_srlv_vbmi2_epi8(b, cnt))
#define _mm256_maskz_srlv_vbmi2_epi8(k, a, cnt)		_mm256_maskz_mov_epi8(k, _mm256_srlv_vbmi2_epi8(a, cnt))
#define _mm512_srlv_vbmi2_epi8(a, cnt)				_mm512_shldi_epi16(_mm512_srlv_epi16(a, _mm512_adds_epu8(_mm512_set1_epi32(0x00080008), _mm512_srli_epi16(cnt, 8))), _mm512_srlv_epi16(_mm512_swaphl_epi8(a), _mm512_and_si512(_mm512_set1_epi32(0x00ff00ff), cnt)), 8)
#define _mm512_mask_srlv_vbmi2_epi8(a, k, b, cnt)	_mm512_mask_mov_epi8(a, k, _mm512_srlv_vbmi2_epi8(b, cnt))
#define _mm512_maskz_srlv_vbmi2_epi8(k, a, cnt)		_mm512_maskz_mov_epi8(k, _mm512_srlv_vbmi2_epi8(a, cnt))

/* Variable rotate right bytes by cnt bits, cnt mod 8 */

#define _mm_rorv_vbmi2_epi8(a, cnt)					_mm_shldi_epi16(_mm_shrdv_epi16(a, _mm_swaphl_epi8(a), _mm_alignr_epi8(_rotate_vcnt_r(mm, cnt), _rotate_vcnt_r(mm, cnt), 1)), _mm_shrdv_epi16(_mm_swaphl_epi8(a), a, _rotate_vcnt_r(mm, cnt)), 8)
#define _mm_mask_rorv_vbmi2_epi8(a, k, b, cnt)		_mm_mask_mov_epi8(a, k, _mm_rorv_vbmi2_epi8(b, cnt))
#define _mm_maskz_rorv_vbmi2_epi8(k, a, cnt)		_mm_maskz_mov_epi8(k, _mm_rorv_vbmi2_epi8(a, cnt))
#define _mm256_rorv_vbmi2_epi8(a, cnt)				_mm256_shldi_epi16(_mm256_shrdv_epi16(a, _mm256_swaphl_epi8(a), _mm256_alignr_epi8(_rotate_vcnt_r(mm256, cnt), _rotate_vcnt_r(mm256, cnt), 1)), _mm256_shrdv_epi16(_mm256_swaphl_epi8(a), a, _rotate_vcnt_r(mm256, cnt)), 8)
#define _mm256_mask_rorv_vbmi2_epi8(a, k, b, cnt)	_mm256_mask_mov_epi8(a, k, _mm256_rorv_vbmi2_epi8(b, cnt))
#define _mm256_maskz_rorv_vbmi2_epi8(k, a, cnt)		_mm256_maskz_mov_epi8(k, _mm256_rorv_vbmi2_epi8(a, cnt))
#define _mm512_rorv_vbmi2_epi8(a, cnt)				_mm512_shldi_epi16(_mm512_shrdv_epi16(a, _mm512_swaphl_epi8(a), _mm512_alignr_epi8(_rotate_vcnt_r(mm512, cnt), _rotate_vcnt_r(mm512, cnt), 1)), _mm512_shrdv_epi16(_mm512_swaphl_epi8(a), a, _rotate_vcnt_r(mm512, cnt)), 8)
#define _mm512_mask_rorv_vbmi2_epi8(a, k, b, cnt)	_mm512_mask_mov_epi8(a, k, _mm512_rorv_vbmi2_epi8(b, cnt))
#define _mm512_maskz_rorv_vbmi2_epi8(k, a, cnt)		_mm512_maskz_mov_epi8(k, _mm512_rorv_vbmi2_epi8(a, cnt))

/* Logical shift left bytes by imm bits, if imm > 7, result is 0 */

#define _mm_slli_vbmi2_epi8(a, cnt)					_mm_shldi_epi16(_mm_slli_epi16(_mm_swaplh_epi8(a), _shift_cnt2(cnt)), _mm_slli_epi16(a, _shift_cnt1(cnt)), 8)
#define _mm_mask_slli_vbmi2_epi8(a, k, b, cnt)		_mm_mask_mov_epi8(a, k, _mm_slli_vbmi2_epi8(b, cnt))
#define _mm_maskz_slli_vbmi2_epi8(k, a, cnt)		_mm_maskz_mov_epi8(k, _mm_slli_vbmi2_epi8(a, cnt))
#define _mm256_slli_vbmi2_epi8(a, cnt)				_mm256_shldi_epi16(_mm256_slli_epi16(_mm256_swaplh_epi8(a), _shift_cnt2(cnt)), _mm256_slli_epi16(a, _shift_cnt1(cnt)), 8)
#define _mm256_mask_slli_vbmi2_epi8(a, k, b, cnt)	_mm256_mask_mov_epi8(a, k, _mm256_slli_vbmi2_epi8(b, cnt))
#define _mm256_maskz_slli_vbmi2_epi8(k, a, cnt)		_mm256_maskz_mov_epi8(k, _mm256_slli_vbmi2_epi8(a, cnt))
#define _mm512_slli_vbmi2_epi8(a, cnt)				_mm512_shldi_epi16(_mm512_slli_epi16(_mm512_swaplh_epi8(a), _shift_cnt2(cnt)), _mm512_slli_epi16(a, _shift_cnt1(cnt)), 8)
#define _mm512_mask_slli_vbmi2_epi8(a, k, b, cnt)	_mm512_mask_mov_epi8(a, k, _mm512_slli_vbmi2_epi8(b, cnt))
#define _mm512_maskz_slli_vbmi2_epi8(k, a, cnt)		_mm512_maskz_mov_epi8(k, _mm512_slli_vbmi2_epi8(a, cnt))

/* Rotate left bytes by imm8 bits mod 8 */

#define _mm_rol_vbmi2_epi8(a, cnt)					_mm_shldi_epi16(_mm_shldi_epi16(_mm_swaplh_epi8(a), a, _rotate_cnt2(cnt)), _mm_shldi_epi16(a, _mm_swaplh_epi8(a), _rotate_cnt1(cnt)), 8)
#define _mm_mask_rol_vbmi2_epi8(a, k, b, cnt)		_mm_mask_mov_epi8(a, k, _mm_rol_vbmi2_epi8(b, cnt))
#define _mm_maskz_rol_vbmi2_epi8(k, a, cnt)			_mm_maskz_mov_epi8(k, _mm_rol_vbmi2_epi8(a, cnt))
#define _mm256_rol_vbmi2_epi8(a, cnt)				_mm256_shldi_epi16(_mm256_shldi_epi16(_mm256_swaplh_epi8(a), a, _rotate_cnt2(cnt)), _mm256_shldi_epi16(a, _mm256_swaplh_epi8(a), _rotate_cnt1(cnt)), 8)
#define _mm256_mask_rol_vbmi2_epi8(a, k, b, cnt)	_mm256_mask_mov_epi8(a, k, _mm256_rol_vbmi2_epi8(b, cnt))
#define _mm256_maskz_rol_vbmi2_epi8(k, a, cnt)		_mm256_maskz_mov_epi8(k, _mm256_rol_vbmi2_epi8(a, cnt))
#define _mm512_rol_vbmi2_epi8(a, cnt)				_mm512_shldi_epi16(_mm512_shldi_epi16(_mm512_swaplh_epi8(a), a, _rotate_cnt2(cnt)), _mm512_shldi_epi16(a, _mm512_swaplh_epi8(a), _rotate_cnt1(cnt)), 8)
#define _mm512_mask_rol_vbmi2_epi8(a, k, b, cnt)	_mm512_mask_mov_epi8(a, k, _mm512_rol_vbmi2_epi8(b, cnt))
#define _mm512_maskz_rol_vbmi2_epi8(k, a, cnt)		_mm512_maskz_mov_epi8(k, _mm512_rol_vbmi2_epi8(a, cnt))

/* Variable logical shift left bytes by cnt bits, if cnt > 7, result is 0 */

#define _mm_sllv_vbmi2_epi8(a, cnt)					_mm_shldi_epi16(_mm_sllv_epi16(_mm_swaplh_epi8(a), _mm_srli_epi16(cnt, 8)), _mm_sllv_epi16(a, _mm_adds_epu8(_mm_set1_epi32(0x00080008), _mm_and_si128(_mm_set1_epi32(0x00ff00ff), cnt))), 8)
#define _mm_mask_sllv_vbmi2_epi8(a, k, b, cnt)		_mm_mask_mov_epi8(a, k, _mm_sllv_vbmi2_epi8(b, cnt))
#define _mm_maskz_sllv_vbmi2_epi8(k, a, cnt)		_mm_maskz_mov_epi8(k, _mm_sllv_vbmi2_epi8(a, cnt))
#define _mm256_sllv_vbmi2_epi8(a, cnt)				_mm256_shldi_epi16(_mm256_sllv_epi16(_mm256_swaplh_epi8(a), _mm256_srli_epi16(cnt, 8)), _mm256_sllv_epi16(a, _mm256_adds_epu8(_mm256_set1_epi32(0x00080008), _mm256_and_si256(_mm256_set1_epi32(0x00ff00ff), cnt))), 8)
#define _mm256_mask_sllv_vbmi2_epi8(a, k, b, cnt)	_mm256_mask_mov_epi8(a, k, _mm256_sllv_vbmi2_epi8(b, cnt))
#define _mm256_maskz_sllv_vbmi2_epi8(k, a, cnt)		_mm256_maskz_mov_epi8(k, _mm256_sllv_vbmi2_epi8(a, cnt))
#define _mm512_sllv_vbmi2_epi8(a, cnt)				_mm512_shldi_epi16(_mm512_sllv_epi16(_mm512_swaplh_epi8(a), _mm512_srli_epi16(cnt, 8)), _mm512_sllv_epi16(a, _mm512_adds_epu8(_mm512_set1_epi32(0x00080008), _mm512_and_si512(_mm512_set1_epi32(0x00ff00ff), cnt))), 8)
#define _mm512_mask_sllv_vbmi2_epi8(a, k, b, cnt)	_mm512_mask_mov_epi8(a, k, _mm512_sllv_vbmi2_epi8(b, cnt))
#define _mm512_maskz_sllv_vbmi2_epi8(k, a, cnt)		_mm512_maskz_mov_epi8(k, _mm512_sllv_vbmi2_epi8(a, cnt))

/* Variable rotate left bytes by cnt bits, cnt mod 8 */

#define _mm_rolv_vbmi2_epi8(a, cnt)					_mm_shldi_epi16(_mm_shldv_epi16(_mm_swaphl_epi8(a), a, _mm_alignr_epi8(_rotate_vcnt_l(mm, cnt), _rotate_vcnt_l(mm, cnt), 1)), _mm_shldv_epi16(a, _mm_swaphl_epi8(a), _rotate_vcnt_l(mm, cnt)), 8)
#define _mm_mask_rolv_vbmi2_epi8(a, k, b, cnt)		_mm_mask_mov_epi8(a, k, _mm_rolv_vbmi2_epi8(b, cnt))
#define _mm_maskz_rolv_vbmi2_epi8(k, a, cnt)		_mm_maskz_mov_epi8(k, _mm_rolv_vbmi2_epi8(a, cnt))
#define _mm256_rolv_vbmi2_epi8(a, cnt)				_mm256_shldi_epi16(_mm256_shldv_epi16(_mm256_swaplh_epi8(a), a, _mm256_alignr_epi8(_rotate_vcnt_l(mm256, cnt), _rotate_vcnt_l(mm256, cnt), 1)), _mm256_shldv_epi16(a, _mm256_swaplh_epi8(a), _rotate_vcnt_l(mm256, cnt)), 8)
#define _mm256_mask_rolv_vbmi2_epi8(a, k, b, cnt)	_mm256_mask_mov_epi8(a, k, _mm256_rolv_vbmi2_epi8(b, cnt))
#define _mm256_maskz_rolv_vbmi2_epi8(k, a, cnt)		_mm256_maskz_mov_epi8(k, _mm256_rolv_vbmi2_epi8(a, cnt))
#define _mm512_rolv_vbmi2_epi8(a, cnt)				_mm512_shldi_epi16(_mm512_shldv_epi16(_mm512_swaplh_epi8(a), a, _mm512_alignr_epi8(_rotate_vcnt_l(mm512, cnt), _rotate_vcnt_l(mm512, cnt), 1)), _mm512_shldv_epi16(a, _mm512_swaplh_epi8(a), _rotate_vcnt_l(mm512, cnt)), 8)
#define _mm512_mask_rolv_vbmi2_epi8(a, k, b, cnt)	_mm512_mask_mov_epi8(a, k, _mm512_rolv_vbmi2_epi8(b, cnt))
#define _mm512_maskz_rolv_vbmi2_epi8(k, a, cnt)		_mm512_maskz_mov_epi8(k, _mm512_rolv_vbmi2_epi8(a, cnt))

/* Arithmetcial shift right bytes by imm bits, if imm > 7, result is filled with MSB */

#define _mm_srai_vbmi2_epi8(a, cnt)					_mm_shldi_epi16(_mm_srai_epi16(a, _shift_cnt1(cnt)), _mm_srai_epi16(_mm_swaphl_epi8(a), _shift_cnt2(cnt)), 8)
#define _mm_mask_srai_vbmi2_epi8(a, k, b, cnt)		_mm_mask_mov_epi8(a, k, _mm_srai_vbmi2_epi8(b, cnt))
#define _mm_maskz_srai_vbmi2_epi8(k, a, cnt)		_mm_maskz_mov_epi8(k, _mm_srai_vbmi2_epi8(a, cnt))
#define _mm256_srai_vbmi2_epi8(a, cnt)				_mm256_shldi_epi16(_mm256_srai_epi16(a, _shift_cnt1(cnt)), _mm256_srai_epi16(_mm256_swaphl_epi8(a), _shift_cnt2(cnt)), 8)
#define _mm256_mask_srai_vbmi2_epi8(a, k, b, cnt)	_mm256_mask_mov_epi8(a, k, _mm256_srai_vbmi2_epi8(b, cnt))
#define _mm256_maskz_srai_vbmi2_epi8(k, a, cnt)		_mm256_maskz_mov_epi8(k, _mm256_srai_vbmi2_epi8(a, cnt))
#define _mm512_srai_vbmi2_epi8(a, cnt)				_mm512_shldi_epi16(_mm512_srai_epi16(a, _shift_cnt1(cnt)), _mm512_srai_epi16(_mm512_swaphl_epi8(a), _shift_cnt2(cnt)), 8)
#define _mm512_mask_srai_vbmi2_epi8(a, k, b, cnt)	_mm512_mask_mov_epi8(a, k, _mm512_srai_vbmi2_epi8(b, cnt))
#define _mm512_maskz_srai_vbmi2_epi8(k, a, cnt)		_mm512_maskz_mov_epi8(k, _mm512_srai_vbmi2_epi8(a, cnt))

/* Variable arithmetical shift right bytes by cnt bits, if cnt > 7, result is filled with MSB */

#define _mm_srav_vbmi2_epi8(a, cnt)					_mm_shldi_epi16(_mm_srav_epi16(a, _mm_adds_epu8(_mm_srli_epi16(cnt, 8), _mm_set1_epi32(0x00080008))), _mm_srav_epi16(_mm_swaphl_epi8(a), _mm_and_si128(_mm_set1_epi32(0x00ff00ff), cnt)), 8)
#define _mm_mask_srav_vbmi2_epi8(a, k, b, cnt)		_mm_mask_mov_epi8(a, k, _mm_srav_vbmi2_epi8(b, cnt))
#define _mm_maskz_srav_vbmi2_epi8(k, a, cnt)		_mm_maskz_mov_epi8(k, _mm_srav_vbmi2_epi8(a, cnt))
#define _mm256_srav_vbmi2_epi8(a, cnt)				_mm256_shldi_epi16(_mm256_srav_epi16(a, _mm256_adds_epu8(_mm256_srli_epi16(cnt, 8), _mm256_set1_epi32(0x00080008))), _mm256_srav_epi16(_mm256_swaphl_epi8(a), _mm256_and_si256(_mm256_set1_epi32(0x00ff00ff), cnt)), 8)
#define _mm256_mask_srav_vbmi2_epi8(a, k, b, cnt)	_mm256_mask_mov_epi8(a, k, _mm256_srav_vbmi2_epi8(b, cnt))
#define _mm256_maskz_srav_vbmi2_epi8(k, a, cnt)		_mm256_maskz_mov_epi8(k, _mm256_srav_vbmi2_epi8(a, cnt))
#define _mm512_srav_vbmi2_epi8(a, cnt)				_mm512_shldi_epi16(_mm512_srav_epi16(a, _mm512_adds_epu8(_mm512_srli_epi16(cnt, 8), _mm512_set1_epi32(0x00080008))), _mm512_srav_epi16(_mm512_swaphl_epi8(a), _mm512_and_si512(_mm512_set1_epi32(0x00ff00ff), cnt)), 8)
#define _mm512_mask_srav_vbmi2_epi8(a, k, b, cnt)	_mm512_mask_mov_epi8(a, k, _mm512_srav_vbmi2_epi8(b, cnt))
#define _mm512_maskz_srav_vbmi2_epi8(k, a, cnt)		_mm512_maskz_mov_epi8(k, _mm512_srav_vbmi2_epi8(a, cnt))


#define _GFNI_DEMO_IDENT		0x0102040810204080
#define _GFNI_DEMO_REVBIT		0x8040201008040201
#define _GFNI_DEMO_BCST			0x0101010101010101
#define _GFNI_DEMO_PREFXOR		0x0103070f1f3f7fff
#define _GFNI_DEMO_TZCNT		0xaaccf0ff00000000
#define _GFNI_DEMO_MULMASK		0x0103070f1f3f7fff
#define _GFNI_DEMO_MULBIT		0x8040201008040201

#define _GFNI_DEMO_SLL(i)		((0x0102040810204080 >> (i)) & (0x0101010101010101ULL * (0xff >> (i))))
#define _GFNI_DEMO_SRL(i)		((0x0102040810204080 << (i)) & (0x0101010101010101ULL * ((0xff << (i)) & 0xff)))
#define _GFNI_DEMO_SLA(i)		(_GFNI_DEMO_SLL(i) | ((0x0101010101010101ULL << (64 - (8 * i))) & (0ULL - (i > 0))))
#define _GFNI_DEMO_SRA(i)		(_GFNI_DEMO_SRL(i) | ((0x8080808080808080ULL >> (64 - (8 * i))) & (0ULL - (i > 0))))
#define _GFNI_DEMO_ROL(i)		(_GFNI_DEMO_SRL(8 - i) | _GFNI_DEMO_SLL(i))
#define _GFNI_DEMO_ROR(i)		(_GFNI_DEMO_SLL(8 - i) | _GFNI_DEMO_SRL(i))

/* Logical shift right by imm, if imm > 7, result is 0 */

#define _mm_srli_gfni_epi8(a, cnt)					_mm_gf2p8affine_epi64_epi8(a, _mm_set1_epi64x(_GFNI_DEMO_SRL(cnt)), 0)
#define _mm_mask_srli_gfni_epi8(s, k, a, cnt)		_mm_mask_gf2p8affine_epi64_epi8(s, k, a, _mm_set1_epi64x(_GFNI_DEMO_SRL(cnt)), 0)
#define _mm_maskz_srli_gfni_epi8(k, a, cnt)			_mm_maskz_gf2p8affine_epi64_epi8(k, a, _mm_set1_epi64x(_GFNI_DEMO_SRL(cnt)), 0)

#define _mm256_srli_gfni_epi8(a, cnt)				_mm256_gf2p8affine_epi64_epi8(a, _mm256_set1_epi64x(_GFNI_DEMO_SRL(cnt)), 0)
#define _mm256_mask_srli_gfni_epi8(s, k, a, cnt)	_mm256_mask_gf2p8affine_epi64_epi8(s, k, a, _mm256_set1_epi64x(_GFNI_DEMO_SRL(cnt)), 0)
#define _mm256_maskz_srli_gfni_epi8(k, a, cnt)		_mm256_maskz_gf2p8affine_epi64_epi8(k, a, _mm256_set1_epi64x(_GFNI_DEMO_SRL(cnt)), 0)

#define _mm512_srli_gfni_epi8(a, cnt)				_mm512_gf2p8affine_epi64_epi8(a, _mm512_set1_epi64(_GFNI_DEMO_SRL(cnt)), 0)
#define _mm512_mask_srli_gfni_epi8(s, k, a, cnt)	_mm512_mask_gf2p8affine_epi64_epi8(s, k, a, _mm512_set1_epi64(_GFNI_DEMO_SRL(cnt)), 0)
#define _mm512_maskz_srli_gfni_epi8(k, a, cnt)		_mm512_maskz_gf2p8affine_epi64_epi8(k, a, _mm512_set1_epi64(_GFNI_DEMO_SRL(cnt)), 0)

/* Logical shift right by b[2:0], if b[63:0] > 7, result is 0 */

#define _mm_srl_gfni_epi8(a, b)						_mm_gf2p8affine_epi64_epi8(a, _mm_set1_epi64x(_GFNI_DEMO_SRL(_mm_cvtsi128_si32(b))), 0)
#define _mm_mask_srl_gfni_epi8(s, k, a, b)			_mm_mask_gf2p8affine_epi64_epi8(s, k, a, _mm_set1_epi64x(_GFNI_DEMO_SRL(_mm_cvtsi128_si32(b))), 0)
#define _mm_maskz_srl_gfni_epi8(k, a, b)			_mm_maskz_gf2p8affine_epi64_epi8(k, a, _mm_set1_epi64x(_GFNI_DEMO_SRL(_mm_cvtsi128_si32(b))), 0)

#define _mm256_srl_gfni_epi8(a, b)					_mm256_gf2p8affine_epi64_epi8(a, _mm256_set1_epi64x(_GFNI_DEMO_SRL(_mm_cvtsi128_si32(b))), 0)
#define _mm256_mask_srl_gfni_epi8(s, k, a, b)		_mm256_mask_gf2p8affine_epi64_epi8(s, k, a, _mm256_set1_epi64x(_GFNI_DEMO_SRL(_mm_cvtsi128_si32(b))), 0)
#define _mm256_maskz_srl_gfni_epi8(k, a, b)			_mm256_maskz_gf2p8affine_epi64_epi8(k, a, _mm256_set1_epi64x(_GFNI_DEMO_SRL(_mm_cvtsi128_si32(b))), 0)

#define _mm512_srl_gfni_epi8(a, b)					_mm512_gf2p8affine_epi64_epi8(a, _mm512_set1_epi64(_GFNI_DEMO_SRL(_mm_cvtsi128_si32(b))), 0)
#define _mm512_mask_srl_gfni_epi8(s, k, a, b)		_mm512_mask_gf2p8affine_epi64_epi8(s, k, a, _mm512_set1_epi64(_GFNI_DEMO_SRL(_mm_cvtsi128_si32(b))), 0)
#define _mm512_maskz_srl_gfni_epi8(k, a, b)			_mm512_maskz_gf2p8affine_epi64_epi8(k, a, _mm512_set1_epi64(_GFNI_DEMO_SRL(_mm_cvtsi128_si32(b))), 0)

/* Arithmetical shift right by imm, if imm > 7, result is filled with MSB */

#define _mm_srai_gfni_epi8(a, cnt)					_mm_gf2p8affine_epi64_epi8(a, _mm_set1_epi64x(_GFNI_DEMO_SRA(cnt)), 0)
#define _mm_mask_srai_gfni_epi8(s, k, a, cnt)		_mm_mask_gf2p8affine_epi64_epi8(s, k, a, _mm_set1_epi64x(_GFNI_DEMO_SRA(cnt)), 0)
#define _mm_maskz_srai_gfni_epi8(k, a, cnt)			_mm_maskz_gf2p8affine_epi64_epi8(k, a, _mm_set1_epi64x(_GFNI_DEMO_SRA(cnt)), 0)

#define _mm256_srai_gfni_epi8(a, cnt)				_mm256_gf2p8affine_epi64_epi8(a, _mm256_set1_epi64x(_GFNI_DEMO_SRA(cnt)), 0)
#define _mm256_mask_srai_gfni_epi8(s, k, a, cnt)	_mm256_mask_gf2p8affine_epi64_epi8(s, k, a, _mm256_set1_epi64x(_GFNI_DEMO_SRA(cnt)), 0)
#define _mm256_maskz_srai_gfni_epi8(k, a, cnt)		_mm256_maskz_gf2p8affine_epi64_epi8(k, a, _mm256_set1_epi64x(_GFNI_DEMO_SRA(cnt)), 0)

#define _mm512_srai_gfni_epi8(a, cnt)				_mm512_gf2p8affine_epi64_epi8(a, _mm512_set1_epi64(_GFNI_DEMO_SRA(cnt)), 0)
#define _mm512_mask_srai_gfni_epi8(s, k, a, cnt)	_mm512_mask_gf2p8affine_epi64_epi8(s, k, a, _mm512_set1_epi64(_GFNI_DEMO_SRA(cnt)), 0)
#define _mm512_maskz_srai_gfni_epi8(k, a, cnt)		_mm512_maskz_gf2p8affine_epi64_epi8(k, a, _mm512_set1_epi64(_GFNI_DEMO_SRA(cnt)), 0)

/* Arithmetical shift right by b[2:0], if b[63:0] > 7, result is filled with MSB */

#define _mm_sra_gfni_epi8(a, b)						_mm_gf2p8affine_epi64_epi8(a, _mm_set1_epi64x(_GFNI_DEMO_SRA(_mm_cvtsi128_si32(b))), 0)
#define _mm_mask_sra_gfni_epi8(s, k, a, b)			_mm_mask_gf2p8affine_epi64_epi8(s, k, a, _mm_set1_epi64x(_GFNI_DEMO_SRA(_mm_cvtsi128_si32(b))), 0)
#define _mm_maskz_sra_gfni_epi8(k, a, b)			_mm_maskz_gf2p8affine_epi64_epi8(k, a, _mm_set1_epi64x(_GFNI_DEMO_SRA(_mm_cvtsi128_si32(b))), 0)

#define _mm256_sra_gfni_epi8(a, b)					_mm256_gf2p8affine_epi64_epi8(a, _mm256_set1_epi64x(_GFNI_DEMO_SRA(_mm_cvtsi128_si32(b))), 0)
#define _mm256_mask_sra_gfni_epi8(s, k, a, b)		_mm256_mask_gf2p8affine_epi64_epi8(s, k, a, _mm256_set1_epi64x(_GFNI_DEMO_SRA(_mm_cvtsi128_si32(b))), 0)
#define _mm256_maskz_sra_gfni_epi8(k, a, b)			_mm256_maskz_gf2p8affine_epi64_epi8(k, a, _mm256_set1_epi64x(_GFNI_DEMO_SRA(_mm_cvtsi128_si32(b))), 0)

#define _mm512_sra_gfni_epi8(a, b)					_mm512_gf2p8affine_epi64_epi8(a, _mm512_set1_epi64(_GFNI_DEMO_SRA(_mm_cvtsi128_si32(b))), 0)
#define _mm512_mask_sra_gfni_epi8(s, k, a, b)		_mm512_mask_gf2p8affine_epi64_epi8(s, k, a, _mm512_set1_epi64(_GFNI_DEMO_SRA(_mm_cvtsi128_si32(b))), 0)
#define _mm512_maskz_sra_gfni_epi8(k, a, b)			_mm512_maskz_gf2p8affine_epi64_epi8(k, a, _mm512_set1_epi64(_GFNI_DEMO_SRA(_mm_cvtsi128_si32(b))), 0)

/* Logical shift left by imm, if imm > 7, result is 0 */

#define _mm_slli_gfni_epi8(a, cnt)					_mm_gf2p8affine_epi64_epi8(a, _mm_set1_epi64x(_GFNI_DEMO_SLL(cnt)), 0)
#define _mm_mask_slli_gfni_epi8(s, k, a, cnt)		_mm_mask_gf2p8affine_epi64_epi8(s, k, a, _mm_set1_epi64x(_GFNI_DEMO_SLL(cnt)), 0)
#define _mm_maskz_slli_gfni_epi8(k, a, cnt)			_mm_maskz_gf2p8affine_epi64_epi8(k, a, _mm_set1_epi64x(_GFNI_DEMO_SLL(cnt)), 0)

#define _mm256_slli_gfni_epi8(a, cnt)				_mm256_gf2p8affine_epi64_epi8(a, _mm256_set1_epi64x(_GFNI_DEMO_SLL(cnt)), 0)
#define _mm256_mask_slli_gfni_epi8(s, k, a, cnt)	_mm256_mask_gf2p8affine_epi64_epi8(s, k, a, _mm256_set1_epi64x(_GFNI_DEMO_SLL(cnt)), 0)
#define _mm256_maskz_slli_gfni_epi8(k, a, cnt)		_mm256_maskz_gf2p8affine_epi64_epi8(k, a, _mm256_set1_epi64x(_GFNI_DEMO_SLL(cnt)), 0)

#define _mm512_slli_gfni_epi8(a, cnt)				_mm512_gf2p8affine_epi64_epi8(a, _mm512_set1_epi64(_GFNI_DEMO_SLL(cnt)), 0)
#define _mm512_mask_slli_gfni_epi8(s, k, a, cnt)	_mm512_mask_gf2p8affine_epi64_epi8(s, k, a, _mm512_set1_epi64(_GFNI_DEMO_SLL(cnt)), 0)
#define _mm512_maskz_slli_gfni_epi8(k, a, cnt)		_mm512_maskz_gf2p8affine_epi64_epi8(k, a, _mm512_set1_epi64(_GFNI_DEMO_SLL(cnt)), 0)

/* Logical shift left by b[2:0], if b[63:0] > 7, result is 0 */

#define _mm_sll_gfni_epi8(a, b)						_mm_gf2p8affine_epi64_epi8(a, _mm_set1_epi64x(_GFNI_DEMO_SLL(_mm_cvtsi128_si32(b))), 0)
#define _mm_mask_sll_gfni_epi8(s, k, a, b)			_mm_mask_gf2p8affine_epi64_epi8(s, k, a, _mm_set1_epi64x(_GFNI_DEMO_SLL(_mm_cvtsi128_si32(b))), 0)
#define _mm_maskz_sll_gfni_epi8(k, a, b)			_mm_maskz_gf2p8affine_epi64_epi8(k, a, _mm_set1_epi64x(_GFNI_DEMO_SLL(_mm_cvtsi128_si32(b))), 0)

#define _mm256_sll_gfni_epi8(a, b)					_mm256_gf2p8affine_epi64_epi8(a, _mm256_set1_epi64x(_GFNI_DEMO_SLL(_mm_cvtsi128_si32(b))), 0)
#define _mm256_mask_sll_gfni_epi8(s, k, a, b)		_mm256_mask_gf2p8affine_epi64_epi8(s, k, a, _mm256_set1_epi64x(_GFNI_DEMO_SLL(_mm_cvtsi128_si32(b))), 0)
#define _mm256_maskz_sll_gfni_epi8(k, a, b)			_mm256_maskz_gf2p8affine_epi64_epi8(k, a, _mm256_set1_epi64x(_GFNI_DEMO_SLL(_mm_cvtsi128_si32(b))), 0)

#define _mm512_sll_gfni_epi8(a, b)					_mm512_gf2p8affine_epi64_epi8(a, _mm512_set1_epi64(_GFNI_DEMO_SLL(_mm_cvtsi128_si32(b))), 0)
#define _mm512_mask_sll_gfni_epi8(s, k, a, b)		_mm512_mask_gf2p8affine_epi64_epi8(s, k, a, _mm512_set1_epi64(_GFNI_DEMO_SLL(_mm_cvtsi128_si32(b))), 0)
#define _mm512_maskz_sll_gfni_epi8(k, a, b)			_mm512_maskz_gf2p8affine_epi64_epi8(k, a, _mm512_set1_epi64(_GFNI_DEMO_SLL(_mm_cvtsi128_si32(b))), 0)

/* Arithmetical shift left by imm, if imm > 7, result is filled with LSB */

#define _mm_slai_gfni_epi8(a, cnt)					_mm_gf2p8affine_epi64_epi8(a, _mm_set1_epi64x(_GFNI_DEMO_SLA(cnt)), 0)
#define _mm_mask_slai_gfni_epi8(s, k, a, cnt)		_mm_mask_gf2p8affine_epi64_epi8(s, k, a, _mm_set1_epi64x(_GFNI_DEMO_SLA(cnt)), 0)
#define _mm_maskz_slai_gfni_epi8(k, a, cnt)			_mm_maskz_gf2p8affine_epi64_epi8(k, a, _mm_set1_epi64x(_GFNI_DEMO_SLA(cnt)), 0)

#define _mm256_slai_gfni_epi8(a, cnt)				_mm256_gf2p8affine_epi64_epi8(a, _mm256_set1_epi64x(_GFNI_DEMO_SLA(cnt)), 0)
#define _mm256_mask_slai_gfni_epi8(s, k, a, cnt)	_mm256_mask_gf2p8affine_epi64_epi8(s, k, a, _mm256_set1_epi64x(_GFNI_DEMO_SLA(cnt)), 0)
#define _mm256_maskz_slai_gfni_epi8(k, a, cnt)		_mm256_maskz_gf2p8affine_epi64_epi8(k, a, _mm256_set1_epi64x(_GFNI_DEMO_SLA(cnt)), 0)

#define _mm512_slai_gfni_epi8(a, cnt)				_mm512_gf2p8affine_epi64_epi8(a, _mm512_set1_epi64(_GFNI_DEMO_SLA(cnt)), 0)
#define _mm512_mask_slai_gfni_epi8(s, k, a, cnt)	_mm512_mask_gf2p8affine_epi64_epi8(s, k, a, _mm512_set1_epi64(_GFNI_DEMO_SLA(cnt)), 0)
#define _mm512_maskz_slai_gfni_epi8(k, a, cnt)		_mm512_maskz_gf2p8affine_epi64_epi8(k, a, _mm512_set1_epi64(_GFNI_DEMO_SLA(cnt)), 0)

/* Arithmetical shift left by b[2:0], if b[63:0] > 7, result is filled with LSB */

#define _mm_sla_gfni_epi8(a, b)						_mm_gf2p8affine_epi64_epi8(a, _mm_set1_epi64x(_GFNI_DEMO_SLA(_mm_cvtsi128_si32(b))), 0)
#define _mm_mask_sla_gfni_epi8(s, k, a, b)			_mm_mask_gf2p8affine_epi64_epi8(s, k, a, _mm_set1_epi64x(_GFNI_DEMO_SLA(_mm_cvtsi128_si32(b))), 0)
#define _mm_maskz_sla_gfni_epi8(k, a, b)			_mm_maskz_gf2p8affine_epi64_epi8(k, a, _mm_set1_epi64x(_GFNI_DEMO_SLA(_mm_cvtsi128_si32(b))), 0)

#define _mm256_sla_gfni_epi8(a, b)					_mm256_gf2p8affine_epi64_epi8(a, _mm256_set1_epi64x(_GFNI_DEMO_SLA(_mm_cvtsi128_si32(b))), 0)
#define _mm256_mask_sla_gfni_epi8(s, k, a, b)		_mm256_mask_gf2p8affine_epi64_epi8(s, k, a, _mm256_set1_epi64x(_GFNI_DEMO_SLA(_mm_cvtsi128_si32(b))), 0)
#define _mm256_maskz_sla_gfni_epi8(k, a, b)			_mm256_maskz_gf2p8affine_epi64_epi8(k, a, _mm256_set1_epi64x(_GFNI_DEMO_SLA(_mm_cvtsi128_si32(b))), 0)

#define _mm512_sla_gfni_epi8(a, b)					_mm512_gf2p8affine_epi64_epi8(a, _mm512_set1_epi64(_GFNI_DEMO_SLA(_mm_cvtsi128_si32(b))), 0)
#define _mm512_mask_sla_gfni_epi8(s, k, a, b)		_mm512_mask_gf2p8affine_epi64_epi8(s, k, a, _mm512_set1_epi64(_GFNI_DEMO_SLA(_mm_cvtsi128_si32(b))), 0)
#define _mm512_maskz_sla_gfni_epi8(k, a, b)			_mm512_maskz_gf2p8affine_epi64_epi8(k, a, _mm512_set1_epi64(_GFNI_DEMO_SLA(_mm_cvtsi128_si32(b))), 0)

/* Rotate right by imm8 mod 8 */
#define _mm_ror_gfni_epi8(a, cnt)					_mm_gf2p8affine_epi64_epi8(a, _mm_set1_epi64x(_GFNI_DEMO_ROR(cnt)), 0)
#define _mm_mask_ror_gfni_epi8(s, k, a, cnt)		_mm_mask_gf2p8affine_epi64_epi8(s, k, a, _mm_set1_epi64x(_GFNI_DEMO_ROR(cnt)), 0)
#define _mm_maskz_ror_gfni_epi8(k, a, cnt)			_mm_maskz_gf2p8affine_epi64_epi8(k, a, _mm_set1_epi64x(_GFNI_DEMO_ROR(cnt)), 0)

#define _mm256_ror_gfni_epi8(a, cnt)				_mm256_gf2p8affine_epi64_epi8(a, _mm256_set1_epi64x(_GFNI_DEMO_ROR(cnt)), 0)
#define _mm256_mask_ror_gfni_epi8(s, k, a, cnt)		_mm256_mask_gf2p8affine_epi64_epi8(s, k, a, _mm256_set1_epi64x(_GFNI_DEMO_ROR(cnt)), 0)
#define _mm256_maskz_ror_gfni_epi8(k, a, cnt)		_mm256_maskz_gf2p8affine_epi64_epi8(k, a, _mm256_set1_epi64x(_GFNI_DEMO_ROR(cnt)), 0)

#define _mm512_ror_gfni_epi8(a, cnt)				_mm512_gf2p8affine_epi64_epi8(a, _mm512_set1_epi64(_GFNI_DEMO_ROR(cnt)), 0)
#define _mm512_mask_ror_gfni_epi8(s, k, a, cnt)		_mm512_mask_gf2p8affine_epi64_epi8(s, k, a, _mm512_set1_epi64(_GFNI_DEMO_ROR(cnt)), 0)
#define _mm512_maskz_ror_gfni_epi8(k, a, cnt)		_mm512_maskz_gf2p8affine_epi64_epi8(k, a, _mm512_set1_epi64(_GFNI_DEMO_ROR(cnt)), 0)

/* Rotate left by imm8 mod 8 */
#define _mm_rol_gfni_epi8(a, cnt)					_mm_gf2p8affine_epi64_epi8(a, _mm_set1_epi64x(_GFNI_DEMO_ROL(cnt)), 0)
#define _mm_mask_rol_gfni_epi8(s, k, a, cnt)		_mm_mask_gf2p8affine_epi64_epi8(s, k, a, _mm_set1_epi64x(_GFNI_DEMO_ROL(cnt)), 0)
#define _mm_maskz_rol_gfni_epi8(k, a, cnt)			_mm_maskz_gf2p8affine_epi64_epi8(k, a, _mm_set1_epi64x(_GFNI_DEMO_ROL(cnt)), 0)

#define _mm256_rol_gfni_epi8(a, cnt)				_mm256_gf2p8affine_epi64_epi8(a, _mm256_set1_epi64x(_GFNI_DEMO_ROL(cnt)), 0)
#define _mm256_mask_rol_gfni_epi8(s, k, a, cnt)		_mm256_mask_gf2p8affine_epi64_epi8(s, k, a, _mm256_set1_epi64x(_GFNI_DEMO_ROL(cnt)), 0)
#define _mm256_maskz_rol_gfni_epi8(k, a, cnt)		_mm256_maskz_gf2p8affine_epi64_epi8(k, a, _mm256_set1_epi64x(_GFNI_DEMO_ROL(cnt)), 0)

#define _mm512_rol_gfni_epi8(a, cnt)				_mm512_gf2p8affine_epi64_epi8(a, _mm512_set1_epi64(_GFNI_DEMO_ROL(cnt)), 0)
#define _mm512_mask_rol_gfni_epi8(s, k, a, cnt)		_mm512_mask_gf2p8affine_epi64_epi8(s, k, a, _mm512_set1_epi64(_GFNI_DEMO_ROL(cnt)), 0)
#define _mm512_maskz_rol_gfni_epi8(k, a, cnt)		_mm512_maskz_gf2p8affine_epi64_epi8(k, a, _mm512_set1_epi64(_GFNI_DEMO_ROL(cnt)), 0)

/* Variable logical shift left bytes by cnt bits, if cnt > 7, result is 0 */
#define _mm_sllv_gfni_epi8(a, cnt)					_mm_gf2p8mul_epi8(_mm_and_si128(a, _mm_shuffle_epi8(_mm_set_epi64x(0, _GFNI_DEMO_MULMASK), cnt)), _mm_shuffle_epi8(_mm_set_epi64x(0, _GFNI_DEMO_MULBIT), cnt))
#define _mm_mask_sllv_gfni_epi8(a, k, b, cnt)		_mm_mask_gf2p8mul_epi8(a, k, _mm_and_si128(b, _mm_shuffle_epi8(_mm_set_epi64x(0, _GFNI_DEMO_MULMASK), cnt)), _mm_shuffle_epi8(_mm_set_epi64x(0, _GFNI_DEMO_MULBIT), cnt))
#define _mm_maskz_sllv_gfni_epi8(k, a, cnt)			_mm_maskz_gf2p8mul_epi8(k, _mm_and_si128(a, _mm_shuffle_epi8(_mm_set_epi64x(0, _GFNI_DEMO_MULMASK), cnt)), _mm_shuffle_epi8(_mm_set_epi64x(0, _GFNI_DEMO_MULBIT), cnt))

#define _mm256_sllv_gfni_epi8(a, cnt)				_mm256_gf2p8mul_epi8(_mm256_and_si256(a, _mm256_shuffle_epi8(_mm256_set_epi64x(0, _GFNI_DEMO_MULMASK, 0, _GFNI_DEMO_MULMASK), cnt)), _mm256_shuffle_epi8(_mm256_set_epi64x(0, _GFNI_DEMO_MULBIT, 0, _GFNI_DEMO_MULBIT), cnt))
#define _mm256_mask_sllv_gfni_epi8(a, k, b, cnt)	_mm256_mask_gf2p8mul_epi8(a, k, _mm256_and_si256(b, _mm256_shuffle_epi8(_mm256_set_epi64x(0, _GFNI_DEMO_MULMASK, 0, _GFNI_DEMO_MULMASK), cnt)), _mm256_shuffle_epi8(_mm256_set_epi64x(0, _GFNI_DEMO_MULBIT, 0, _GFNI_DEMO_MULBIT), cnt))
#define _mm256_maskz_sllv_gfni_epi8(k, a, cnt)		_mm256_maskz_gf2p8mul_epi8(k, _mm256_and_si256(a, _mm256_shuffle_epi8(_mm256_set_epi64x(0, _GFNI_DEMO_MULMASK, 0, _GFNI_DEMO_MULMASK), cnt)), _mm256_shuffle_epi8(_mm256_set_epi64x(0, _GFNI_DEMO_MULBIT, 0, _GFNI_DEMO_MULBIT), cnt))

#define _mm512_sllv_gfni_epi8(a, cnt)				_mm512_gf2p8mul_epi8(_mm512_and_si512(a, _mm512_shuffle_epi8(_mm512_set_epi64(0, _GFNI_DEMO_MULMASK, 0, _GFNI_DEMO_MULMASK, 0, _GFNI_DEMO_MULMASK, 0, _GFNI_DEMO_MULMASK), cnt)), _mm512_shuffle_epi8(_mm512_set_epi64(0, _GFNI_DEMO_MULBIT, 0, _GFNI_DEMO_MULBIT, 0, _GFNI_DEMO_MULBIT, 0, _GFNI_DEMO_MULBIT), cnt))
#define _mm512_mask_sllv_gfni_epi8(a, k, b, cnt)	_mm512_mask_gf2p8mul_epi8(a, k, _mm512_and_si512(b, _mm512_shuffle_epi8(_mm512_set_epi64(0, _GFNI_DEMO_MULMASK, 0, _GFNI_DEMO_MULMASK, 0, _GFNI_DEMO_MULMASK, 0, _GFNI_DEMO_MULMASK), cnt)), _mm512_shuffle_epi8(_mm512_set_epi64(0, _GFNI_DEMO_MULBIT, 0, _GFNI_DEMO_MULBIT, 0, _GFNI_DEMO_MULBIT, 0, _GFNI_DEMO_MULBIT), cnt))
#define _mm512_maskz_sllv_gfni_epi8(k, a, cnt)		_mm512_maskz_gf2p8mul_epi8(k, _mm512_and_si512(a, _mm512_shuffle_epi8(_mm512_set_epi64(0, _GFNI_DEMO_MULMASK, 0, _GFNI_DEMO_MULMASK, 0, _GFNI_DEMO_MULMASK, 0, _GFNI_DEMO_MULMASK), cnt)), _mm512_shuffle_epi8(_mm512_set_epi64(0, _GFNI_DEMO_MULBIT, 0, _GFNI_DEMO_MULBIT, 0, _GFNI_DEMO_MULBIT, 0, _GFNI_DEMO_MULBIT), cnt))

/* Variable logical shift right bytes by cnt bits, if cnt > 7, result is 0 */
#define _mm_srlv_gfni_epi8(a, cnt)					_mm_revbit_epi8(_mm_sllv_gfni_epi8(_mm_revbit_epi8(a), cnt))
#define _mm_mask_srlv_gfni_epi8(a, k, b, cnt)		_mm_mask_revbit_epi8(a, k, _mm_sllv_gfni_epi8(_mm_revbit_epi8(b), cnt))
#define _mm_maskz_srlv_gfni_epi8(k, a, cnt)			_mm_maskz_revbit_epi8(k, _mm_sllv_gfni_epi8(_mm_revbit_epi8(a), cnt))

#define _mm256_srlv_gfni_epi8(a, cnt)				_mm256_revbit_epi8(_mm256_sllv_gfni_epi8(_mm256_revbit_epi8(a), cnt))
#define _mm256_mask_srlv_gfni_epi8(a, k, b, cnt)	_mm256_mask_revbit_epi8(a, k, _mm256_sllv_gfni_epi8(_mm256_revbit_epi8(b), cnt))
#define _mm256_maskz_srlv_gfni_epi8(k, a, cnt)		_mm256_maskz_revbit_epi8(k, _mm256_sllv_gfni_epi8(_mm256_revbit_epi8(a), cnt))

#define _mm512_srlv_gfni_epi8(a, cnt)				_mm512_revbit_epi8(_mm512_sllv_gfni_epi8(_mm512_revbit_epi8(a), cnt))
#define _mm512_mask_srlv_gfni_epi8(a, k, b, cnt)	_mm512_mask_revbit_epi8(a, k, _mm512_sllv_gfni_epi8(_mm512_revbit_epi8(b), cnt))
#define _mm512_maskz_srlv_gfni_epi8(k, a, cnt)		_mm512_maskz_revbit_epi8(k, _mm512_sllv_gfni_epi8(_mm512_revbit_epi8(a), cnt))

/* Variable rotate left bytes by cnt bits, cnt mod 8 */
#define _mm_rolv_gfni_epi8(a, cnt)					_mm_or_si128(_mm_sllv_gfni_epi8(a, _mm_and_si128(_mm_set1_epi32(0x07070707), cnt)), _mm_srlv_gfni_epi8(a, _mm_sub_epi8(_mm_set1_epi32(0x08080808), _mm_and_si128(_mm_set1_epi32(0x07070707), cnt))))
#define _mm_mask_rolv_gfni_epi8(a, k, b, cnt)		_mm_or_si128(_mm_mask_sllv_gfni_epi8(a, k, b, _mm_and_si128(_mm_set1_epi32(0x07070707), cnt)), _mm_mask_srlv_gfni_epi8(a, k, b, _mm_sub_epi8(_mm_set1_epi32(0x08080808), _mm_and_si128(_mm_set1_epi32(0x07070707), cnt))))
#define _mm_maskz_rolv_gfni_epi8(k, a, cnt)			_mm_or_si128(_mm_maskz_sllv_gfni_epi8(k, a, _mm_and_si128(_mm_set1_epi32(0x07070707), cnt)), _mm_maskz_srlv_gfni_epi8(k, a, _mm_sub_epi8(_mm_set1_epi32(0x08080808), _mm_and_si128(_mm_set1_epi32(0x07070707), cnt))))

#define _mm256_rolv_gfni_epi8(a, cnt)				_mm256_or_si256(_mm256_sllv_gfni_epi8(a, _mm256_and_si256(_mm256_set1_epi32(0x07070707), cnt)), _mm256_srlv_gfni_epi8(a, _mm256_sub_epi8(_mm256_set1_epi32(0x08080808), _mm256_and_si256(_mm256_set1_epi32(0x07070707), cnt))))
#define _mm256_mask_rolv_gfni_epi8(a, k, b, cnt)	_mm256_or_si256(_mm256_mask_sllv_gfni_epi8(a, k, b, _mm256_and_si256(_mm256_set1_epi32(0x07070707), cnt)), _mm256_mask_srlv_gfni_epi8(a, k, b, _mm256_sub_epi8(_mm256_set1_epi32(0x08080808), _mm256_and_si256(_mm256_set1_epi32(0x07070707), cnt))))
#define _mm256_maskz_rolv_gfni_epi8(k, a, cnt)		_mm256_or_si256(_mm256_maskz_sllv_gfni_epi8(k, a, _mm256_and_si256(_mm256_set1_epi32(0x07070707), cnt)), _mm256_maskz_srlv_gfni_epi8(k, a, _mm256_sub_epi8(_mm256_set1_epi32(0x08080808), _mm256_and_si256(_mm256_set1_epi32(0x07070707), cnt))))

#define _mm512_rolv_gfni_epi8(a, cnt)				_mm512_or_si512(_mm512_sllv_gfni_epi8(a, _mm512_and_si512(_mm512_set1_epi32(0x07070707), cnt)), _mm512_srlv_gfni_epi8(a, _mm512_sub_epi8(_mm512_set1_epi32(0x08080808), _mm512_and_si512(_mm512_set1_epi32(0x07070707), cnt))))
#define _mm512_mask_rolv_gfni_epi8(a, k, b, cnt)	_mm512_or_si512(_mm512_mask_sllv_gfni_epi8(a, k, b, _mm512_and_si512(_mm512_set1_epi32(0x07070707), cnt)), _mm512_mask_srlv_gfni_epi8(a, k, b, _mm512_sub_epi8(_mm512_set1_epi32(0x08080808), _mm512_and_si512(_mm512_set1_epi32(0x07070707), cnt))))
#define _mm512_maskz_rolv_gfni_epi8(k, a, cnt)		_mm512_or_si512(_mm512_maskz_sllv_gfni_epi8(k, a, _mm512_and_si512(_mm512_set1_epi32(0x07070707), cnt)), _mm512_maskz_srlv_gfni_epi8(k, a, _mm512_sub_epi8(_mm512_set1_epi32(0x08080808), _mm512_and_si512(_mm512_set1_epi32(0x07070707), cnt))))

/* Variable rotate right bytes by cnt bits, cnt mod 8 */

#define _mm_rorv_gfni_epi8(a, cnt)					_mm_or_si128(_mm_sllv_gfni_epi8(a, _mm_sub_epi8(_mm_set1_epi32(0x08080808), _mm_and_si128(_mm_set1_epi32(0x07070707), cnt))), _mm_srlv_gfni_epi8(a, _mm_and_si128(_mm_set1_epi32(0x07070707), cnt)))
#define _mm_mask_rorv_gfni_epi8(a, k, b, cnt)		_mm_or_si128(_mm_mask_sllv_gfni_epi8(a, k, b, _mm_sub_epi8(_mm_set1_epi32(0x08080808), _mm_and_si128(_mm_set1_epi32(0x07070707), cnt))), _mm_mask_srlv_gfni_epi8(a, k, b, _mm_and_si128(_mm_set1_epi32(0x07070707), cnt)))
#define _mm_maskz_rorv_gfni_epi8(k, a, cnt)			_mm_or_si128(_mm_maskz_sllv_gfni_epi8(k, a, _mm_sub_epi8(_mm_set1_epi32(0x08080808), _mm_and_si128(_mm_set1_epi32(0x07070707), cnt))), _mm_maskz_srlv_gfni_epi8(k, a, _mm_and_si128(_mm_set1_epi32(0x07070707), cnt)))

#define _mm256_rorv_gfni_epi8(a, cnt)				_mm256_or_si256(_mm256_sllv_gfni_epi8(a, _mm256_sub_epi8(_mm256_set1_epi32(0x08080808), _mm256_and_si256(_mm256_set1_epi32(0x07070707), cnt))), _mm256_srlv_gfni_epi8(a, _mm256_and_si256(_mm256_set1_epi32(0x07070707), cnt)))
#define _mm256_mask_rorv_gfni_epi8(a, k, b, cnt)	_mm256_or_si256(_mm256_mask_sllv_gfni_epi8(a, k, b, _mm256_sub_epi8(_mm256_set1_epi32(0x08080808), _mm256_and_si256(_mm256_set1_epi32(0x07070707), cnt))), _mm256_mask_srlv_gfni_epi8(a, k, b, _mm256_and_si256(_mm256_set1_epi32(0x07070707), cnt)))
#define _mm256_maskz_rorv_gfni_epi8(k, a, cnt)		_mm256_or_si256(_mm256_maskz_sllv_gfni_epi8(k, a, _mm256_sub_epi8(_mm256_set1_epi32(0x08080808), _mm256_and_si256(_mm256_set1_epi32(0x07070707), cnt))), _mm256_maskz_srlv_gfni_epi8(k, a, _mm256_and_si256(_mm256_set1_epi32(0x07070707), cnt)))

#define _mm512_rorv_gfni_epi8(a, cnt)				_mm512_or_si512(_mm512_sllv_gfni_epi8(a, _mm512_sub_epi8(_mm512_set1_epi32(0x08080808), _mm512_and_si512(_mm512_set1_epi32(0x07070707), cnt))), _mm512_srlv_gfni_epi8(a, _mm512_and_si512(_mm512_set1_epi32(0x07070707), cnt)))
#define _mm512_mask_rorv_gfni_epi8(a, k, b, cnt)	_mm512_or_si512(_mm512_mask_sllv_gfni_epi8(a, k, b, _mm512_sub_epi8(_mm512_set1_epi32(0x08080808), _mm512_and_si512(_mm512_set1_epi32(0x07070707), cnt))), _mm512_mask_srlv_gfni_epi8(a, k, b, _mm512_and_si512(_mm512_set1_epi32(0x07070707), cnt)))
#define _mm512_maskz_rorv_gfni_epi8(k, a, cnt)		_mm512_or_si512(_mm512_maskz_sllv_gfni_epi8(k, a, _mm512_sub_epi8(_mm512_set1_epi32(0x08080808), _mm512_and_si512(_mm512_set1_epi32(0x07070707), cnt))), _mm512_maskz_srlv_gfni_epi8(k, a, _mm512_and_si512(_mm512_set1_epi32(0x07070707), cnt)))

/* Reverse bits within bytes */
/* In  : MSB B7 B6 B5 B4 B3 B2 B1 B0 LSB */
/* Out : MSB B0 B1 B2 B3 B4 B5 B6 B7 LSB */

#define _mm_revbit_epi8(a)							_mm_gf2p8affine_epi64_epi8(a, _mm_set1_epi64x(_GFNI_DEMO_REVBIT), 0)
#define _mm_mask_revbit_epi8(s, k, a)				_mm_mask_gf2p8affine_epi64_epi8(s, k, a, _mm_set1_epi64x(_GFNI_DEMO_REVBIT), 0)
#define _mm_maskz_revbit_epi8(k, a)					_mm_maskz_gf2p8affine_epi64_epi8(k, a, _mm_set1_epi64x(_GFNI_DEMO_REVBIT), 0)

#define _mm256_revbit_epi8(a)						_mm256_gf2p8affine_epi64_epi8(a, _mm256_set1_epi64x(_GFNI_DEMO_REVBIT), 0)
#define _mm256_mask_revbit_epi8(s, k, a)			_mm256_mask_gf2p8affine_epi64_epi8(s, k, a, _mm256_set1_epi64x(_GFNI_DEMO_REVBIT), 0)
#define _mm256_maskz_revbit_epi8(k, a)				_mm256_maskz_gf2p8affine_epi64_epi8(k, a, _mm256_set1_epi64x(_GFNI_DEMO_REVBIT), 0)

#define _mm512_revbit_epi8(a)						_mm512_gf2p8affine_epi64_epi8(a, _mm512_set1_epi64(_GFNI_DEMO_REVBIT), 0)
#define _mm512_mask_revbit_epi8(s, k, a)			_mm512_mask_gf2p8affine_epi64_epi8(s, k, a, _mm512_set1_epi64(_GFNI_DEMO_REVBIT), 0)
#define _mm512_maskz_revbit_epi8(k, a)				_mm512_maskz_gf2p8affine_epi64_epi8(k, a, _mm512_set1_epi64(_GFNI_DEMO_REVBIT), 0)

/* Broadcast b-th bit within bytes */
/* In  : MSB B7 B6 B5 B4 B3 B2 B1 B0 LSB */
/* Out : MSB Bb Bb Bb Bb Bb Bb Bb Bb LSB */

#define _mm_bcstbit_epi8(a, b)						_mm_gf2p8affine_epi64_epi8(a, _mm_set1_epi64x(_GFNI_DEMO_BCST << b), 0)
#define _mm_mask_bcstbit_epi8(s, k, a, b)			_mm_mask_gf2p8affine_epi64_epi8(s, k, a, _mm_set1_epi64x(_GFNI_DEMO_BCST << b), 0)
#define _mm_maskz_bcstbit_epi8(k, a, b)				_mm_maskz_gf2p8affine_epi64_epi8(k, a, _mm_set1_epi64x(_GFNI_DEMO_BCST << b), 0)

#define _mm256_bcstbit_epi8(a, b)					_mm256_gf2p8affine_epi64_epi8(a, _mm256_set1_epi64x(_GFNI_DEMO_BCST << b), 0)
#define _mm256_mask_bcstbit_epi8(s, k, a, b)		_mm256_mask_gf2p8affine_epi64_epi8(s, k, a, _mm256_set1_epi64x(_GFNI_DEMO_BCST << b), 0)
#define _mm256_maskz_bcstbit_epi8(k, a, b)			_mm256_maskz_gf2p8affine_epi64_epi8(k, a, _mm256_set1_epi64x(_GFNI_DEMO_BCST << b), 0)

#define _mm512_bcstbit_epi8(a, b)					_mm512_gf2p8affine_epi64_epi8(a, _mm512_set1_epi64(_GFNI_DEMO_BCST << b), 0)
#define _mm512_mask_bcstbit_epi8(s, k, a, b)		_mm512_mask_gf2p8affine_epi64_epi8(s, k, a, _mm512_set1_epi64(_GFNI_DEMO_BCST << b), 0)
#define _mm512_maskz_bcstbit_epi8(k, a, b)			_mm512_maskz_gf2p8affine_epi64_epi8(k, a, _mm512_set1_epi64(_GFNI_DEMO_BCST << b), 0)

/* Prefix xor in bytes */
/* In  : MSB B7 B6 B5 B4 B3 B2 B1 B0 LSB */
/* ------------------------------------- */
/*           B0                          */
/*           ^                           */
/*           B1 B0                       */
/*           ^  ^                        */
/*           B2 B1 B0                    */
/*           ^  ^  ^                     */
/*           B3 B2 B1 B0                 */
/*           ^  ^  ^  ^                  */
/*           B4 B3 B2 B1 B0              */
/*           ^  ^  ^  ^  ^               */
/*           B5 B4 B3 B2 B1 B0           */
/*           ^  ^  ^  ^  ^  ^            */
/*           B6 B5 B4 B3 B2 B1 B0        */
/*           ^  ^  ^  ^  ^  ^  ^         */
/* Out : MSB B7 B6 B5 B4 B3 B2 B1 B0 LSB */

#define _mm_prefix_xor_epi8(a)						_mm_gf2p8affine_epi64_epi8(a, _mm_set1_epi64x(_GFNI_DEMO_PREFXOR), 0)
#define _mm_mask_prefix_xor_epi8(s, k, a)			_mm_mask_gf2p8affine_epi64_epi8(s, k, a, _mm_set1_epi64x(_GFNI_DEMO_PREFXOR), 0)
#define _mm_maskz_prefix_xor_epi8(k, a)				_mm_maskz_gf2p8affine_epi64_epi8(k, a, _mm_set1_epi64x(_GFNI_DEMO_PREFXOR), 0)

#define _mm256_prefix_xor_epi8(a)					_mm256_gf2p8affine_epi64_epi8(a, _mm256_set1_epi64x(_GFNI_DEMO_PREFXOR), 0)
#define _mm256_mask_prefix_xor_epi8(s, k, a)		_mm256_mask_gf2p8affine_epi64_epi8(s, k, a, _mm256_set1_epi64x(_GFNI_DEMO_PREFXOR), 0)
#define _mm256_maskz_prefix_xor_epi8(k, a)			_mm256_maskz_gf2p8affine_epi64_epi8(k, a, _mm256_set1_epi64x(_GFNI_DEMO_PREFXOR), 0)

#define _mm512_prefix_xor_epi8(a)					_mm512_gf2p8affine_epi64_epi8(a, _mm512_set1_epi64(_GFNI_DEMO_PREFXOR), 0)
#define _mm512_mask_prefix_xor_epi8(s, k, a)		_mm512_mask_gf2p8affine_epi64_epi8(s, k, a, _mm512_set1_epi64(_GFNI_DEMO_PREFXOR), 0)
#define _mm512_maskz_prefix_xor_epi8(k, a)			_mm512_maskz_gf2p8affine_epi64_epi8(k, a, _mm512_set1_epi64(_GFNI_DEMO_PREFXOR), 0)

/* Rotating bit in qwords around center  */
/*                                       */
/*           In[i,j] -> Out[j,7-i]       */
/*                                       */
/* In  : MSB 77 76 75 74 73 72 71 70     */
/*           67 66 65 64 63 62 61 60     */
/*           57 56 55 54 53 52 51 50     */
/*           47 46 45 44 43 42 41 40     */
/*           37 36 35 34 33 32 31 30     */
/*           27 26 25 24 23 22 21 20     */
/*           17 16 15 14 13 12 11 10     */
/*           07 06 05 04 03 02 01 00 LSB */
/*                                       */
/* Out : MSB 07 17 27 37 47 57 67 77     */
/*           06 16 26 36 46 56 66 76     */
/*           05 15 25 35 45 55 65 75     */
/*           04 14 24 34 44 54 64 74     */
/*           03 13 23 33 43 53 63 73     */
/*           02 12 22 32 42 52 62 72     */
/*           01 11 21 31 41 51 61 71     */
/*           00 10 20 30 40 50 60 70 LSB */

#define _mm_rotate_8x8(a)							_mm_gf2p8affine_epi64_epi8(_mm_set1_epi64x(_GFNI_DEMO_REVBIT), a, 0)
#define _mm_mask_rotate_8x8(s, k, a)				_mm_mask_gf2p8affine_epi64_epi8(s, k, _mm_set1_epi64x(_GFNI_DEMO_REVBIT), a, 0)
#define _mm_maskz_rotate_8x8(k, a)					_mm_maskz_gf2p8affine_epi64_epi8(k, _mm_set1_epi64x(_GFNI_DEMO_REVBIT), a, 0)

#define _mm256_rotate_8x8(a)						_mm256_gf2p8affine_epi64_epi8(_mm256_set1_epi64x(_GFNI_DEMO_REVBIT), a, 0)
#define _mm256_mask_rotate_8x8(s, k, a)				_mm256_mask_gf2p8affine_epi64_epi8(s, k, _mm256_set1_epi64x(_GFNI_DEMO_REVBIT), a, 0)
#define _mm256_maskz_rotate_8x8(k, a)				_mm256_maskz_gf2p8affine_epi64_epi8(k, _mm256_set1_epi64x(_GFNI_DEMO_REVBIT), a, 0)

#define _mm512_rotate_8x8(a)						_mm512_gf2p8affine_epi64_epi8(_mm512_set1_epi64(_GFNI_DEMO_REVBIT), a, 0)
#define _mm512_mask_rotate_8x8(s, k, a)				_mm512_mask_gf2p8affine_epi64_epi8(s, k, _mm512_set1_epi64(_GFNI_DEMO_REVBIT), a, 0)
#define _mm512_maskz_rotate_8x8(k, a)				_mm512_maskz_gf2p8affine_epi64_epi8(k, _mm512_set1_epi64(_GFNI_DEMO_REVBIT), a, 0)

/* Mirror bits in qwords, through the    */
/* 07-16-25-34-43-52-61-70 diagonal axis */
/*                                       */
/*           In[i,j] -> Out[7-j,7-i]     */
/*                                       */
/* In  : MSB 77 76 75 74 73 72 71 70     */
/*           67 66 65 64 63 62 61 60     */
/*           57 56 55 54 53 52 51 50     */
/*           47 46 45 44 43 42 41 40     */
/*           37 36 35 34 33 32 31 30     */
/*           27 26 25 24 23 22 21 20     */
/*           17 16 15 14 13 12 11 10     */
/*           07 06 05 04 03 02 01 00 LSB */
/*                                       */
/* Out : MSB 00 10 20 30 40 50 60 70     */
/*           01 11 21 31 41 51 61 71     */
/*           02 12 22 32 42 52 62 72     */
/*           03 13 23 33 43 53 63 73     */
/*           04 14 24 34 44 54 64 74     */
/*           05 15 25 35 45 55 65 75     */
/*           06 16 26 36 46 56 66 76     */
/*           07 17 27 37 47 57 67 77 LSB */

#define _mm_mirror_8x8(a)							_mm_gf2p8affine_epi64_epi8(_mm_set1_epi64x(_GFNI_DEMO_IDENT), a, 0)
#define _mm_mask_mirror_8x8(s, k, a)				_mm_mask_gf2p8affine_epi64_epi8(s, k, _mm_set1_epi64x(_GFNI_DEMO_IDENT), a, 0)
#define _mm_maskz_mirror_8x8(k, a)					_mm_maskz_gf2p8affine_epi64_epi8(k, _mm_set1_epi64x(_GFNI_DEMO_IDENT), a, 0)

#define _mm256_mirror_8x8(a)						_mm256_gf2p8affine_epi64_epi8(_mm256_set1_epi64x(_GFNI_DEMO_IDENT), a, 0)
#define _mm256_mask_mirror_8x8(s, k, a)				_mm256_mask_gf2p8affine_epi64_epi8(s, k, _mm256_set1_epi64x(_GFNI_DEMO_IDENT), a, 0)
#define _mm256_maskz_mirror_8x8(k, a)				_mm256_maskz_gf2p8affine_epi64_epi8(k, _mm256_set1_epi64x(_GFNI_DEMO_IDENT), a, 0)

#define _mm512_mirror_8x8(a)						_mm512_gf2p8affine_epi64_epi8(_mm512_set1_epi64(_GFNI_DEMO_IDENT), a, 0)
#define _mm512_mask_mirror_8x8(s, k, a)				_mm512_mask_gf2p8affine_epi64_epi8(s, k, _mm512_set1_epi64(_GFNI_DEMO_IDENT), a, 0)
#define _mm512_maskz_mirror_8x8(k, a)				_mm512_maskz_gf2p8affine_epi64_epi8(k, _mm512_set1_epi64(_GFNI_DEMO_IDENT), a, 0)

/* Multiplication 8x8 bits in qwords */
/*                                   */
/*              7                    */
/* res[i][j] = XOR(a[i][k],b[k][j])  */
/*             k=0                   */

#define _mm_multiplication_8x8(a, b)				_mm_gf2p8affine_epi64_epi8(_mm_mirror_8x8(b), a, 0)
#define _mm_mask_multiplication_8x8(s, k, a, b)		_mm_mask_gf2p8affine_epi64_epi8(s, k, _mm_mirror_8x8(b), a, 0)
#define _mm_maskz_multiplication_8x8(k, a, b)		_mm_maskz_gf2p8affine_epi64_epi8(k, _mm_mirror_8x8(b), a, 0)

#define _mm256_multiplication_8x8(a, b)				_mm256_gf2p8affine_epi64_epi8(_mm256_mirror_8x8(b), a, 0)
#define _mm256_mask_multiplication_8x8(s, k, a, b)	_mm256_mask_gf2p8affine_epi64_epi8(s, k, _mm256_mirror_8x8(b), a, 0)
#define _mm256_maskz_multiplication_8x8(k, a, b)	_mm256_maskz_gf2p8affine_epi64_epi8(k, _mm256_mirror_8x8(b), a, 0)

#define _mm512_multiplication_8x8(a, b)				_mm512_gf2p8affine_epi64_epi8(_mm512_mirror_8x8(b), a, 0)
#define _mm512_mask_multiplication_8x8(s, k, a, b)	_mm512_mask_gf2p8affine_epi64_epi8(s, k, _mm512_mirror_8x8(b), a, 0)
#define _mm512_maskz_multiplication_8x8(k, a, b)	_mm512_maskz_gf2p8affine_epi64_epi8(k, _mm512_mirror_8x8(b), a, 0)

/* Inverse bits within result bytes, conjunction with other operations */
/* In  : MSB  B7  B6  B5  B4  B3  B2  B1  B0 LSB*/
/* Out : MSB ~B7 ~B6 ~B5 ~B4 ~B3 ~B2 ~B1 ~B0 LSB*/

#define _mm_inverse_epi8(a)							_mm_gf2p8affine_epi64_epi8(a, _mm_set1_epi64x(_GFNI_DEMO_IDENT), 0xff)
#define _mm_mask_inverse_epi8(s, k, a)				_mm_mask_gf2p8affine_epi64_epi8(s, k, a, _mm_set1_epi64x(_GFNI_DEMO_IDENT), 0xff)
#define _mm_maskz_inverse_epi8(k, a)				_mm_maskz_gf2p8affine_epi64_epi8(k, a, _mm_set1_epi64x(_GFNI_DEMO_IDENT), 0xff)

#define _mm256_inverse_epi8(a)						_mm256_gf2p8affine_epi64_epi8(a, _mm256_set1_epi64x(_GFNI_DEMO_IDENT), 0xff)
#define _mm256_mask_inverse_epi8(s, k, a)			_mm256_mask_gf2p8affine_epi64_epi8(s, k, a, _mm256_set1_epi64x(_GFNI_DEMO_IDENT), 0xff)
#define _mm256_maskz_inverse_epi8(k, a)				_mm256_maskz_gf2p8affine_epi64_epi8(k, a, _mm256_set1_epi64x(_GFNI_DEMO_IDENT), 0xff)

#define _mm512_inverse_epi8(a)						_mm512_gf2p8affine_epi64_epi8(a, _mm512_set1_epi64(_GFNI_DEMO_IDENT), 0xff)
#define _mm512_mask_inverse_epi8(s, k, a)			_mm512_mask_gf2p8affine_epi64_epi8(s, k, a, _mm512_set1_epi64(_GFNI_DEMO_IDENT), 0xff)
#define _mm512_maskz_inverse_epi8(k, a)				_mm512_maskz_gf2p8affine_epi64_epi8(k, a, _mm512_set1_epi64(_GFNI_DEMO_IDENT), 0xff)

/* Compile time known imm byte a broadcast without touching the memory or Port5 */

#define _mm_set1_gfni_epi8(a)						_mm_gf2p8affine_epi64_epi8(_mm_setzero_si128(), _mm_setzero_si128(), a)
#define _mm_mask_set1_gfni_epi8(s, k, a)			_mm_mask_gf2p8affine_epi64_epi8(s, k, _mm_setzero_si128(), _mm_setzero_si128(), a)
#define _mm_maskz_set1_gfni_epi8(k, a)				_mm_maskz_gf2p8affine_epi64_epi8(k, _mm_setzero_si128(), _mm_setzero_si128(), a)

#define _mm256_set1_gfni_epi8(a)					_mm256_gf2p8affine_epi64_epi8(_mm256_setzero_si256(), _mm256_setzero_si256(), a)
#define _mm256_mask_set1_gfni_epi8(s, k, a)			_mm256_mask_gf2p8affine_epi64_epi8(s, k, _mm256_setzero_si256(), _mm256_setzero_si256(), a)
#define _mm256_maskz_set1_gfni_epi8(k, a)			_mm256_maskz_gf2p8affine_epi64_epi8(k, _mm256_setzero_si256(), _mm256_setzero_si256(), a)

#define _mm512_set1_gfni_epi8(a)					_mm512_gf2p8affine_epi64_epi8(_mm512_setzero_si512(), _mm512_setzero_si512(), a)
#define _mm512_mask_set1_gfni_epi8(s, k, a)			_mm512_mask_gf2p8affine_epi64_epi8(s, k, _mm512_setzero_si512(), _mm512_setzero_si512(), a)
#define _mm512_maskz_set1_gfni_epi8(k, a)			_mm512_maskz_gf2p8affine_epi64_epi8(k, _mm512_setzero_si512(), _mm512_setzero_si512(), a)

#if defined(__AVX512F__) && defined(_M_X64)
/* Pospopcount u8  */
/* In  : MSB        77 76 75 74 73 72 71 70      */
/*                  67 66 65 64 63 62 61 60      */
/*                  57 56 55 54 53 52 51 50      */
/*                  47 46 45 44 43 42 41 40      */
/*                  37 36 35 34 33 32 31 30      */
/*                  27 26 25 24 23 22 21 20      */
/*                  17 16 15 14 13 12 11 10      */
/*                  07 06 05 04 03 02 01 00 LSB  */
/*                                               */
/* Out : MSB popcnt(77,67,57,47,37,27,17,07)     */
/*           popcnt(76,66,56,46,36,26,16,06)     */
/*           popcnt(75,65,55,45,35,25,15,05)     */
/*           popcnt(74,64,54,44,34,24,14,04)     */
/*           popcnt(73,63,53,43,33,23,13,03)     */
/*           popcnt(72,62,52,42,32,22,12,02)     */
/*           popcnt(71,61,51,41,31,21,11,01)     */
/*           popcnt(70,60,50,40,30,20,10,00) LSB */

#define _mm_pospopcnt_u8_si128_epi8(a)				_mm_cvtepi16_epi8(_mm_popcnt_epi16(_mm_shuffle_epi8(_mm_rotate_8x8(a), _mm_set_epi64x(0x0f070e060d050c04, 0x0b030a0209010800))))
#define _mm256_pospopcnt_u8_si256_epi8(a)			_mm256_cvtepi32_epi8(_mm256_popcnt_epi32(_mm256_permutexvar_epi8(_mm256_set_epi64x(0x1f170f071e160e06, 0x1d150d051c140c04, 0x1b130b031a120a02, 0x1911090118100800), _mm256_rotate_8x8(a))))
#define _mm512_pospopcnt_u8_si512_epi8(a)			_mm512_cvtepi64_epi8(_mm512_popcnt_epi64(_mm512_permutexvar_epi8(_mm512_set_epi64(0x3f372f271f170f07, 0x3e362e261e160e06, 0x3d352d251d150d05, 0x3c342c241c140c04, 0x3b332b231b130b03, 0x3a322a221a120a02, 0x3931292119110901, 0x3830282018100800), _mm512_rotate_8x8(a))))

/* Pospopcount u16  */
/* In:  MSB 7F 7E 7D 7C 7B 7A 79 78 77 76 75 74 73 72 71 70     */
/*          6F 6E 6D 6C 6B 6A 69 68 67 66 65 64 63 62 61 60     */
/*          5F 5E 5D 5C 5B 5A 59 58 57 56 55 54 53 52 51 50     */
/*          4F 4E 4D 4C 4B 4A 49 48 47 46 45 44 43 42 41 40     */
/*          3F 3E 3D 3C 3B 3A 39 38 37 36 35 34 33 32 31 30     */
/*          2F 2E 2D 2C 2B 2A 29 28 27 26 25 24 23 22 21 20     */
/*          1F 1E 1D 1C 1B 1A 19 18 17 16 15 14 13 12 11 10     */
/*          0F 0E 0D 0C 0B 0A 09 08 07 06 05 04 03 02 01 00 LSB */
/*                                              */
/* Out: MSB popcnt(7F,6F,5F,4F,3F,2F,1F,0F)     */
/*          popcnt(7E,6E,5E,4E,3E,2E,1E,0E)     */
/*          popcnt(7D,6D,5D,4D,3D,2D,1D,0D)     */
/*          popcnt(7C,6C,5C,4C,3C,2C,1C,0C)     */
/*          popcnt(7B,6B,5B,4B,3B,2B,1B,0B)     */
/*          popcnt(7A,6A,5A,4A,3A,2A,1A,0A)     */
/*          popcnt(79,69,59,49,39,29,19,09)     */
/*          popcnt(78,68,58,48,38,28,18,08)     */
/*          popcnt(77,67,57,47,37,27,17,07)     */
/*          popcnt(76,66,56,46,36,26,16,06)     */
/*          popcnt(75,65,55,45,35,25,15,05)     */
/*          popcnt(74,64,54,44,34,24,14,04)     */ 
/*          popcnt(73,63,53,43,33,23,13,03)     */
/*          popcnt(72,62,52,42,32,22,12,02)     */
/*          popcnt(71,61,51,41,31,21,11,01)     */
/*          popcnt(70,60,50,40,30,20,10,00) LSB */

#define _mm_pospopcnt_u16_si128_epi8(a)				_mm_popcnt_epi8(_mm_rotate_8x8(_mm_shuffle_epi8(a, _mm_set_epi64x(0x0f0d0b0907050301, 0x0e0c0a0806040200))))
#define _mm256_pospopcnt_u16_si256_epi8(a)			_mm256_cvtepi16_epi8(_mm256_popcnt_epi16(_mm256_permutexvar_epi8(_mm256_set_epi64x(0x1f0f1e0e1d0d1c0c, 0x1b0b1a0a19091808, 0x1707160615051404, 0x1303120211011000), _mm256_rotate_8x8(_mm256_shuffle_epi8(a, _mm256_broadcastsi128_si256(_mm_set_epi64x(0x0f0d0b0907050301, 0x0e0c0a0806040200)))))))
#define _mm512_pospopcnt_u16_si512_epi8(a)			_mm512_cvtepi32_epi8(_mm512_popcnt_epi32(_mm512_permutexvar_epi8(_mm512_set_epi64(0x3f2f1f0f3e2e1e0e, 0x3d2d1d0d3c2c1c0c, 0x3b2b1b0b3a2a1a0a, 0x3929190938281808, 0x3727170736261606, 0x3525150534241404, 0x3323130332221202, 0x3121110130201000), _mm512_rotate_8x8(_mm512_shuffle_epi8(a, _mm512_broadcast_i32x4(_mm_set_epi64x(0x0f0d0b0907050301, 0x0e0c0a0806040200)))))))
#endif

/* Count the number of trailing zero bits for packed bytes */
/* In  : MSB B7 B6 B5 B4 B3 B2 B1 B0 LSB */
/* Out : MSB  0  0  0  0       tzcnt LSB */

#define _mm_tzcnt_gfni_epi8(a)						_mm_gf2p8affine_epi64_epi8(_mm_andnot_si128(_mm_add_epi8(a, _mm_set1_epi32(-1)), a), _mm_set1_epi64x(_GFNI_DEMO_TZCNT), 0x8)
#define _mm_mask_tzcnt_gfni_epi8(s, k, a)			_mm_mask_gf2p8affine_epi64_epi8(s, k, _mm_andnot_si128(_mm_add_epi8(a, _mm_set1_epi32(-1)), a), _mm_set1_epi64x(_GFNI_DEMO_TZCNT), 0x8)
#define _mm_maskz_tzcnt_gfni_epi8(k, a)				_mm_maskz_gf2p8affine_epi64_epi8(k, _mm_andnot_si128(_mm_add_epi8(a, _mm_set1_epi32(-1)), a), _mm_set1_epi64x(_GFNI_DEMO_TZCNT), 0x8)

#define _mm256_tzcnt_gfni_epi8(a)					_mm256_gf2p8affine_epi64_epi8(_mm256_andnot_si256(_mm256_add_epi8(a, _mm256_set1_epi32(-1)), a), _mm256_set1_epi64x(_GFNI_DEMO_TZCNT), 0x8)
#define _mm256_mask_tzcnt_gfni_epi8(s, k, a)		_mm256_mask_gf2p8affine_epi64_epi8(s, k, _mm256_andnot_si256(_mm256_add_epi8(a, _mm256_set1_epi32(-1)), a), _mm256_set1_epi64x(_GFNI_DEMO_TZCNT), 0x8)
#define _mm256_maskz_tzcnt_gfni_epi8(k, a)			_mm256_maskz_gf2p8affine_epi64_epi8(k, _mm256_andnot_si256(_mm256_add_epi8(a, _mm256_set1_epi32(-1)), a), _mm256_set1_epi64x(_GFNI_DEMO_TZCNT), 0x8)

#define _mm512_tzcnt_gfni_epi8(a)					_mm512_gf2p8affine_epi64_epi8(_mm512_andnot_si512(_mm512_add_epi8(a, _mm512_set1_epi32(-1)), a), _mm512_set1_epi64(_GFNI_DEMO_TZCNT), 0x8)
#define _mm512_mask_tzcnt_gfni_epi8(s, k, a)		_mm512_mask_gf2p8affine_epi64_epi8(s, k, _mm512_andnot_si512(_mm512_add_epi8(a, _mm512_set1_epi32(-1)), a), _mm512_set1_epi64(_GFNI_DEMO_TZCNT), 0x8)
#define _mm512_maskz_tzcnt_gfni_epi8(k, a)			_mm512_maskz_gf2p8affine_epi64_epi8(k, _mm512_andnot_si512(_mm512_add_epi8(a, _mm512_set1_epi32(-1)), a), _mm512_set1_epi64(_GFNI_DEMO_TZCNT), 0x8)

/* Count the number of leading zero bits for packed bytes */
/* In  : MSB B7 B6 B5 B4 B3 B2 B1 B0 LSB */
/* Out : MSB  0  0  0  0       lzcnt LSB */

#define _mm_lzcnt_gfni_epi8(a)						_mm_tzcnt_gfni_epi8(_mm_revbit_epi8(a))
#define _mm_mask_lzcnt_gfni_epi8(s, k, a)			_mm_mask_tzcnt_gfni_epi8(s, k, _mm_revbit_epi8(a))	
#define _mm_maskz_lzcnt_gfni_epi8(k, a)				_mm_maskz_tzcnt_gfni_epi8(k, _mm_revbit_epi8(a))		

#define _mm256_lzcnt_gfni_epi8(a)					_mm256_tzcnt_gfni_epi8(_mm256_revbit_epi8(a))			
#define _mm256_mask_lzcnt_gfni_epi8(s, k, a)		_mm256_mask_tzcnt_gfni_epi8(s, k, _mm256_revbit_epi8(a))
#define _mm256_maskz_lzcnt_gfni_epi8(k, a)			_mm256_maskz_tzcnt_gfni_epi8(k, _mm256_revbit_epi8(a))	

#define _mm512_lzcnt_gfni_epi8(a)					_mm512_tzcnt_gfni_epi8(_mm512_revbit_epi8(a))			
#define _mm512_mask_lzcnt_gfni_epi8(s, k, a)		_mm512_mask_tzcnt_gfni_epi8(s, k, _mm512_revbit_epi8(a))
#define _mm512_maskz_lzcnt_gfni_epi8(k, a)			_mm512_maskz_tzcnt_gfni_epi8(k, _mm512_revbit_epi8(a))	



static inline
__m128i _mm_lzcnt_epi8(const __m128i a) noexcept {
	__m128i u = _mm_undefined_si128();
	__m128i r = _mm_revbit_epi8(a);
	return _mm_popcnt_epi8(_mm_andnot_si128(r, _mm_add_epi8(r, _mm_cmpeq_epi8(u, u))));
}

static inline
__m256i _mm256_lzcnt_epi8(const __m256i a) {
	__m256i u = _mm256_undefined_si256();
	__m256i r = _mm256_revbit_epi8(a);
	return _mm256_popcnt_epi8(_mm256_andnot_si256(r, _mm256_add_epi8(r, _mm256_cmpeq_epi8(u, u))));
}

static inline
__m512i _mm512_lzcnt_epi8(__m512i a) {
	__m512i u = _mm512_undefined_epi32();
	__m512i r = _mm512_revbit_epi8(a);
	return _mm512_popcnt_epi8(_mm512_andnot_si512(r, _mm512_add_epi8(r, _mm512_ternarylogic_epi32(u, u, u, 0xff))));
}

static inline
__m128i _mm_lzcnt_epi16(const __m128i a) noexcept {
	__m128i u = _mm_undefined_si128();
	__m128i r = _mm_revbit_epi8(_mm_swaplh_epi8(a));
	return _mm_popcnt_epi16(_mm_andnot_si128(r, _mm_add_epi16(r, _mm_cmpeq_epi16(u, u))));
}

static inline
__m256i _mm256_lzcnt_epi16(const __m256i a) noexcept {
	__m256i u = _mm256_undefined_si256();
	__m256i r = _mm256_revbit_epi8(_mm256_swaplh_epi8(a));
	return _mm256_popcnt_epi16(_mm256_andnot_si256(r,_mm256_add_epi16(r, _mm256_cmpeq_epi16(u, u))));
}

static inline
__m512i _mm512_lzcnt_epi16(const __m512i a) noexcept {
	__m512i u = _mm512_undefined_epi32();
	__m512i r = _mm512_revbit_epi8(_mm512_swaplh_epi8(a));
	return _mm512_popcnt_epi16(_mm512_andnot_si512(r, _mm512_add_epi16(r, _mm512_ternarylogic_epi32(u, u, u, 0xff))));
}

static inline
__m128i _mm_lzcnt_fp16_epi16(const __m128i a) noexcept {
	return _mm_min_epi16(_mm_sub_epi16(_mm_set1_epi16(0x1e), _mm_srli_epi16(_mm_cvtepu16_ph(a), 10)), _mm_set1_epi16(0x10));
}

static inline
__m256i _mm256_lzcnt_fp16_epi16(const __m256i a) noexcept {
	return _mm256_min_epi16(_mm256_sub_epi16(_mm256_set1_epi16(0x1e), _mm256_srli_epi16(_mm256_cvtepu16_ph(a), 10)), _mm256_set1_epi16(0x10));
}

static inline
__m512i _mm512_lzcnt_fp16_epi16(const __m512i a) noexcept {
	return _mm512_min_epi16(_mm512_sub_epi16(_mm512_set1_epi16(0x1e), _mm512_srli_epi16(_mm512_cvtepu16_ph(a), 10)), _mm512_set1_epi16(0x10));
}

static inline
__m128i _mm_tzcnt_epi8(const __m128i a) noexcept {
	__m128i u = _mm_undefined_si128();
	return _mm_popcnt_epi8(_mm_andnot_si128(a, _mm_add_epi8(a, _mm_cmpeq_epi8(u, u))));
}

static inline
__m256i _mm256_tzcnt_epi8(const __m256i a) noexcept {
	__m256i u = _mm256_undefined_si256();
	return _mm256_popcnt_epi8(_mm256_andnot_si256(a, _mm256_add_epi8(a, _mm256_cmpeq_epi8(u, u))));
}

static inline
__m512i _mm512_tzcnt_epi8(const __m512i a) noexcept {
	__m512i u = _mm512_undefined_epi32();
	return _mm512_popcnt_epi8(_mm512_andnot_si512(a, _mm512_add_epi8(a, _mm512_ternarylogic_epi32(u, u, u, 0xff))));
}

static inline
__m128i _mm_tzcnt_epi16(const __m128i a) noexcept {
	__m128i u = _mm_undefined_si128();
	return _mm_popcnt_epi16(_mm_andnot_si128(a, _mm_add_epi16(a, _mm_cmpeq_epi16(u, u))));
}

static inline
__m256i _mm256_tzcnt_epi16(const __m256i a) noexcept {
	__m256i u = _mm256_undefined_si256();
	return _mm256_popcnt_epi16(_mm256_andnot_si256(a,_mm256_add_epi16(a, _mm256_cmpeq_epi16(u, u))));
}

static inline
__m512i _mm512_tzcnt_epi16(const __m512i a) noexcept {
	__m512i u = _mm512_undefined_epi32();
	return _mm512_popcnt_epi16(_mm512_andnot_si512(a, _mm512_add_epi16(a, _mm512_ternarylogic_epi32(u, u, u, 0xff))));
} 

static inline
__m128i _mm_tzcnt_epi32(const __m128i a) noexcept {
	__m128i u = _mm_undefined_si128();
	return _mm_popcnt_epi32(_mm_andnot_si128(a, _mm_add_epi32(a, _mm_cmpeq_epi32(u, u))));
}

static inline
__m256i _mm256_tzcnt_epi32(const __m256i a) noexcept {
	__m256i u = _mm256_undefined_si256();
	return _mm256_popcnt_epi32(_mm256_andnot_si256(a, _mm256_add_epi32(a, _mm256_cmpeq_epi32(u, u))));
}

static inline
__m512i _mm512_tzcnt_epi32(const __m512i a) noexcept {
	__m512i u = _mm512_undefined_epi32();
	return _mm512_popcnt_epi32(_mm512_andnot_si512(a, _mm512_add_epi32(a, _mm512_ternarylogic_epi32(u, u, u, 0xff))));
}

static inline
__m128i _mm_tzcnt_epi64(const __m128i a) noexcept {
	__m128i u = _mm_undefined_si128();
	return _mm_popcnt_epi64(_mm_andnot_si128(a, _mm_add_epi64(a, _mm_cmpeq_epi64(u, u))));
}

static inline
__m256i _mm256_tzcnt_epi64(const __m256i a) noexcept {
	__m256i u = _mm256_undefined_si256();
	return _mm256_popcnt_epi64(_mm256_andnot_si256(a, _mm256_add_epi64(a, _mm256_cmpeq_epi64(u, u))));
}

static inline
__m512i _mm512_tzcnt_epi64(const __m512i a) noexcept {
	__m512i u = _mm512_undefined_epi32();
	return _mm512_popcnt_epi64(_mm512_andnot_si512(a, _mm512_add_epi64(a, _mm512_ternarylogic_epi64(u, u, u, 0xff))));
}

static inline
__m128i _mm_prefix_xor_clmul_si128(const __m128i a) noexcept {
	const __m128i full	= _mm_set1_epi32(0xffffffff);
	__m128i clmul0_63	= _mm_clmulepi64_si128(a, full, 0x00);
	__m128i clmul64_127	= _mm_clmulepi64_si128(a, full, 0x01);
#if !defined(__AVX512VL__)
	clmul64_127			= _mm_xor_si128(clmul64_127, _mm_shuffle_epi32(_mm_srai_epi32(clmul0_63, 31), 0x05));
#else
	clmul64_127			= _mm_xor_si128(clmul64_127, _mm_srai_epi64(clmul0_63, 63));
#endif
	return				_mm_unpacklo_epi64(clmul0_63, clmul64_127);
}

#if defined(__AVX2__)
static inline
__m256i _mm256_prefix_xor_clmul_si256(const __m256i a) noexcept {
	const __m256i full	= _mm256_set1_epi32(0xffffffff);
	__m256i clmul0_63	= _mm256_clmulepi64_epi128(a, full, 0x00);
	__m256i clmul64_127	= _mm256_clmulepi64_epi128(a, full, 0x01);
#if !defined(__AVX512VL__)
	clmul64_127			= _mm256_xor_si256(clmul64_127, _mm256_shuffle_epi32(_mm256_srai_epi32(clmul0_63, 31), 0x05));
#else
	clmul64_127			= _mm256_xor_si256(clmul64_127, _mm256_srai_epi64(clmul0_63, 63));
#endif
	__m256i clmul0_127	= _mm256_unpacklo_epi64(clmul0_63, clmul64_127);
#if !defined(__AVX512VL__)
	__m256i corr128_255	= _mm256_inserti128_si256(_mm256_setzero_si256(), _mm_shuffle_epi32(_mm_srai_epi32(_mm256_castsi256_si128(clmul0_127), 31), 0xff), 1);
#else
	__m256i corr128_255	= _mm256_maskz_permutex_epi64(0xc, _mm256_srai_epi64(clmul64_127, 63), 0);
#endif
	return				_mm256_xor_si256(clmul0_127, corr128_255);
}
#endif

#if defined(__AVX512F__)
static inline
__m512i _mm512_prefix_xor_clmul_si512(const __m512i a) noexcept {
	const __m512i full	= _mm512_set1_epi32(0xffffffff);
	__m512i clmul0_63	= _mm512_clmulepi64_epi128(a, full, 0x00);
	__m512i clmul64_127	= _mm512_clmulepi64_epi128(a, full, 0x01);
	clmul64_127			= _mm512_xor_si512(clmul64_127, _mm512_srai_epi64(clmul0_63, 63));
	__m512i clmul0_127	= _mm512_unpacklo_epi64(clmul0_63, clmul64_127);
	__m512i corr128_255	= _mm512_maskz_permutex_epi64(0xcc, _mm512_srai_epi64(clmul64_127, 63), 0);
	__m512i clmul0_255	= _mm512_xor_si512(clmul0_127, corr128_255);
	__m512i corr256_511	= _mm512_maskz_permutexvar_epi64(0xf0, _mm512_set1_epi64(3), _mm512_srai_epi64(clmul0_255, 63));
	return				 _mm512_xor_si512(clmul0_255, corr256_511);
}
#endif

// https://gist.github.com/animetosho/6cb732ccb5ecd86675ca0a442b3c0622
// Arbitrary Modular GF(2w) Multiplication
// 
// This might not seem so out-of-band, and you might even point out that there’s a GF2P8MULB instruction in the same extension, however this use case may not be so obvious, and has its benefits.
// 
// A key problem of the GF2P8MULB instruction is that it only applies to GF(28) with polynomial 0x11B. Multiplying via affine, however, does not have this limitation. Limitations of multiplying via affine, are that all values in an 8-byte group must be multiplied by the same coefficient, and you’ll need to compute the required matrices (or have them pre-computed).
// 
// Example: multiply a vector of bytes by b in GF(28) with polynomial 0x11d (commonly used in error correction, e.g. RAID6):
static const uint64_t gf2p8_11d_mul_matrices[256] = {
	0,0x102040810204080ULL,0x8001828488102040ULL,0x8103868c983060c0ULL,0x408041c2c4881020ULL,0x418245cad4a850a0ULL,0xc081c3464c983060ULL,0xc183c74e5cb870e0ULL,0x2040a061e2c48810ULL,0x2142a469f2e4c890ULL,0xa04122e56ad4a850ULL,0xa14326ed7af4e8d0ULL,0x60c0e1a3264c9830ULL,0x61c2e5ab366cd8b0ULL,0xe0c16327ae5cb870ULL,0xe1c3672fbe7cf8f0ULL,0x102050b071e2c488ULL,0x112254b861c28408ULL,0x9021d234f9f2e4c8ULL,0x9123d63ce9d2a448ULL,0x50a01172b56ad4a8ULL,0x51a2157aa54a9428ULL,0xd0a193f63d7af4e8ULL,0xd1a397fe2d5ab468ULL,0x3060f0d193264c98ULL,0x3162f4d983060c18ULL,0xb06172551b366cd8ULL,0xb163765d0b162c58ULL,0x70e0b11357ae5cb8ULL,0x71e2b51b478e1c38ULL,0xf0e13397dfbe7cf8ULL,0xf1e3379fcf9e3c78ULL,0x8810a8d83871e2c4ULL,0x8912acd02851a244ULL,0x8112a5cb061c284ULL,0x9132e54a0418204ULL,0xc890e91afcf9f2e4ULL,0xc992ed12ecd9b264ULL,0x48916b9e74e9d2a4ULL,0x49936f9664c99224ULL,0xa85008b9dab56ad4ULL,0xa9520cb1ca952a54ULL,0x28518a3d52a54a94ULL,0x29538e3542850a14ULL,0xe8d0497b1e3d7af4ULL,0xe9d24d730e1d3a74ULL,0x68d1cbff962d5ab4ULL,0x69d3cff7860d1a34ULL,0x9830f8684993264cULL,0x9932fc6059b366ccULL,0x18317aecc183060cULL,0x19337ee4d1a3468cULL,0xd8b0b9aa8d1b366cULL,0xd9b2bda29d3b76ecULL,0x58b13b2e050b162cULL,0x59b33f26152b56acULL,0xb8705809ab57ae5cULL,0xb9725c01bb77eedcULL,0x3871da8d23478e1cULL,0x3973de853367ce9cULL,0xf8f019cb6fdfbe7cULL,0xf9f21dc37ffffefcULL,0x78f19b4fe7cf9e3cULL,0x79f39f47f7efdebcULL,0xc488d46c1c3871e2ULL,0xc58ad0640c183162ULL,0x448956e8942851a2ULL,0x458b52e084081122ULL,0x840895aed8b061c2ULL,0x850a91a6c8902142ULL,0x409172a50a04182ULL,0x50b132240800102ULL,0xe4c8740dfefcf9f2ULL,0xe5ca7005eedcb972ULL,0x64c9f68976ecd9b2ULL,0x65cbf28166cc9932ULL,0xa44835cf3a74e9d2ULL,0xa54a31c72a54a952ULL,0x2449b74bb264c992ULL,0x254bb343a2448912ULL,0xd4a884dc6ddab56aULL,0xd5aa80d47dfaf5eaULL,0x54a90658e5ca952aULL,0x55ab0250f5ead5aaULL,0x9428c51ea952a54aULL,0x952ac116b972e5caULL,0x1429479a2142850aULL,0x152b43923162c58aULL,0xf4e824bd8f1e3d7aULL,0xf5ea20b59f3e7dfaULL,0x74e9a639070e1d3aULL,0x75eba231172e5dbaULL,0xb468657f4b962d5aULL,0xb56a61775bb66ddaULL,0x3469e7fbc3860d1aULL,0x356be3f3d3a64d9aULL,0x4c987cb424499326ULL,0x4d9a78bc3469d3a6ULL,0xcc99fe30ac59b366ULL,0xcd9bfa38bc79f3e6ULL,0xc183d76e0c18306ULL,0xd1a397ef0e1c386ULL,0x8c19bff268d1a346ULL,0x8d1bbbfa78f1e3c6ULL,0x6cd8dcd5c68d1b36ULL,0x6ddad8ddd6ad5bb6ULL,0xecd95e514e9d3b76ULL,0xeddb5a595ebd7bf6ULL,0x2c589d1702050b16ULL,0x2d5a991f12254b96ULL,0xac591f938a152b56ULL,0xad5b1b9b9a356bd6ULL,0x5cb82c0455ab57aeULL,0x5dba280c458b172eULL,0xdcb9ae80ddbb77eeULL,0xddbbaa88cd9b376eULL,0x1c386dc69123478eULL,0x1d3a69ce8103070eULL,0x9c39ef42193367ceULL,0x9d3beb4a0913274eULL,0x7cf88c65b76fdfbeULL,0x7dfa886da74f9f3eULL,0xfcf90ee13f7ffffeULL,0xfdfb0ae92f5fbf7eULL,0x3c78cda773e7cf9eULL,0x3d7ac9af63c78f1eULL,0xbc794f23fbf7efdeULL,0xbd7b4b2bebd7af5eULL,0xe2c46a368e1c3871ULL,0xe3c66e3e9e3c78f1ULL,0x62c5e8b2060c1831ULL,0x63c7ecba162c58b1ULL,0xa2442bf44a942851ULL,0xa3462ffc5ab468d1ULL,0x2245a970c2840811ULL,0x2347ad78d2a44891ULL,0xc284ca576cd8b061ULL,0xc386ce5f7cf8f0e1ULL,0x428548d3e4c89021ULL,0x43874cdbf4e8d0a1ULL,0x82048b95a850a041ULL,0x83068f9db870e0c1ULL,0x205091120408001ULL,0x3070d193060c081ULL,0xf2e43a86fffefcf9ULL,0xf3e63e8eefdebc79ULL,0x72e5b80277eedcb9ULL,0x73e7bc0a67ce9c39ULL,0xb2647b443b76ecd9ULL,0xb3667f4c2b56ac59ULL,0x3265f9c0b366cc99ULL,0x3367fdc8a3468c19ULL,0xd2a49ae71d3a74e9ULL,0xd3a69eef0d1a3469ULL,0x52a51863952a54a9ULL,0x53a71c6b850a1429ULL,0x9224db25d9b264c9ULL,0x9326df2dc9922449ULL,0x122559a151a24489ULL,0x13275da941820409ULL,0x6ad4c2eeb66ddab5ULL,0x6bd6c6e6a64d9a35ULL,0xead5406a3e7dfaf5ULL,0xebd744622e5dba75ULL,0x2a54832c72e5ca95ULL,0x2b56872462c58a15ULL,0xaa5501a8faf5ead5ULL,0xab5705a0ead5aa55ULL,0x4a94628f54a952a5ULL,0x4b96668744891225ULL,0xca95e00bdcb972e5ULL,0xcb97e403cc993265ULL,0xa14234d90214285ULL,0xb16274580010205ULL,0x8a15a1c9183162c5ULL,0x8b17a5c108112245ULL,0x7af4925ec78f1e3dULL,0x7bf69656d7af5ebdULL,0xfaf510da4f9f3e7dULL,0xfbf714d25fbf7efdULL,0x3a74d39c03070e1dULL,0x3b76d79413274e9dULL,0xba7551188b172e5dULL,0xbb7755109b376eddULL,0x5ab4323f254b962dULL,0x5bb63637356bd6adULL,0xdab5b0bbad5bb66dULL,0xdbb7b4b3bd7bf6edULL,0x1a3473fde1c3860dULL,0x1b3677f5f1e3c68dULL,0x9a35f17969d3a64dULL,0x9b37f57179f3e6cdULL,0x264cbe5a92244993ULL,0x274eba5282040913ULL,0xa64d3cde1a3469d3ULL,0xa74f38d60a142953ULL,0x66ccff9856ac59b3ULL,0x67cefb90468c1933ULL,0xe6cd7d1cdebc79f3ULL,0xe7cf7914ce9c3973ULL,0x60c1e3b70e0c183ULL,0x70e1a3360c08103ULL,0x860d9cbff8f0e1c3ULL,0x870f98b7e8d0a143ULL,0x468c5ff9b468d1a3ULL,0x478e5bf1a4489123ULL,0xc68ddd7d3c78f1e3ULL,0xc78fd9752c58b163ULL,0x366ceeeae3c68d1bULL,0x376eeae2f3e6cd9bULL,0xb66d6c6e6bd6ad5bULL,0xb76f68667bf6eddbULL,0x76ecaf28274e9d3bULL,0x77eeab20376eddbbULL,0xf6ed2dacaf5ebd7bULL,0xf7ef29a4bf7efdfbULL,0x162c4e8b0102050bULL,0x172e4a831122458bULL,0x962dcc0f8912254bULL,0x972fc807993265cbULL,0x56ac0f49c58a152bULL,0x57ae0b41d5aa55abULL,0xd6ad8dcd4d9a356bULL,0xd7af89c55dba75ebULL,0xae5c1682aa55ab57ULL,0xaf5e128aba75ebd7ULL,0x2e5d940622458b17ULL,0x2f5f900e3265cb97ULL,0xeedc57406eddbb77ULL,0xefde53487efdfbf7ULL,0x6eddd5c4e6cd9b37ULL,0x6fdfd1ccf6eddbb7ULL,0x8e1cb6e348912347ULL,0x8f1eb2eb58b163c7ULL,0xe1d3467c0810307ULL,0xf1f306fd0a14387ULL,0xce9cf7218c193367ULL,0xcf9ef3299c3973e7ULL,0x4e9d75a504091327ULL,0x4f9f71ad142953a7ULL,0xbe7c4632dbb76fdfULL,0xbf7e423acb972f5fULL,0x3e7dc4b653a74f9fULL,0x3f7fc0be43870f1fULL,0xfefc07f01f3f7fffULL,0xfffe03f80f1f3f7fULL,0x7efd8574972f5fbfULL,0x7fff817c870f1f3fULL,0x9e3ce6533973e7cfULL,0x9f3ee25b2953a74fULL,0x1e3d64d7b163c78fULL,0x1f3f60dfa143870fULL,0xdebca791fdfbf7efULL,0xdfbea399eddbb76fULL,0x5ebd251575ebd7afULL,0x5fbf211d65cb972fULL
};

static inline
__m128i _mm_gf2p8mul_11d_epi8(const __m128i a,
                              const uint8_t b) noexcept {
	return _mm_gf2p8affine_epi64_epi8(a, _mm_set1_epi64x(gf2p8_11d_mul_matrices[b]), 0);
	/*
	movddup xmm1, [gf2p8_11d_mul_matrices + b*8]
	gf2p8affineqb xmm0, xmm1, 0
	
	# OR, with EVEX broadcast:
	vgf2p8affineqb xmm0, [gf2p8_11d_mul_matrices + b*8]{1to2}, 0
	*/
}


// Fixed 2-bit Packed Arithmetic
// If you’ve got packed 2-bit integers, you can do some basic arithmetic with constants without needing to unpack.
// same as sub3
static inline 
__m128i _mm_add1_epi2(const __m128i a) noexcept {
	return _mm_gf2p8affine_epi64_epi8(a, _mm_set_epi32(0x0103040c, 0x103040c0, 0x0103040c, 0x103040c0), 0x55);
}

// same as sub2
static inline 
__m128i _mm_add2_epi2(const __m128i a) noexcept { 
	return _mm_xor_si128(a, _mm_set1_epi8(0xaa));
}

// same as sub1
static inline 
__m128i _mm_add3_epi2(const __m128i a) noexcept { 
	return _mm_gf2p8affine_epi64_epi8(a, _mm_set_epi32(0x0103040c, 0x103040c0, 0x0103040c, 0x103040c0), 0xff);
}

// 1-x
static inline 
__m128i _mm_1sub_epi2(const __m128i a) noexcept {
	return _mm_xor_si128(a, _mm_set1_epi8(0x55));
}

// 2-x
static inline 
__m128i _mm_2sub_epi2(__m128i a) noexcept { 
	return _mm_gf2p8affine_epi64_epi8(a, _mm_set_epi32(0x0103040c, 0x103040c0, 0x0103040c, 0x103040c0), 0xaa);
}

// 3-x
static inline 
__m128i _mm_3sub_epi2(const __m128i a) noexcept { 
	return _mm_xor_si128(a, _mm_set1_epi8(0xff));
}

static inline 
__m128i _mm_mul2_epi2(const __m128i a) noexcept {
	return _mm_gf2p8affine_epi64_epi8(a, _mm_set_epi32(0x10004, 0x100080, 0x10004, 0x100080), 0);
}

// same as 0-x
static inline 
__m128i _mm_mul3_epi2(const __m128i a) noexcept {
	return _mm_gf2p8affine_epi64_epi8(a, _mm_set_epi32(0x0103040c, 0x103040c0, 0x0103040c, 0x103040c0), 0);
}

// Byte-wise variable shift
// Not using affine instruction, but one could use the mulb instruction for variable left-shifts
static inline
__m128i _mm_sllv_epi8(const __m128i a,
                      const __m128i count) noexcept {
	__m128i mask = _mm_shuffle_epi8(_mm_set_epi32(0,0, 0x0103070f, 0x1f3f7fff), count);
	__m128i b = _mm_and_si128(a, mask);
	__m128i multiplier = _mm_shuffle_epi8(_mm_set_epi32(0,0, 0x80402010, 0x08040201), count);
	return _mm_gf2p8mul_epi8(b, multiplier);
	
	/*
	movdqa xmm2, [mask_tbl]
	pshufb xmm2, xmm1
	pand xmm0, xmm2
	movdqa xmm2, [mult_tbl]
	pshufb xmm2, xmm1
	gf2p8mulb xmm0, xmm2
	 */
}

static inline
__m128i _mm_srlv_epi8(__m128i a,
                      const __m128i count) noexcept {
	// I can't think of a faster way than reversing the bits twice and using the above :(
	a = _mm_gf2p8affine_epi64_epi8(a, _mm_set_epi32(0x80402010, 0x08040201, 0x80402010, 0x08040201), 0);
	a = _mm_sllv_epi8(a, count);
	a = _mm_gf2p8affine_epi64_epi8(a, _mm_set_epi32(0x80402010, 0x08040201, 0x80402010, 0x08040201), 0);
	return a;
	
	// if AVX512 is available, this is probably better
	__m128i mask = _mm_set1_epi16(0xff); // alternatively, this can be implemented using a mask register instead
	__m128i lo = _mm_srlv_epi16(_mm_and_si128(a, mask), _mm_and_si128(count, mask));
	__m128i hi = _mm_srlv_epi16(a, _mm_srli_epi16(count, 8));
	return _mm_ternarylogic_epi32(lo, hi, mask, 0xe4); // same as, but generally faster than, _mm_blendv_epi8(hi, lo, mask)
	
	/*
	vpcmpeqw xmm2, xmm2, xmm2
	vpsrlw xmm3, xmm1, 8
	vpsrlw xmm2, xmm2, 8
	vpsrlvw xmm3, xmm0, xmm3
	vpand xmm0, xmm0, xmm2
	vpand xmm1, xmm1, xmm2
	vpsrlvw xmm0, xmm0, xmm1
	vpternlogd xmm0, xmm3, xmm2, 0xe4
	 */
}

// bonus: variable rotates using AVX512
static inline
__m128i _mm_rorv_epi8(__m128i a, 
                      __m128i count) noexcept {
	// rotate by 4
	a = _mm_mask_gf2p8affine_epi64_epi8(
		a, _mm_test_epi8_mask(count, _mm_set1_epi8(4)),
		a, _mm_set_epi32(0x10204080, 0x01020408, 0x10204080, 0x01020408),
		0
	);
	// rotate by 2
	a = _mm_mask_gf2p8affine_epi64_epi8(
		a, _mm_test_epi8_mask(count, _mm_set1_epi8(2)),
		a, _mm_set_epi32(0x04081020, 0x40800102, 0x04081020, 0x40800102),
		0
	);
	// rotate by 1
	a = _mm_mask_gf2p8affine_epi64_epi8(
		a, _mm_test_epi8_mask(count, _mm_set1_epi8(1)),
		a, _mm_set_epi32(0x02040810, 0x20408001, 0x02040810, 0x20408001),
		0
	);
	return a;
	
	/*
	vptestmb k1, xmm1, [vec4]
	vgf2p8affineqb xmm0 {k1}, xmm0, [rot4], 0
	vptestmb k1, xmm1, [vec2]
	vgf2p8affineqb xmm0 {k1}, xmm0, [rot2], 0
	vptestmb k1, xmm1, [vec1]
	vgf2p8affineqb xmm0 {k1}, xmm0, [rot1], 0
	 */
	
	// is this better?
	__m128i lo = _mm_shuffle_epi8(a, _mm_set_epi32(0x0e0e0c0c, 0x0a0a0808, 0x06060404, 0x02020000));
	__m128i hi = _mm_shuffle_epi8(a, _mm_set_epi32(0x0f0d0d0b, 0x0b09090f, 0x07050503, 0x03010107));
	count = _mm_ternarylogic_epi32(count, _mm_set1_epi8(7), _mm_set_epi32(0x38302820, 0x18100800, 0x38302820, 0x18100800), 0xea); // (count & 7) | magic
	// if count is always <= 7, you can use the following line instead of the above
	//count = _mm_or_si128(count, _mm_set_epi32(0x38302820, 0x18100800, 0x38302820, 0x18100800));
	__m128i result = _mm_multishift_epi64_epi8(count, lo);
	return _mm_mask_multishift_epi64_epi8(result, 0xaaaa, count, hi);
	
	/*
	mov r8w, 0xaaaa
	kmovw k1, r8w
	vpshufb xmm2, xmm0, [shuf1]
	vmovdqa xmm3, [vec7]
	vpternlogd xmm1, xmm3, [positions]
	vpmultishiftqb xmm2, xmm1, xmm2
	vpshufb xmm0, xmm0, [shuf2]
	vpmultishiftqb xmm2 {k1}, xmm1, xmm0
	 */
}

static inline
__m128i _mm_rolv_epi8(__m128i a,
                      const __m128i count) noexcept {
	// rotate by 4
	a = _mm_mask_gf2p8affine_epi64_epi8(
		a, _mm_test_epi8_mask(count, _mm_set1_epi8(4)),
		a, _mm_set_epi32(0x10204080, 0x01020408, 0x10204080, 0x01020408),
		0
	);
	// rotate by 2
	a = _mm_mask_gf2p8affine_epi64_epi8(
		a, _mm_test_epi8_mask(count, _mm_set1_epi8(2)),
		a, _mm_set_epi32(0x40800102, 0x04081020, 0x40800102, 0x04081020),
		0
	);
	// rotate by 1
	a = _mm_mask_gf2p8affine_epi64_epi8(
		a, _mm_test_epi8_mask(count, _mm_set1_epi8(1)),
		a, _mm_set_epi32(0x80010204, 0x08102040, 0x80010204, 0x08102040),
		0
	);
	return a;
}
