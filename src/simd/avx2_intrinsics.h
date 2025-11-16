#include <immintrin.h>


static inline
__m128i _mm_adds_epi32(const __m128i a,
                       const __m128i b) noexcept {
	__m128i movwhdup	=	_mm_set_epi32(0x0f0e0f0e, 0x0b0a0b0a, 0x07060706, 0x03020302);
	__m128i _2x32768	=	_mm_set1_epi32(0x80008000);
	__m128i one			=	_mm_set1_epi32(0x1);
	__m128i b_high		=	_mm_shuffle_epi8(b, movwhdup);
	__m128i temp		=	_mm_dpwusds_epi32(a, _2x32768, b_high);
	__m128i zpn_one		=	_mm_sign_epi16(one, _mm_or_si128(b_high, one));
	return					_mm_dpwsuds_epi32(temp, zpn_one, b);
}

static inline
__m128i _mm_subs_epi32(const __m128i a,
                       const __m128i b) noexcept {
	__m128i movwhdup	=	_mm_set_epi32(0x0f0e0f0e, 0x0b0a0b0a, 0x07060706, 0x03020302);
	__m128i _m2x32768	=	_mm_set1_epi32(0x80008000);
	__m128i one			=	_mm_set1_epi32(0x1);
	__m128i m_one		=	_mm_set1_epi32(0xffff);
	__m128i b_high		=	_mm_shuffle_epi8(b, movwhdup);
	__m128i temp		=	_mm_dpwssds_avx_epi32(a, _m2x32768, b_high);
	__m128i zpn_one		=	_mm_sign_epi16(m_one, _mm_or_si128(b_high, one));
	return					_mm_dpwsuds_epi32(temp, zpn_one, b);
}

static inline
__m128i _mm_adds_epu32(const __m128i a,
                       const __m128i b) noexcept {
	return _mm_add_epi32(_mm_min_epu32(a, _mm_xor_si128(b, _mm_cmpeq_epi32(b, b))), b);
}

static inline
__m128i _mm_subs_epu32(const __m128i a,
                       const __m128i b) noexcept {
	return _mm_sub_epi32(_mm_max_epu32(a, b), b);
}

static inline
__m256i _mm256_adds_epi32(const __m256i a,
                          const __m256i b) noexcept {
	__m256i movwhdup	=	_mm256_set_epi32(0x0f0e0f0e, 0x0b0a0b0a, 0x07060706, 0x03020302, 0x0f0e0f0e, 0x0b0a0b0a, 0x07060706, 0x03020302);
	__m256i _2x32768	=	_mm256_set1_epi32(0x80008000);
	__m256i one			=	_mm256_set1_epi32(0x1);
	__m256i b_high		=	_mm256_shuffle_epi8(b, movwhdup);
	__m256i temp		=	_mm256_dpwusds_epi32(a, _2x32768, b_high);
	__m256i zpn_one		=	_mm256_sign_epi16(one, _mm256_or_si256(b_high, one));
	return					_mm256_dpwsuds_epi32(temp, zpn_one, b);
}

static inline
__m256i _mm256_subs_epi32(const __m256i a,
                          const __m256i b) noexcept {
	__m256i movwhdup	=	_mm256_set_epi32(0x0f0e0f0e, 0x0b0a0b0a, 0x07060706, 0x03020302, 0x0f0e0f0e, 0x0b0a0b0a, 0x07060706, 0x03020302);
	__m256i _m2x32768	=	_mm256_set1_epi32(0x80008000);
	__m256i one			=	_mm256_set1_epi32(0x1);
	__m256i m_one		=	_mm256_set1_epi32(0xffff);
	__m256i b_high		=	_mm256_shuffle_epi8(b, movwhdup);
	__m256i temp		=	_mm256_dpwssds_avx_epi32(a, _m2x32768, b_high);
	__m256i zpn_one		=	_mm256_sign_epi16(m_one, _mm256_or_si256(b_high, one));
	return					_mm256_dpwsuds_epi32(temp, zpn_one, b);
}

static inline
__m256i _mm256_adds_epu32(const __m256i a, 
                          const __m256i b) noexcept {
	return _mm256_add_epi32(_mm256_min_epu32(a, _mm256_xor_si256(b, _mm256_cmpeq_epi32(b, b))), b);
}

static inline
__m256i _mm256_subs_epu32(const __m256i a,
                          const __m256i b) noexcept {
	return _mm256_sub_epi32(_mm256_max_epu32(a, b), b);
}

/// source: https://arxiv.org/pdf/2112.06342
inline __m256i _mm256_2intersect_epi16_mask(const __m256i a,
                                            const __m256i b) {
	__m256i a1 = _mm256_permute2x128_si256(a, a, 1);
	__m256i b1 = _mm256_shuffle_epi32(b, _MM_PERM_ADCB);
	__m256i b2 = _mm256_shuffle_epi32(b, _MM_PERM_BADC);
	__m256i b3 = _mm256_shuffle_epi32(b, _MM_PERM_CBAD);
	__m256i b01 = _mm256_or_si256(_mm256_srli_epi32(b, 16), _mm256_slli_epi32(b, 16));
	__m256i b11 = _mm256_shuffle_epi32(b01, _MM_PERM_ADCB);
	__m256i b21 = _mm256_shuffle_epi32(b01, _MM_PERM_BADC);
	__m256i b31 = _mm256_shuffle_epi32(b01, _MM_PERM_CBAD);

	__m256i l1l = _mm256_or_si256(_mm256_cmpeq_epi16(a , b ), _mm256_cmpeq_epi16(a , b01));
	__m256i l1h = _mm256_or_si256(_mm256_cmpeq_epi16(a , b1), _mm256_cmpeq_epi16(a , b11));
	__m256i l2l = _mm256_or_si256(_mm256_cmpeq_epi16(a , b2), _mm256_cmpeq_epi16(a , b21));
	__m256i l2h = _mm256_or_si256(_mm256_cmpeq_epi16(a , b3), _mm256_cmpeq_epi16(a , b31));
	__m256i h1l = _mm256_or_si256(_mm256_cmpeq_epi16(a1, b ), _mm256_cmpeq_epi16(a1, b01));
	__m256i h1h = _mm256_or_si256(_mm256_cmpeq_epi16(a1, b1), _mm256_cmpeq_epi16(a1, b11));
	__m256i h2l = _mm256_or_si256(_mm256_cmpeq_epi16(a1, b2), _mm256_cmpeq_epi16(a1, b21));
	__m256i h2h = _mm256_or_si256(_mm256_cmpeq_epi16(a1, b3), _mm256_cmpeq_epi16(a1, b31));
	__m256i l1 = _mm256_or_si256(l1l, l1h);
	__m256i l2 = _mm256_or_si256(l2l, l2h);
	__m256i h1 = _mm256_or_si256(h1l, h1h);
	__m256i h2 = _mm256_or_si256(h2l, h2h);
	__m256i l = _mm256_or_si256(l1, l2);
	__m256i h = _mm256_or_si256(h1, h2);
	return _mm256_or_si256(l, _mm256_permute2x128_si256(h, h, 1));
}

/// source: https://arxiv.org/pdf/2112.06342
inline __m256i _mm256_2intersect_epi32_mask(const __m256i a,
                                            const __m256i b) {
	__m256i a1 = _mm256_permute2x128_si256(a, a, 1);
	__m256i b1 = _mm256_shuffle_epi32(b, _MM_PERM_ADCB);
	__m256i b2 = _mm256_shuffle_epi32(b, _MM_PERM_BADC);
	__m256i b3 = _mm256_shuffle_epi32(b, _MM_PERM_CBAD);

	__m256i ll = _mm256_or_si256(_mm256_cmpeq_epi32(a , b ), _mm256_cmpeq_epi32(a , b1));
	__m256i lh = _mm256_or_si256(_mm256_cmpeq_epi32(a , b2), _mm256_cmpeq_epi32(a , b3));
	__m256i hl = _mm256_or_si256(_mm256_cmpeq_epi32(a1, b ), _mm256_cmpeq_epi32(a1, b1));
	__m256i hh = _mm256_or_si256(_mm256_cmpeq_epi32(a1, b2), _mm256_cmpeq_epi32(a1, b3));
	__m256i l = _mm256_or_si256(ll, lh);
	__m256i h = _mm256_or_si256(hl, hh);
	return _mm256_or_si256(l, _mm256_permute2x128_si256(h, h, 1));
}

/// source: https://arxiv.org/pdf/2112.06342
inline __m256i _mm256_2intersect_epi64_mask(const __m256i a,
                                            const __m256i b) {
	__m256i a1 = _mm256_permute2x128_si256(a, a, 1);
	__m256i b1 = _mm256_shuffle_epi32(b, _MM_PERM_BADC);
	__m256i l = _mm256_or_si256(_mm256_cmpeq_epi64(a , b), _mm256_cmpeq_epi64(a , b1));
	__m256i h = _mm256_or_si256(_mm256_cmpeq_epi64(a1, b), _mm256_cmpeq_epi64(a1, b1));
	return _mm256_or_si256(l, _mm256_permute2x128_si256(h, h, 1));
}
