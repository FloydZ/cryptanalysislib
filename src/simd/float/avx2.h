#ifndef CRYPTANALYSISLIB_FLOAT_AVX2_H
#define CRYPTANALYSISLIB_FLOAT_AVX2_H

#ifndef USE_AVX2
#error "no avx"
#endif

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <immintrin.h>

struct f32x8_t {
	union {
		float  f[8];
		double d[4];
		__m256 v;
	};

	constexpr inline f32x8_t() noexcept {
		for (uint32_t i = 0; i < 8u; ++i) {
			f[i] = 0.f;
		}
	}

	constexpr inline f32x8_t(const uint32x8_t in) noexcept {
#ifdef __clang__
  		v = (__m256)__builtin_convertvector((__v8si)in.v256, __v8sf);
#else
  		v = (__m256)__builtin_ia32_cvtdq2ps256 ((__v8si) in.v256);
#endif
	}

	constexpr static inline uint32x8_t uint32x8(const f32x8_t in) noexcept {
		uint32x8_t ret;
  		ret.v256 = (__m256i)__builtin_ia32_cvtps2dq256 ((__v8sf) in.v);
		return ret;
	}

	constexpr static inline f32x8_t floor(const f32x8_t in) noexcept {
		f32x8_t ret;
#ifdef __clang__
		ret.v = _mm256_floor_ps(in.v);
#else
		ret.v = ((__m256) __builtin_ia32_roundps256 ((__v8sf)(__m256)(in.v), (int)(_MM_FROUND_FLOOR)));
#endif
		return ret;
	}

	constexpr static inline f32x8_t sqrt(const f32x8_t in) noexcept {
		f32x8_t ret;
  		ret.v = (__m256) __builtin_ia32_sqrtps256 ((__v8sf)in.v);
		return ret;
	}
};


struct f64x4_t {
	union {
		float  	f[8];
		double 	d[4];
		__m256d v;
	};

	constexpr inline f64x4_t() noexcept {
		for (uint32_t i = 0; i < 4u; ++i) {
			d[i] = 0.;
		}
	}

	constexpr inline f64x4_t(const uint64x4_t in) noexcept {
		for (uint32_t i = 0; i < 4u; ++i) {
			d[i] = (double)in.v64[i];
		}
	}

	constexpr static inline uint64x4_t uint64x4(const f64x4_t in) noexcept {
		uint64x4_t ret;
		for (uint32_t i = 0; i < 4u; ++i) {
			ret.v64[i] = std::floor(in.d[i]);
		}

		return ret;
	}

	constexpr static inline f64x4_t floor(const f64x4_t in) noexcept {
		f64x4_t ret;
#ifdef __clang__
		ret.v = _mm256_floor_pd(in.v);
#else
  		ret.v = ((__m256d) __builtin_ia32_roundpd256 ((__v4df)(__m256d)(in.v), (int)(_MM_FROUND_FLOOR)));
#endif
		return ret;
	}

	constexpr static inline f64x4_t sqrt(const f64x4_t in) noexcept {
		f64x4_t ret;

  		ret.v = (__m256d) __builtin_ia32_sqrtpd256 ((__v4df)in.v);
		return ret;
	}
};
#endif//CRYPTANALYSISLIB_FLOAT_AVX2_H
