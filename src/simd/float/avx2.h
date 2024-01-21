#ifndef CRYPTANALYSISLIB_FLOAT_AVX2_H
#define CRYPTANALYSISLIB_FLOAT_AVX2_H

#ifndef USE_AVX2
#error "no avx"
#endif

#include <cstdint>
#include <cstdio>
#include <immintrin.h>

struct f32x8_t {
	union {
		float  f[8];
		double d[4];
		__m256 v;
	};

	constexpr f32x8_t() {
		for (uint32_t i = 0; i < 8u; ++i) {
			f[i] = 0.f;
		}
	}

	constexpr f32x8_t(const uint32x8_t in) {
		v = _mm256_cvtepi32_ps(in.v256);
	}

	constexpr static uint32x8_t uint32x8(const f32x8_t in) {
		uint32x8_t ret;
		ret.v256 = _mm256_cvtps_epi32(in.v);
		return ret;
	}

	constexpr static f32x8_t floor(const f32x8_t in) {
		f32x8_t ret;
		ret.v = _mm256_floor_ps(in.v);
		return ret;
	}

	constexpr static f32x8_t sqrt(const f32x8_t in) {
		f32x8_t ret;
		ret.v = _mm256_sqrt_ps(in.v);
		return ret;
	}
};


struct f64x4_t {
	union {
		float  	f[8];
		double 	d[4];
		__m256d v;
	};

	constexpr f64x4_t() {
		for (uint32_t i = 0; i < 4u; ++i) {
			d[i] = 0.;
		}
	}

	constexpr f64x4_t(const uint64x4_t in) {
		for (uint32_t i = 0; i < 4u; ++i) {
			d[i] = (double)in.v64[i];
		}
	}

	constexpr static uint64x4_t uint64x4(const f64x4_t in) {
		uint64x4_t ret;
		for (uint32_t i = 0; i < 4u; ++i) {
			ret.v64[i] = std::floor(in.d[i]);
		}

		return ret;
	}

	constexpr static f64x4_t floor(const f64x4_t in) {
		f64x4_t ret;
		ret.v = _mm256_floor_pd(in.v);
		return ret;
	}

	constexpr static f64x4_t sqrt(const f64x4_t in) {
		f64x4_t ret;
		ret.v = _mm256_sqrt_pd(in.v);
		return ret;
	}
};
#endif//CRYPTANALYSISLIB_FLOAT_AVX2_H
