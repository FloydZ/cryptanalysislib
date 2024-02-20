#ifndef CRYPTANALYSISLIB_FLOAT_SIMD_H
#define CRYPTANALYSISLIB_FLOAT_SIMD_H

#include <cmath>
#include <cstdint>
struct f32x8_t {
	union {
		float f[8];
		double d[4];
	};

	constexpr f32x8_t() {
		for (uint32_t i = 0; i < 8u; ++i) {
			f[i] = 0.f;
		}
	}

	constexpr f32x8_t(const uint32x8_t in) {
		for (uint32_t i = 0; i < 8u; ++i) {
			f[i] = (float) in.v32[i];
		}
	}

	constexpr static uint32x8_t uint32x8(const f32x8_t in) {
		uint32x8_t ret;
		for (uint32_t i = 0; i < 8u; ++i) {
			ret.v32[i] = std::floor(in.f[i]);
		}

		return ret;
	}

	constexpr static f32x8_t floor(const f32x8_t in) {
		f32x8_t ret;
		for (uint32_t i = 0; i < 8u; ++i) {
			ret.f[i] = std::floor(in.f[i]);
		}

		return ret;
	}

	constexpr static f32x8_t sqrt(const f32x8_t in) {
		f32x8_t ret;
		for (uint32_t i = 0; i < 8u; ++i) {
			ret.f[i] = std::sqrt(in.f[i]);
		}

		return ret;
	}
};


struct f64x4_t {
	union {
		float f[8];
		double d[4];
	};

	constexpr f64x4_t() {
		for (uint32_t i = 0; i < 4u; ++i) {
			d[i] = 0.;
		}
	}

	constexpr f64x4_t(const uint64x4_t in) {
		for (uint32_t i = 0; i < 4u; ++i) {
			d[i] = (float) in.v64[i];
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
		for (uint32_t i = 0; i < 4u; ++i) {
			ret.d[i] = std::floor(in.d[i]);
		}

		return ret;
	}

	constexpr static f64x4_t sqrt(const f64x4_t in) {
		f64x4_t ret;
		for (uint32_t i = 0; i < 4u; ++i) {
			ret.d[i] = std::sqrt(in.d[i]);
		}

		return ret;
	}
};
#endif//CRYPTANALYSISLIB_FLOAT_SIMD_H
