#ifndef CRYPTANALYSISLIB_SIMD_GENERIC_H
#define CRYPTANALYSISLIB_SIMD_GENERIC_H

#ifndef CRYPTANALYSISLIB_SIMD_H
#error "dont inlcude this file. Use `#include \"simd/simd.h\"`."
#endif
#include <type_traits>

#include "popcount/popcount.h"
#include "random.h"
#include "simd/simd.h"
using namespace cryptanalysislib;

///
template<typename T, const uint32_t N>
requires std::is_integral_v<T>
class TxN_t {
public:
	constexpr static size_t lb = sizeof(T);
	constexpr static size_t limb_bits = sizeof(T) * 8;
	constexpr static size_t total_bits = limb_bits * N;
	constexpr static size_t simd256_limbs = total_bits / 256;
	constexpr static size_t simd512_limbs = total_bits / 512;

	constexpr static uint32_t nr_limbs_in_simd256 = 256 / limb_bits;
	constexpr static uint32_t nr_limbs_in_simd512 = 512 / limb_bits;

	constexpr static bool simd256_enable = simd256_limbs > 0;
#ifdef USE_AVX512
	constexpr static bool simd512_enable = simd512_limbs > 0;
#else 
	constexpr static bool simd512_enable = false;
#endif

	// if true the size of the datastruct is a multiple of simd limbs
	// thus special unrolling etc can be applied
	constexpr static bool simd256_fits = simd256_limbs * 256 == total_bits;
	constexpr static bool simd512_fits = simd512_limbs * 512 == total_bits;

	using data_type = T;
	using simd256_type =
	        typename std::conditional<lb == 1u, uint32x8_t,
	                                  typename std::conditional<lb == 2u, uint16x16_t,
	                                                            typename std::conditional<lb == 4u, uint32x8_t,
	                                                                                      typename std::conditional<lb == 8u, uint64x4_t, void>::type>::type>::type>::type;

#ifdef USE_AVX512
	using simd512_type =
	        typename std::conditional<lb == 1u, uint64x8_t,
	                                  typename std::conditional<lb == 2u, uint32x16_t,
	                                                            typename std::conditional<lb == 4u, uint16x32_t,
	                                                                                      typename std::conditional<lb == 8u, uint64x8_t, void>::type>::type>::type>::type;
#else 
	using simd512_type = sisimd256_type;
#endif

	union {
		data_type d[N];
		simd256_type v256[simd256_limbs];
		simd512_type v512[simd512_limbs];
	};

	constexpr inline void print(bool binary, bool hex) const noexcept {
		if (binary) {
			for (uint32_t i = 0; i < N; i++) {
				print_binary(this->d[i], limb_bits);
			}

			return;
		}

		if (hex) {
			for (uint32_t i = 0; i < N; i++) {
				printf("%hhx ", this->d[i]);
			}

			return;
		}

		for (uint32_t i = 0; i < N; i++) {
			printf("%u ", this->d[i]);
		}
		printf("\n");
	}
	static inline TxN_t random() noexcept {
		TxN_t ret;
		for (uint32_t i = 0; i < N; i++) {
			ret.d[i] = fastrandombytes_uint64();
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline TxN_t set(const T *data) noexcept {
		ASSERT(data);
		TxN_t ret;
		for (uint32_t i = 0; i < N; i++) {
			ret.d[i] = data[N - i - 1];
		}
		return ret;
	}

	[[nodiscard]] constexpr static inline TxN_t setr(const T *data) noexcept {
		ASSERT(data);
		TxN_t ret;
		for (uint32_t i = 0; i < N; i++) {
			ret.d[i] = data[i];
		}
		return ret;
	}

	[[nodiscard]] constexpr static inline TxN_t set1(const T data) noexcept {
		TxN_t ret;
		for (uint32_t i = 0; i < N; i++) {
			ret.d[i] = data;
		}
		return ret;
	}

	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline TxN_t load(const void *ptr) {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline TxN_t aligned_load(const void *ptr) noexcept {
		TxN_t ret;

		T *ptrT = (T *) ptr;
		uint32_t i = 0;
		if constexpr (simd512_enable) {
			for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
				ret.v512[i / nr_limbs_in_simd512] = simd512_type::aligned_load(ptrT + i);
			}

			if constexpr (simd512_fits) {
				return ret;
			}
		}

		if constexpr (simd256_enable) {
			for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
				ret.v256[i / nr_limbs_in_simd256] = simd256_type::aligned_load(ptrT + i);
			}

			if constexpr (simd256_fits) {
				return ret;
			}
		}

		for (; i < N; ++i) {
			ret.d[i] = ptrT[i];
		}

		return ret;
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline TxN_t unaligned_load(const void *ptr) noexcept {
		TxN_t ret;

		T *ptrT = (T *) ptr;
		uint32_t i = 0;
		if constexpr (simd512_enable) {
			for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
				ret.v512[i / nr_limbs_in_simd512] = simd512_type::unaligned_load(ptrT + i);
			}

			if constexpr (simd512_fits) {
				return ret;
			}
		}

		if constexpr (simd256_enable) {
			for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
				ret.v256[i / nr_limbs_in_simd256] = simd256_type::unaligned_load(ptrT + i);
			}

			if constexpr (simd256_fits) {
				return ret;
			}
		}

		for (; i < N; ++i) {
			ret.d[i] = ptrT[i];
		}

		return ret;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(void *ptr, const TxN_t &in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		unaligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(void *ptr, const TxN_t &in) noexcept {
		T *ptrT = (T *) ptr;
		uint32_t i = 0;
		if constexpr (simd512_enable) {
			for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
				simd512_type::aligned_store(ptrT + i, in.v512[i / nr_limbs_in_simd512]);
			}

			if constexpr (simd512_fits) {
				return;
			}
		}

		if constexpr (simd256_enable) {
			for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
				simd256_type::aligned_store(ptrT + i, in.v256[i / nr_limbs_in_simd256]);
			}

			if constexpr (simd256_fits) {
				return;
			}
		}

		for (; i < N; ++i) {
			ptrT[i] = in.d[i];
		}
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const TxN_t in) noexcept {
		T *ptrT = (T *) ptr;
		uint32_t i = 0;
		if constexpr (simd512_enable) {
			for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
				simd512_type::unaligned_store(ptrT + i, in.v512[i / nr_limbs_in_simd512]);
			}

			if constexpr (simd512_fits) {
				return;
			}
		}

		if constexpr (simd256_enable) {
			for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
				simd256_type::unaligned_store(ptrT + i, in.v256[i / nr_limbs_in_simd256]);
			}

			if constexpr (simd256_fits) {
				return;
			}
		}

		for (; i < N; ++i) {
			ptrT[i] = in.d[i];
		}
	}


	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline TxN_t xor_(const TxN_t in1,
	                                                 const TxN_t in2) noexcept {
		TxN_t out;
		uint32_t i = 0;
		if constexpr (simd512_enable) {
			for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
				out.v512[i / nr_limbs_in_simd512] = in1.v512[i / nr_limbs_in_simd512] ^ in2.v512[i / nr_limbs_in_simd512];
			}

			if constexpr (simd512_fits) {
				return out;
			}
		}

		if constexpr (simd256_enable) {
			for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
				out.v256[i / nr_limbs_in_simd256] = in1.v256[i / nr_limbs_in_simd512] ^ in2.v256[i / nr_limbs_in_simd256];
			}

			if constexpr (simd256_fits) {
				return out;
			}
		}

		for (; i < N; ++i) {
			out.d[i] = in1.d[i] ^ in2.d[i];
		}
		return out;
	}
};


///
template<typename T, const uint32_t N>
inline TxN_t<T, N> operator*(const TxN_t<T, N> &lhs, const TxN_t<T, N> &rhs) {
	return TxN_t<T, N>::mullo(lhs, rhs);
}
template<typename T, const uint32_t N>
inline TxN_t<T, N> operator*(const TxN_t<T, N> &lhs, const uint8_t &rhs) {
	return TxN_t<T, N>::mullo(lhs, rhs);
}
template<typename T, const uint32_t N>
inline TxN_t<T, N> operator*(const uint8_t &lhs, const TxN_t<T, N> &rhs) {
	return TxN_t<T, N>::mullo(rhs, lhs);
}
template<typename T, const uint32_t N>
inline TxN_t<T, N> operator+(const TxN_t<T, N> &lhs, const TxN_t<T, N> &rhs) {
	return TxN_t<T, N>::add(lhs, rhs);
}
template<typename T, const uint32_t N>
inline TxN_t<T, N> operator-(const TxN_t<T, N> &lhs, const TxN_t<T, N> &rhs) {
	return TxN_t<T, N>::sub(lhs, rhs);
}
template<typename T, const uint32_t N>
inline TxN_t<T, N> operator&(const TxN_t<T, N> &lhs, const TxN_t<T, N> &rhs) {
	return TxN_t<T, N>::and_(lhs, rhs);
}
template<typename T, const uint32_t N>
inline TxN_t<T, N> operator^(const TxN_t<T, N> &lhs, const TxN_t<T, N> &rhs) {
	return TxN_t<T, N>::xor_(lhs, rhs);
}
template<typename T, const uint32_t N>
inline TxN_t<T, N> operator|(const TxN_t<T, N> &lhs, const TxN_t<T, N> &rhs) {
	return TxN_t<T, N>::or_(lhs, rhs);
}
template<typename T, const uint32_t N>
inline TxN_t<T, N> operator~(const TxN_t<T, N> &lhs) {
	return TxN_t<T, N>::not_(lhs);
}
//template<typename T, const uint32_t N>
//inline TxN_t<T, N> operator>> (const TxN_t<T, N>& lhs, const uint32_t rhs) {
//	return TxN_t<T, N>::srli(lhs, rhs);
//}
//template<typename T, const uint32_t N>
//inline TxN_t<T, N> operator<< (const TxN_t<T, N>& lhs, const uint32_t rhs) {
//	return TxN_t<T, N>::slli(lhs, rhs);
//}
template<typename T, const uint32_t N>
inline TxN_t<T, N> operator^=(TxN_t<T, N> &lhs, const TxN_t<T, N> &rhs) {
	lhs = TxN_t<T, N>::xor_(lhs, rhs);
	return lhs;
}
template<typename T, const uint32_t N>
inline TxN_t<T, N> operator&=(TxN_t<T, N> &lhs, const TxN_t<T, N> &rhs) {
	lhs = TxN_t<T, N>::and_(lhs, rhs);
	return lhs;
}
template<typename T, const uint32_t N>
inline TxN_t<T, N> operator|=(TxN_t<T, N> &lhs, const TxN_t<T, N> &rhs) {
	lhs = TxN_t<T, N>::or_(lhs, rhs);
	return lhs;
}
#endif//CRYPTANALYSISLIB_GENERIC_H
