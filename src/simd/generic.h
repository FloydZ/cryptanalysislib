#ifndef CRYPTANALYSISLIB_SIMD_GENERIC_H
#define CRYPTANALYSISLIB_SIMD_GENERIC_H

#ifndef CRYPTANALYSISLIB_SIMD_H
#error "dont inlcude this file. Use `#include \"simd/simd.h\"`."
#endif

#include <type_traits>

#include "algorithm/bits/popcount.h"
#include "helper.h"
#include "random.h"
#include "simd/simd.h"
using namespace cryptanalysislib;


/// NOTE: the SIMD parts call the named functions (`simd256_type::NAME`)
/// 	instead of the operators, which only exist for the unsigned types
#define SIMD_ARITH_MACRO(OPERATION, NAME) 										\
[[nodiscard]] constexpr static inline TxN_t NAME(const TxN_t &in1,				\
												 const TxN_t &in2) noexcept {	\
	TxN_t out;																	\
	if (std::is_constant_evaluated()) {											\
		for (uint32_t i = 0; i < N; ++i) {										\
			out.d[i] = in1.d[i] OPERATION in2[i];								\
		}																		\
		return out;																\
	}																			\
	uint32_t i = 0;																\
	if constexpr (simd512_enable) {												\
		for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {		\
			out.v512[i / nr_limbs_in_simd512] = simd512_type::NAME(in1.v512[i / nr_limbs_in_simd512], in2.v512[i / nr_limbs_in_simd512]);\
		}																		\
		if constexpr (simd512_fits) {											\
			return out;															\
		}																		\
	}																			\
	if constexpr (simd256_enable) {												\
		for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {		\
			out.v256[i / nr_limbs_in_simd256] = simd256_type::NAME(in1.v256[i / nr_limbs_in_simd256], in2.v256[i / nr_limbs_in_simd256]);\
		}																		\
		if constexpr (simd256_fits) {											\
			return out;															\
		}																		\
	}																			\
	for (; i < N; ++i) {														\
		out.d[i] = in1.d[i] OPERATION in2.d[i];									\
	}																			\
	return out;																	\
}

/// NOTE: this functions expands the result of a simple integer operation
/// to all bits within this limb
#define SIMD_FUNCTION_2ARGS_MACRO(OPERATION, INT_OPERATION) 					\
[[nodiscard]] constexpr static inline TxN_t OPERATION(const TxN_t &in1,			\
												 const TxN_t &in2) noexcept {	\
	TxN_t out;																	\
	if (std::is_constant_evaluated()) {											\
		for (uint32_t i = 0; i < N; ++i) {										\
			out.d[i] = (in1.d[i] INT_OPERATION in2[i]) * -1ull;					\
		}																		\
		return out;																\
	}																			\
	uint32_t i = 0;																\
	if constexpr (simd512_enable) {												\
		for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {		\
			out.v512[i / nr_limbs_in_simd512] = simd512_type::OPERATION(in1.v512[i / nr_limbs_in_simd512], in2.v512[i / nr_limbs_in_simd512]);\
		}																		\
		if constexpr (simd512_fits) {											\
			return out;															\
		}																		\
	}																			\
	if constexpr (simd256_enable) {												\
		for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {		\
			out.v256[i / nr_limbs_in_simd256] = simd256_type::OPERATION(in1.v256[i / nr_limbs_in_simd256], in2.v256[i / nr_limbs_in_simd256]);\
		}																		\
		if constexpr (simd256_fits) {											\
			return out;															\
		}																		\
	}                                                         					\
	for (; i < N; ++i) {														\
		out.d[i] = (in1.d[i] INT_OPERATION in2.d[i]) * -1ull;					\
	}																			\
	return out;																	\
}



/// NOTE: the signed will be extracted from T
template<typename T,
         const uint32_t N>
#if __cplusplus > 201709L
    requires std::is_integral_v<T>
#endif
class TxN_t {
public:
	constexpr static uint32_t LIMBS = N;
    constexpr static bool __unsigned = std::is_unsigned_v<T>;
	using limb_type = T;
	using S = TxN_t<T, N>;
	/// the shifts/rotations are computed in the unsigned type (logical shifts)
	using U = std::make_unsigned_t<T>;

	static_assert(N > 0);

	constexpr static std::size_t lb = sizeof(T);
	constexpr static std::size_t limb_bits = sizeof(T) * 8;
	constexpr static std::size_t total_bits = limb_bits * N;
	constexpr static std::size_t simd256_limbs = total_bits / 256;
	constexpr static std::size_t simd512_limbs = total_bits / 512;

	constexpr static uint32_t nr_limbs_in_simd256 = 256 / limb_bits;
	constexpr static uint32_t nr_limbs_in_simd512 = 512 / limb_bits;

	constexpr static bool simd256_enable = simd256_limbs > 0;

#ifdef USE_AVX512F
	constexpr static bool simd512_enable = simd512_limbs > 0;
#else
	constexpr static bool simd512_enable = false;
#endif

	// if true the size of the datastructures is a multiple of simd limbs
	// thus special unrolling etc. can be applied
	constexpr static bool simd256_fits = simd256_limbs * 256 == total_bits;
	constexpr static bool simd512_fits = simd512_limbs * 512 == total_bits;

	using data_type = T;
	using simd256_type =
	   typename std::conditional<lb == 1u, Xint8x32_t<__unsigned>,
	      typename std::conditional<lb == 2u, Xint16x16_t<__unsigned>,
	         typename std::conditional<lb == 4u, Xint32x8_t<__unsigned>,
	            typename std::conditional<lb == 8u, Xint64x4_t<__unsigned>, void>::type>::type>::type>::type;

#ifdef USE_AVX512F
	using simd512_type =
	   typename std::conditional<lb == 1u, Xint8x64_t<__unsigned>,
	      typename std::conditional<lb == 2u, Xint16x32_t<__unsigned>,
	         typename std::conditional<lb == 4u, Xint32x16_t<__unsigned>,
	            typename std::conditional<lb == 8u, Xint64x8_t<__unsigned>, void>::type>::type>::type>::type;
#else
	/// just a dummy value
	using simd512_type = simd256_type;
#endif

	constexpr static uint32_t simd256_bits = sizeof(simd256_type) * 8;
	constexpr static uint32_t simd512_bits = sizeof(simd512_type) * 8;

	union {
		data_type d[N];
		simd256_type v256[simd256_limbs];
		simd512_type v512[simd512_limbs];
	};

	constexpr inline TxN_t() noexcept = default;

	//
	constexpr inline TxN_t(const std::array<T, N> &array) noexcept {
		for (uint32_t i = 0; i < N; ++i) {
			d[i] = array[i];
		}
	}


	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	[[nodiscard]] constexpr inline limb_type& operator[](const uint32_t i) noexcept {
		assert(i < LIMBS);
		return d[i];
	}

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

	[[nodiscard]] static inline TxN_t random() noexcept {
		TxN_t ret{};
		for (uint32_t i = 0; i < N; i++) {
			ret.d[i] = rng();
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline TxN_t setr(const T *data) noexcept {
		assert(data);
		TxN_t ret;
		for (uint32_t i = 0; i < N; i++) {
			ret.d[i] = data[N - i - 1];
		}
		return ret;
	}

	[[nodiscard]] constexpr static inline TxN_t set(const T *data) noexcept {
		assert(data);
		TxN_t ret;
		for (uint32_t i = 0; i < N; i++) {
			ret.d[i] = data[i];
		}
		return ret;
	}

	[[nodiscard]] constexpr static inline TxN_t set1(const T data) noexcept {
		TxN_t ret{};
		for (uint32_t i = 0; i < N; i++) {
			ret.d[i] = data;
		}
		return ret;
	}

	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline TxN_t load(const T *ptr) {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline TxN_t aligned_load(const T *ptr) noexcept {
		TxN_t ret;
		if (std::is_constant_evaluated()) {
			for (uint32_t i = 0; i < LIMBS; ++i) {
				ret.d[i] = ptr[i];
			}
			return ret;
		}

		uint32_t i = 0;
		if constexpr (simd512_enable) {
			for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
				ret.v512[i / nr_limbs_in_simd512] = simd512_type::aligned_load((const typename simd512_type::limb_type *)(ptr + i));
			}

			if constexpr (simd512_fits) {
				return ret;
			}
		}

		if constexpr (simd256_enable) {
			for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
				ret.v256[i / nr_limbs_in_simd256] = simd256_type::aligned_load((const typename simd256_type::limb_type *)(ptr + i));
			}

			if constexpr (simd256_fits) {
				return ret;
			}
		}

		for (; i < N; ++i) {
			ret.d[i] = ptr[i];
		}

		return ret;
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline TxN_t unaligned_load(const T *ptr) noexcept {
		TxN_t ret;
		if (std::is_constant_evaluated()) {
			for (uint32_t i = 0; i < LIMBS; ++i) {
				ret.d[i] = ptr[i];
			}
			return ret;
		}

		uint32_t i = 0;
		if constexpr (simd512_enable) {
			for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
				ret.v512[i / nr_limbs_in_simd512] = simd512_type::unaligned_load((const typename simd512_type::limb_type *)(ptr + i));
			}

			if constexpr (simd512_fits) {
				return ret;
			}
		}

		if constexpr (simd256_enable) {
			for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
				ret.v256[i / nr_limbs_in_simd256] = simd256_type::unaligned_load((const typename simd256_type::limb_type *)(ptr + i));
			}

			if constexpr (simd256_fits) {
				return ret;
			}
		}

		for (; i < N; ++i) {
			ret.d[i] = ptr[i];
		}

		return ret;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(T *ptr, const TxN_t &in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		unaligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(T *ptr, const TxN_t &in) noexcept {
		uint32_t i = 0;
		if constexpr (simd512_enable) {
			for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
				simd512_type::aligned_store((typename simd512_type::limb_type *)(ptr + i), in.v512[i / nr_limbs_in_simd512]);
			}

			if constexpr (simd512_fits) {
				return;
			}
		}

		if constexpr (simd256_enable) {
			for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
				simd256_type::aligned_store((typename simd256_type::limb_type *)(ptr + i), in.v256[i / nr_limbs_in_simd256]);
			}

			if constexpr (simd256_fits) {
				return;
			}
		}

		for (; i < N; ++i) {
			ptr[i] = in.d[i];
		}
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(T *ptr, 
                                                 const TxN_t &in) noexcept {
		uint32_t i = 0;
		if constexpr (simd512_enable) {
			for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
				simd512_type::unaligned_store((typename simd512_type::limb_type *)(ptr + i), in.v512[i / nr_limbs_in_simd512]);
			}

			if constexpr (simd512_fits) {
				return;
			}
		}

		if constexpr (simd256_enable) {
			for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
				simd256_type::unaligned_store((typename simd256_type::limb_type *)(ptr + i), in.v256[i / nr_limbs_in_simd256]);
			}

			if constexpr (simd256_fits) {
				return;
			}
		}

		for (; i < N; ++i) {
			ptr[i] = in.d[i];
		}
	}


	SIMD_ARITH_MACRO(^, xor_)
	SIMD_ARITH_MACRO(&, and_)
	SIMD_ARITH_MACRO(|, or_)
	SIMD_ARITH_MACRO(+, add)
	SIMD_ARITH_MACRO(-, sub)
	SIMD_ARITH_MACRO(*, mullo)


	/// \param in1
	/// \param in2 scalar
	/// \return [in1[0]*in2, ..., in1[N-1]*in2] (lower half of each product)
	[[nodiscard]] constexpr static inline TxN_t mullo(const TxN_t &in1,
	                                                  const T in2) noexcept {
		return mullo(in1, set1(in2));
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline TxN_t andnot_(const TxN_t &in1, 
                                                        const TxN_t in2) noexcept {
		TxN_t ret;
		if (std::is_constant_evaluated()) {
			for (uint32_t i = 0; i < LIMBS; ++i) {
				ret.d[i] = ~in1[i] & in2[i];
			}
			return ret;
		}

		uint32_t i = 0;
		if constexpr (simd512_enable) {
			for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
				ret.v512[i / nr_limbs_in_simd512] = simd512_type::andnot(in1.v512[i / nr_limbs_in_simd512], in2.v512[i / nr_limbs_in_simd512]);
			}

			if constexpr (simd512_fits) {
				return ret;
			}
		}

		if constexpr (simd256_enable) {
			for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
				ret.v256[i / nr_limbs_in_simd256] = simd256_type::andnot(in1.v256[i / nr_limbs_in_simd256], in2.v256[i / nr_limbs_in_simd256]);
			}

			if constexpr (simd256_fits) {
				return ret;
			}
		}

		for (; i < N; ++i) {
			ret.d[i] = ~in1.d[i] & in2[i];
		}

		return ret;
	}

	/// \return (~in1) & in2, same as `andnot_` (the name used by all backends)
	[[nodiscard]] constexpr static inline TxN_t andnot(const TxN_t &in1,
	                                                   const TxN_t &in2) noexcept {
		return andnot_(in1, in2);
	}

	/// double width type for the products in `mul`/`mulhi`
	using mul_type = std::conditional_t<sizeof(T) == 8,
	                                    std::conditional_t<std::is_signed_v<T>, __int128, unsigned __int128>,
	                                    std::conditional_t<std::is_signed_v<T>, int64_t, uint64_t>>;

	/// \param in1
	/// \param in2
	/// \return [in1[0]*in2[0], ..., in1[N-1]*in2[N-1]] (lower half of each product)
	[[nodiscard]] constexpr static inline TxN_t mul(const TxN_t &in1, const TxN_t in2) noexcept {
		TxN_t ret;
		for (uint32_t i = 0; i < N; ++i) {
			ret.d[i] = (T)((mul_type)in1.d[i] * (mul_type)in2.d[i]);
		}
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return upper half of each product in1[i]*in2[i]
	[[nodiscard]] constexpr static inline TxN_t mulhi(const TxN_t &in1, const TxN_t in2) noexcept {
		TxN_t ret;
		for (uint32_t i = 0; i < N; ++i) {
			ret.d[i] = (T)(((mul_type)in1.d[i] * (mul_type)in2.d[i]) >> (sizeof(T) * 8u));
		}
		return ret;
	}


	///
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline TxN_t not_(const TxN_t &in) noexcept {
		TxN_t ret;
		if (std::is_constant_evaluated()) {
			for (uint32_t i = 0; i < LIMBS; ++i) {
				ret.d[i] = ~in[i];
			}
			return ret;
		}

		uint32_t i = 0;
		if constexpr (simd512_enable) {
			for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
				ret.v512[i / nr_limbs_in_simd512] = simd512_type::not_(in.v512[i / nr_limbs_in_simd512]);
			}

			if constexpr (simd512_fits) {
				return ret;
			}
		}

		if constexpr (simd256_enable) {
			for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
				ret.v256[i / nr_limbs_in_simd256] = simd256_type::not_(in.v256[i / nr_limbs_in_simd256]);
			}

			if constexpr (simd256_fits) {
				return ret;
			}
		}

		for (; i < N; ++i) {
			ret.d[i] = ~in.d[i];
		}

		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline TxN_t slli(const TxN_t &in1, const uint32_t in2) noexcept {
		TxN_t ret;
		if (std::is_constant_evaluated()) {
			for (uint32_t i = 0; i < LIMBS; ++i) {
				ret.d[i] = (T)((U)in1[i] << in2);
			}
			return ret;
		}

		uint32_t i = 0;
		if constexpr (simd512_enable) {
			for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
				ret.v512[i / nr_limbs_in_simd512] = simd512_type::slli(in1.v512[i / nr_limbs_in_simd512], in2);
			}

			if constexpr (simd512_fits) {
				return ret;
			}
		}

		if constexpr (simd256_enable) {
			for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
				ret.v256[i / nr_limbs_in_simd256] = simd256_type::slli(in1.v256[i / nr_limbs_in_simd256], in2);
			}

			if constexpr (simd256_fits) {
				return ret;
			}
		}

		for (; i < N; ++i) {
			ret.d[i] = (T)((U)in1.d[i] << in2);
		}

		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline TxN_t srli(const TxN_t &in1, const uint32_t in2) noexcept {
		TxN_t ret;
		if (std::is_constant_evaluated()) {
			for (uint32_t i = 0; i < LIMBS; ++i) {
				ret.d[i] = (T)((U)in1[i] >> in2);
			}
			return ret;
		}

		uint32_t i = 0;
		if constexpr (simd512_enable) {
			for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
				ret.v512[i / nr_limbs_in_simd512] = simd512_type::srli(in1.v512[i / nr_limbs_in_simd512], in2);
			}

			if constexpr (simd512_fits) {
				return ret;
			}
		}

		if constexpr (simd256_enable) {
			for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
				ret.v256[i / nr_limbs_in_simd256] = simd256_type::srli(in1.v256[i / nr_limbs_in_simd256], in2);
			}

			if constexpr (simd256_fits) {
				return ret;
			}
		}

		for (; i < N; ++i) {
			ret.d[i] = (T)((U)in1.d[i] >> in2);
		}

		return ret;
	}


	[[nodiscard]] constexpr static inline TxN_t ror(const TxN_t &in1, const uint32_t in2) noexcept {
		constexpr uint32_t bits = sizeof(T) * 8u;
		const uint32_t s = in2 % bits;
		TxN_t ret;
		for (uint32_t i = 0; i < N; ++i) {
			const U x = (U)in1.d[i];
			ret.d[i] = s ? T((U)(x >> s) | (U)(x << (bits - s))) : in1.d[i];
		}
		return ret;
	}
	[[nodiscard]] constexpr static inline TxN_t rol(const TxN_t &in1, const uint32_t in2) noexcept {
		constexpr uint32_t bits = sizeof(T) * 8u;
		const uint32_t s = in2 % bits;
		TxN_t ret;
		for (uint32_t i = 0; i < N; ++i) {
			const U x = (U)in1.d[i];
			ret.d[i] = s ? T((U)(x << s) | (U)(x >> (bits - s))) : in1.d[i];
		}
		return ret;
	}



	SIMD_FUNCTION_2ARGS_MACRO(lt_, <)
	SIMD_FUNCTION_2ARGS_MACRO(le_, <=)
	SIMD_FUNCTION_2ARGS_MACRO(gt_, >)
	SIMD_FUNCTION_2ARGS_MACRO(ge_, >=)
	// NOTE: `cmp_` is `!=` (-1 if the limbs differ), as in all backends
	SIMD_FUNCTION_2ARGS_MACRO(cmp_, !=)
	SIMD_FUNCTION_2ARGS_MACRO(eq_, ==)

	/// one bit per lane (bit i = lane i): the smallest unsigned integer with
	/// at least N bits, `uint8_t`, ..., `uint64_t`, `__uint128_t`.
	/// NOTE: `gt`/`lt`/`eq`/`move` need N <= 128.
	using mask_type = LogTypeTemplate<N>;

	[[nodiscard]] constexpr static inline mask_type gt(const TxN_t &in1,
													   const TxN_t &in2) noexcept {
		const auto tmp = S::gt_(in1, in2);
		return S::move(tmp);
	}
	[[nodiscard]] constexpr static inline mask_type lt(const TxN_t &in1,
													   const TxN_t &in2) noexcept {
		const auto tmp = S::lt_(in1, in2);
		return S::move(tmp);
	}
	[[nodiscard]] constexpr static inline mask_type eq(const TxN_t &in1,
	                                                   const TxN_t &in2) noexcept {
		const auto tmp = S::eq_(in1, in2);
		return S::move(tmp);
	}
	[[nodiscard]] constexpr static inline mask_type ge(const TxN_t &in1,
	                                                   const TxN_t &in2) noexcept {
		const auto tmp = S::ge_(in1, in2);
		return S::move(tmp);
	}
	[[nodiscard]] constexpr static inline mask_type le(const TxN_t &in1,
	                                                   const TxN_t &in2) noexcept {
		const auto tmp = S::le_(in1, in2);
		return S::move(tmp);
	}
	[[nodiscard]] constexpr static inline mask_type cmp(const TxN_t &in1,
	                                                    const TxN_t &in2) noexcept {
		const auto tmp = S::cmp_(in1, in2);
		return S::move(tmp);
	}

    /// \param in1[in]: 
    /// \return popcnt(in1): 
	[[nodiscard]] constexpr static inline S popcnt(const TxN_t &in1) noexcept {
		TxN_t ret;
		if (std::is_constant_evaluated()) {
			for (uint32_t i = 0; i < LIMBS; ++i) {
				ret[i] = cryptanalysislib::popcount::popcount(in1[i]);
			}
			return ret;
		}

		uint32_t i = 0;
		if constexpr (simd512_enable) {
			for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
				ret.v512[i / nr_limbs_in_simd512] = simd512_type::popcnt(in1.v512[i / nr_limbs_in_simd512]);
			}

			if constexpr (simd512_fits) {
				return ret;
			}
		}

		if constexpr (simd256_enable) {
			for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
				ret.v256[i / nr_limbs_in_simd256] = simd256_type::popcnt(in1.v256[i / nr_limbs_in_simd256]);
			}

			if constexpr (simd256_fits) {
				return ret;
			}
		}

		for (; i < N; ++i) {
			ret[i] = cryptanalysislib::popcount::popcount(in1[i]);
		}

		return ret;
	}

	/// reverses the order of the lanes
	/// \param in1
	/// \return ret[N - 1 - i] = in1[i]
	[[nodiscard]] constexpr static inline TxN_t reverse(const TxN_t &in1) noexcept {
		S ret;
		for (uint32_t i = 0; i < LIMBS; ++i) {
			ret.d[LIMBS - 1 - i] = in1.d[i];
		}

		return ret;
	}

	/// \param in1
	/// \return true if all lanes of `in1` are equal
	[[nodiscard]] constexpr static inline bool all_equal(const TxN_t &in1) noexcept {
		for (uint32_t i = 1; i < N; ++i) {
			if (in1.d[i-1] != in1.d[i]) {
				return false;
			}
		}

		return true;
	}

	/// gathers one limb per lane: ret[i] = limb at byte offset in1[i] * off
	/// of `ptr` (unaligned reads are fine)
	/// \tparam off byte scale of the offsets (as the AVX2/NEON gathers), by
	/// 	default `sizeof(T)`, i.e. `in1` are limb indices
	/// \param ptr base pointer
	/// \param in1 offsets, in units of `off` bytes
	/// \return
	template<uint32_t off = sizeof(T)>
	[[nodiscard]] constexpr static inline TxN_t gather(const limb_type *ptr,
	                                                   const TxN_t &in1) noexcept {
		// `off` is a byte scale (like the AVX2/NEON gathers): read the limb at
		// byte offset in1[i] * off
		S ret;
		const uint8_t *ptr8 = (const uint8_t *)ptr;
		for (uint32_t i = 0; i < N; ++i) {
			const size_t c = size_t(in1.d[i]) * off;
			__builtin_memcpy(&ret.d[i], ptr8 + c, sizeof(limb_type));
		}
		return ret;
	}


	/// scatters one limb per lane: writes in2[i] at byte offset in1[i] * off
	/// of `ptr`. If two offsets collide, the higher lane wins.
	/// \tparam off byte scale of the offsets, by default `sizeof(T)`
	/// \param ptr base pointer
	/// \param in1 offsets, in units of `off` bytes
	/// \param in2 data
	template<uint32_t off = sizeof(T)>
	constexpr static inline void scatter(limb_type *ptr,
	                                     const TxN_t &in1,
	                                     const TxN_t &in2) noexcept {

		// `off` is a byte scale: write the limb at byte offset in1[i] * off
		uint8_t *ptr8 = (uint8_t *)ptr;
		for (uint32_t i = 0; i < N; ++i) {
			const size_t c = size_t(in1.d[i]) * off;
			__builtin_memcpy(ptr8 + c, &in2.d[i], sizeof(limb_type));
		}
	}

	/// scatter permutation: ret[in1[i]] = in2[i]
	/// NOTE: `in1` are the target positions (a permutation of [0, N)) and
	/// 	`in2` the data. This differs from the 8/32/64 bit AVX2/AVX-512
	/// 	`permute(in, perm)`, which gathers `ret[i] = in[perm[i]]`.
	/// \param in1 target positions
	/// \param in2 data
	/// \return
	[[nodiscard]] constexpr static inline S permute(const TxN_t &in1,
	                                                const TxN_t &in2) noexcept {
		S ret;
		for (uint32_t i = 0; i < N; ++i) {
			const size_t pos = in1.d[i];
			ret.d[pos] = in2.d[i];
		}
		return ret;
	}

	/// \param in1
	/// \return the msb of each lane, one bit per lane (see `mask_type`)
	[[nodiscard]] constexpr static inline mask_type move(const TxN_t &in1) noexcept {
		static_assert(N <= 128u, "the lane mask has at most 128 bits");
		mask_type ret = 0;
		uint32_t i = 0;
		if constexpr (simd512_enable) {
			for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
				const uint64_t data = uint64_t(simd512_type::move(in1.v512[i / nr_limbs_in_simd512]));
				ret ^= mask_type(data) << i;
			}

			if constexpr (simd512_fits) {
				return ret;
			}
		}

		if constexpr (simd256_enable) {
			for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
				ret ^= mask_type(simd256_type::move(in1.v256[i / nr_limbs_in_simd256])) << i;
			}

			if constexpr (simd256_fits) {
				return ret;
			}
		}

		// NOTE: in the unsigned type, `(d & mask) > 0` is false for signed limbs
		for (; i < N; ++i) {
			ret ^= mask_type((U)in1.d[i] >> (limb_bits - 1u)) << i;
		}

		return ret;
	}

	/// \returns number of limbs
	[[nodiscard]] constexpr static size_t size() noexcept {
		return N;
	}

	[[nodiscard]] constexpr static bool is_unsigned() noexcept {
		return std::is_unsigned_v<T>;
	}
	constexpr static void info() {
		std::cout << "{ name: \"TxN_t<" << typeid(T).name() << "\"," << N << ">\n"
		          << ", limb_bits: " << limb_bits
				  << ", total_bits: " << total_bits
				  << ", simd256_limbs: " << simd256_limbs
				  << ", simd512_limbs: " << simd512_limbs
				  << ", nr_limbs_in_simd256: " << nr_limbs_in_simd256
				  << ", nr_limbs_in_simd512: " << nr_limbs_in_simd512
				  << ", simd256_enable: " << simd256_enable
				  << ", simd512_enable: " << simd512_enable
				  << ", simd256_fits: " << simd256_fits
				  << ", simd512_fits: " << simd512_fits
				  << ", simd256_bits: " << simd256_bits
				  << ", simd512_bits: " << simd512_bits
				  << ", simd256_type: " << typeid(simd256_type).name()
				  << ", simd512_type: " << typeid(simd512_type).name()
				  << " }" << std::endl;
	}
};


///
template<typename T, const uint32_t N>
constexpr inline TxN_t<T, N> operator*(const TxN_t<T, N> &lhs, const TxN_t<T, N> &rhs) noexcept {
	return TxN_t<T, N>::mullo(lhs, rhs);
}
/// NOTE: the scalar is a `T` (non-deduced), so it is not truncated to 8 bits
template<typename T, const uint32_t N>
constexpr inline TxN_t<T, N> operator*(const TxN_t<T, N> &lhs, const std::type_identity_t<T> &rhs) noexcept {
	return TxN_t<T, N>::mullo(lhs, rhs);
}
template<typename T, const uint32_t N>
constexpr inline TxN_t<T, N> operator*(const std::type_identity_t<T> &lhs, const TxN_t<T, N> &rhs) noexcept {
	return TxN_t<T, N>::mullo(rhs, lhs);
}
template<typename T, const uint32_t N>
constexpr inline TxN_t<T, N> operator+(const TxN_t<T, N> &lhs, const TxN_t<T, N> &rhs) noexcept {
	return TxN_t<T, N>::add(lhs, rhs);
}
template<typename T, const uint32_t N>
constexpr inline TxN_t<T, N> operator-(const TxN_t<T, N> &lhs, const TxN_t<T, N> &rhs) noexcept {
	return TxN_t<T, N>::sub(lhs, rhs);
}
template<typename T, const uint32_t N>
constexpr inline TxN_t<T, N> operator&(const TxN_t<T, N> &lhs, const TxN_t<T, N> &rhs) noexcept {
	return TxN_t<T, N>::and_(lhs, rhs);
}
template<typename T, const uint32_t N>
constexpr inline TxN_t<T, N> operator^(const TxN_t<T, N> &lhs, const TxN_t<T, N> &rhs) noexcept {
	return TxN_t<T, N>::xor_(lhs, rhs);
}
template<typename T, const uint32_t N>
constexpr inline TxN_t<T, N> operator|(const TxN_t<T, N> &lhs, const TxN_t<T, N> &rhs) noexcept {
	return TxN_t<T, N>::or_(lhs, rhs);
}
template<typename T, const uint32_t N>
constexpr inline TxN_t<T, N> operator~(const TxN_t<T, N> &lhs) noexcept {
	return TxN_t<T, N>::not_(lhs);
}
template<typename T, const uint32_t N>
constexpr inline TxN_t<T, N> operator>> (const TxN_t<T, N>& lhs, const uint32_t rhs) noexcept {
	return TxN_t<T, N>::srli(lhs, rhs);
}
template<typename T, const uint32_t N>
constexpr inline TxN_t<T, N> operator<< (const TxN_t<T, N>& lhs, const uint32_t rhs) noexcept {
	return TxN_t<T, N>::slli(lhs, rhs);
}
template<typename T, const uint32_t N>
constexpr inline TxN_t<T, N> operator^=(TxN_t<T, N> &lhs, const TxN_t<T, N> &rhs) noexcept {
	lhs = TxN_t<T, N>::xor_(lhs, rhs);
	return lhs;
}
template<typename T, const uint32_t N>
constexpr inline TxN_t<T, N> operator&=(TxN_t<T, N> &lhs, const TxN_t<T, N> &rhs) noexcept {
	lhs = TxN_t<T, N>::and_(lhs, rhs);
	return lhs;
}
template<typename T, const uint32_t N>
constexpr inline TxN_t<T, N> operator|=(TxN_t<T, N> &lhs, const TxN_t<T, N> &rhs) noexcept {
	lhs = TxN_t<T, N>::or_(lhs, rhs);
	return lhs;
}


template<>
class TxN_t<uint64_t, 4> : public uint64x4_t {
public:
	// ah yes. C++ is love, C++ is life.
	// The problem is, that copy constructors are never inherited
	constexpr inline TxN_t() noexcept = default;
	constexpr inline TxN_t(const uint64x4_t &k) noexcept {
#if defined(USE_AVX2)
		v256 = k.v256;
#elif defined(USE_ARM)
		v128[0] = k.v128[0];
		v128[1] = k.v128[1];
#else
		for (uint32_t i = 0; i < LIMBS; i++) {
			d[i] = k.d[i];
		}
#endif
	};
};

template<>
class TxN_t<uint32_t, 8> : public uint32x8_t {
public:
	constexpr inline TxN_t() noexcept = default;
	constexpr inline TxN_t(const uint32x8_t &k) noexcept {
#if defined(USE_AVX2)
		v256 = k.v256;
#elif defined(USE_ARM)
		v128[0] = k.v128[0];
		v128[1] = k.v128[1];
#else
		for (uint32_t i = 0; i < LIMBS; i++) {
			d[i] = k.d[i];
		}
#endif
	};
};

template<>
class TxN_t<uint16_t, 16>: public uint16x16_t{
public:
	constexpr inline TxN_t() noexcept = default;
	constexpr inline TxN_t(const uint16x16_t &k) noexcept {
#if defined(USE_AVX2)
		v256 = k.v256;
#elif defined(USE_ARM)
		v128[0] = k.v128[0];
		v128[1] = k.v128[1];
#else
		for (uint32_t i = 0; i < LIMBS; i++) {
			d[i] = k.d[i];
		}
#endif
	};
};

template<>
class TxN_t<uint8_t, 32> : public uint8x32_t {
public:
	constexpr inline TxN_t() noexcept = default;
	constexpr inline TxN_t(const uint8x32_t &k) noexcept {
#if defined(USE_AVX2)
		v256 = k.v256;
#elif defined(USE_ARM)
		v128[0] = k.v128[0];
		v128[1] = k.v128[1];
#else
		for (uint32_t i = 0; i < LIMBS; i++) {
			d[i] = k.d[i];
		}
#endif
	};
};

#ifdef USE_AVX512F
template<>
class TxN_t<uint8_t, 64> : public uint8x64_t {
public:
	constexpr inline TxN_t() noexcept = default;
	constexpr inline TxN_t(const uint8x64_t &k) noexcept {
		v512 = k.v512;
	};
};
template<>
class TxN_t<uint16_t, 32> : public uint16x32_t {
public:
	constexpr inline TxN_t() noexcept = default;
	constexpr inline TxN_t(const uint16x32_t &k) noexcept {
		v512 = k.v512;
	};
};
template<>
class TxN_t<uint32_t, 16> : public uint32x16_t {
public:
	constexpr inline TxN_t() noexcept = default;
	constexpr inline TxN_t(const uint32x16_t &k) noexcept {
		v512 = k.v512;
	};
};
template<>
class TxN_t<uint64_t, 8> : public uint64x8_t {
public:
	constexpr inline TxN_t() noexcept = default;
	constexpr inline TxN_t(const uint64x8_t &k) noexcept {
		v512 = k.v512;
	};
};
#endif

#endif//CRYPTANALYSISLIB_GENERIC_H
