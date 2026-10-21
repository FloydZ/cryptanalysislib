#ifndef CRYPTANALYSISLIB_SIMD_GENERIC_H
#define CRYPTANALYSISLIB_SIMD_GENERIC_H

#ifndef CRYPTANALYSISLIB_SIMD_H
#error "dont inlcude this file. Use `#include \"simd/simd.h\"`."
#endif

#include <type_traits>

#include "algorithm/bits/popcount.h"
#include "algorithm/countl.h"
#include "algorithm/countr.h"
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

	/// set the `pos`-th bit in each limb
	/// \param pos[in]: bit position to set
	constexpr inline void set_bit(const uint32_t pos) noexcept {
		*this = set1(T(U(1) << pos));
	}

	/// masked load: lane i is loaded if bit i of `m` is set, otherwise 0.
	/// Masked out lanes are not accessed.
	/// \param ptr
	/// \param m lane mask
	/// \return
	[[nodiscard]] constexpr static inline TxN_t maskz_unaligned_load(const T *ptr,
	                                                                 const mask_type m) noexcept {
		using T512 = typename simd512_type::limb_type;
		using T256 = typename simd256_type::limb_type;
		constexpr uint64_t lm512 = nr_limbs_in_simd512 >= 64u ? ~0ull : ((1ull << nr_limbs_in_simd512) - 1ull);
		constexpr uint64_t lm256 = nr_limbs_in_simd256 >= 64u ? ~0ull : ((1ull << nr_limbs_in_simd256) - 1ull);
		TxN_t ret;
		uint32_t i = 0;
		if (!std::is_constant_evaluated()) {
			if constexpr (simd512_enable && requires(const T512 *p, const Mask<nr_limbs_in_simd512> &k) {
				              simd512_type::maskz_unaligned_load(p, k); }) {
				for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
					const Mask<nr_limbs_in_simd512> k(MaskType<nr_limbs_in_simd512>(uint64_t(m >> i) & lm512));
					ret.v512[i / nr_limbs_in_simd512] = simd512_type::maskz_unaligned_load((const T512 *)(ptr + i), k);
				}
			}
			if constexpr (simd256_enable && requires(const T256 *p, const Mask<nr_limbs_in_simd256> &k) {
				              simd256_type::maskz_unaligned_load(p, k); }) {
				for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
					const Mask<nr_limbs_in_simd256> k(MaskType<nr_limbs_in_simd256>(uint64_t(m >> i) & lm256));
					ret.v256[i / nr_limbs_in_simd256] = simd256_type::maskz_unaligned_load((const T256 *)(ptr + i), k);
				}
			}
		}
		for (; i < N; ++i) {
			ret.d[i] = ((m >> i) & 1u) ? ptr[i] : T(0);
		}
		return ret;
	}

	/// same as `maskz_unaligned_load` (no alignment needed)
	[[nodiscard]] constexpr static inline TxN_t maskz_aligned_load(const T *ptr,
	                                                               const mask_type m) noexcept {
		return maskz_unaligned_load(ptr, m);
	}

	/// \param in1
	/// \param in2 divisor, must not be zero
	/// \return in1 / in2, lane-wise
	[[nodiscard]] constexpr static inline TxN_t div(const TxN_t &in1,
	                                                const T in2) noexcept {
		assert(in2 != 0);
		TxN_t ret;
		uint32_t i = 0;
		if (!std::is_constant_evaluated()) {
			if constexpr (simd512_enable && requires(const simd512_type a, const T b) { simd512_type::div(a, b); }) {
				for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
					ret.v512[i / nr_limbs_in_simd512] = simd512_type::div(in1.v512[i / nr_limbs_in_simd512], in2);
				}
			}
			if constexpr (simd256_enable && requires(const simd256_type a, const T b) { simd256_type::div(a, b); }) {
				for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
					ret.v256[i / nr_limbs_in_simd256] = simd256_type::div(in1.v256[i / nr_limbs_in_simd256], in2);
				}
			}
		}
		for (; i < N; ++i) {
			ret.d[i] = T(in1.d[i] / in2);
		}
		return ret;
	}

	/// \param in1
	/// \return the number of leading zeros of each limb (`limb_bits` for 0)
	[[nodiscard]] constexpr static inline TxN_t lzcnt(const TxN_t &in1) noexcept {
		TxN_t ret;
		uint32_t i = 0;
		if (!std::is_constant_evaluated()) {
			if constexpr (simd512_enable && requires(const simd512_type a) { simd512_type::lzcnt(a); }) {
				for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
					ret.v512[i / nr_limbs_in_simd512] = simd512_type::lzcnt(in1.v512[i / nr_limbs_in_simd512]);
				}
			}
			if constexpr (simd256_enable && requires(const simd256_type a) { simd256_type::lzcnt(a); }) {
				for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
					ret.v256[i / nr_limbs_in_simd256] = simd256_type::lzcnt(in1.v256[i / nr_limbs_in_simd256]);
				}
			}
		}
		for (; i < N; ++i) {
			ret.d[i] = T(cryptanalysislib::countl_zero(U(in1.d[i])));
		}
		return ret;
	}

	/// \param in1
	/// \return the number of trailing zeros of each limb (`limb_bits` for 0)
	[[nodiscard]] constexpr static inline TxN_t tzcnt(const TxN_t &in1) noexcept {
		TxN_t ret;
		uint32_t i = 0;
		if (!std::is_constant_evaluated()) {
			if constexpr (simd512_enable && requires(const simd512_type a) { simd512_type::tzcnt(a); }) {
				for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
					ret.v512[i / nr_limbs_in_simd512] = simd512_type::tzcnt(in1.v512[i / nr_limbs_in_simd512]);
				}
			}
			if constexpr (simd256_enable && requires(const simd256_type a) { simd256_type::tzcnt(a); }) {
				for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
					ret.v256[i / nr_limbs_in_simd256] = simd256_type::tzcnt(in1.v256[i / nr_limbs_in_simd256]);
				}
			}
		}
		for (; i < N; ++i) {
			ret.d[i] = T(cryptanalysislib::countr_zero(U(in1.d[i])));
		}
		return ret;
	}

	/// same as `lzcnt`
	[[nodiscard]] constexpr static inline TxN_t clz(const TxN_t &in1) noexcept { return lzcnt(in1); }

	/// same as `tzcnt`
	[[nodiscard]] constexpr static inline TxN_t ctz(const TxN_t &in1) noexcept { return tzcnt(in1); }

	/// same semantics as `vpconflict` (AVX512CD):
	/// lane i is the bitmask of all lanes j < i with in1[j] == in1[i].
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline TxN_t conflict(const TxN_t &in1) noexcept {
		static_assert(N <= limb_bits, "the lane mask must fit into a limb");
		// compare all lanes with the broadcasted lane j, keep the lanes > j
		TxN_t idx;
		for (uint32_t i = 0; i < N; ++i) {
			idx.d[i] = T(i);
		}
		TxN_t ret = set1(T(0));
		for (uint32_t j = 0; j + 1u < N; ++j) {
			const TxN_t t = and_(eq_(in1, set1(in1.d[j])), gt_(idx, set1(T(j))));
			ret = or_(ret, and_(t, set1(T(U(1) << j))));
		}
		return ret;
	}

	/// packs the lanes of `in1` selected by `m` to the front (in order),
	/// the other lanes are zero, as `_mm512_maskz_compress_epi*`
	/// \param in1
	/// \param m lane mask
	/// \return
	[[nodiscard]] constexpr static inline TxN_t compress(const TxN_t &in1,
	                                                     const mask_type m) noexcept {
		return compress(in1, set1(0), m);
	}

	/// packs the lanes of `in1` selected by `m` to the front (in order),
	/// the other lanes are taken from `in2`, as `_mm512_mask_compress_epi*`
	/// \param in1
	/// \param in2
	/// \param m lane mask
	/// \return
	[[nodiscard]] constexpr static inline TxN_t compress(const TxN_t &in1,
	                                                     const TxN_t &in2,
	                                                     const mask_type m) noexcept {
		TxN_t ret = in2;
		// the selected lanes are compress-stored over the front of `in2`
		compress(ret.d, in1, m);
		return ret;
	}

	/// writes the lanes of `in1` selected by `m` contiguously to `ptr`,
	/// as `_mm512_mask_compressstoreu_epi*`
	/// \param ptr
	/// \param in1
	/// \param m lane mask
	constexpr static inline void compress(T *ptr,
	                                      const TxN_t &in1,
	                                      const mask_type m) noexcept {
		constexpr uint64_t lm512 = nr_limbs_in_simd512 >= 64u ? ~0ull : ((1ull << nr_limbs_in_simd512) - 1ull);
		constexpr uint64_t lm256 = nr_limbs_in_simd256 >= 64u ? ~0ull : ((1ull << nr_limbs_in_simd256) - 1ull);
		uint32_t i = 0, j = 0;
		if (!std::is_constant_evaluated()) {
			if constexpr (simd512_enable && requires(T *p, const simd512_type a, const Mask<nr_limbs_in_simd512> &k) {
				              simd512_type::compress(p, a, k); }) {
				for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
					const uint64_t k = uint64_t(m >> i) & lm512;
					simd512_type::compress(ptr + j, in1.v512[i / nr_limbs_in_simd512],
					                       Mask<nr_limbs_in_simd512>(MaskType<nr_limbs_in_simd512>(k)));
					j += __builtin_popcountll(k);
				}
			}
			if constexpr (simd256_enable && requires(T *p, const simd256_type a, const Mask<nr_limbs_in_simd256> &k) {
				              simd256_type::compress(p, a, k); }) {
				for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
					const uint64_t k = uint64_t(m >> i) & lm256;
					simd256_type::compress(ptr + j, in1.v256[i / nr_limbs_in_simd256],
					                       Mask<nr_limbs_in_simd256>(MaskType<nr_limbs_in_simd256>(k)));
					j += __builtin_popcountll(k);
				}
			}
		}
		for (; i < N; ++i) {
			if ((m >> i) & 1u) { ptr[j++] = in1.d[i]; }
		}
	}

	/// \param in1 register to test
	/// \param bit_pos bit position to test
	/// \return bit i is set, if bit `bit_pos` of `in1[i]` is set
	[[nodiscard]] constexpr static inline mask_type test(const TxN_t &in1,
	                                                     const uint32_t bit_pos) noexcept {
		// move bit `bit_pos` into the msb
		return move(slli(in1, uint32_t(limb_bits - 1u - bit_pos)));
	}

	/// \param in1
	/// \return the minimum of all lanes
	[[nodiscard]] constexpr static inline T reduce_min(const TxN_t &in1) noexcept {
		T ret = in1.d[0];
		uint32_t i = 0;
		if (!std::is_constant_evaluated()) {
			// lane-wise minimum of all chunks, then one reduction
			if constexpr (simd512_enable && requires(const simd512_type a) {
				              simd512_type::min(a, a); simd512_type::reduce_min(a); }) {
				simd512_type acc = in1.v512[0];
				for (i = nr_limbs_in_simd512; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
					acc = simd512_type::min(acc, in1.v512[i / nr_limbs_in_simd512]);
				}
				ret = simd512_type::reduce_min(acc);
			} else if constexpr (simd256_enable && requires(const simd256_type a) {
				                     simd256_type::min(a, a); simd256_type::reduce_min(a); }) {
				simd256_type acc = in1.v256[0];
				for (i = nr_limbs_in_simd256; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
					acc = simd256_type::min(acc, in1.v256[i / nr_limbs_in_simd256]);
				}
				ret = simd256_type::reduce_min(acc);
			}
		}
		for (; i < N; ++i) {
			ret = in1.d[i] < ret ? in1.d[i] : ret;
		}
		return ret;
	}

	/// \param in1
	/// \return the maximum of all lanes
	[[nodiscard]] constexpr static inline T reduce_max(const TxN_t &in1) noexcept {
		T ret = in1.d[0];
		uint32_t i = 0;
		if (!std::is_constant_evaluated()) {
			// lane-wise maximum of all chunks, then one reduction
			if constexpr (simd512_enable && requires(const simd512_type a) {
				              simd512_type::max(a, a); simd512_type::reduce_max(a); }) {
				simd512_type acc = in1.v512[0];
				for (i = nr_limbs_in_simd512; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
					acc = simd512_type::max(acc, in1.v512[i / nr_limbs_in_simd512]);
				}
				ret = simd512_type::reduce_max(acc);
			} else if constexpr (simd256_enable && requires(const simd256_type a) {
				                     simd256_type::max(a, a); simd256_type::reduce_max(a); }) {
				simd256_type acc = in1.v256[0];
				for (i = nr_limbs_in_simd256; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
					acc = simd256_type::max(acc, in1.v256[i / nr_limbs_in_simd256]);
				}
				ret = simd256_type::reduce_max(acc);
			}
		}
		for (; i < N; ++i) {
			ret = in1.d[i] > ret ? in1.d[i] : ret;
		}
		return ret;
	}

	/// \param in1
	/// \return the sum of all lanes (modulo 2**limb_bits)
	[[nodiscard]] constexpr static inline T reduce_add(const TxN_t &in1) noexcept {
		T ret = 0;
		uint32_t i = 0;
		if (!std::is_constant_evaluated()) {
			// lane-wise sum of all chunks, then one reduction
			if constexpr (simd512_enable && requires(const simd512_type a) {
				              simd512_type::add(a, a); simd512_type::reduce_add(a); }) {
				simd512_type acc = in1.v512[0];
				for (i = nr_limbs_in_simd512; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
					acc = simd512_type::add(acc, in1.v512[i / nr_limbs_in_simd512]);
				}
				ret = simd512_type::reduce_add(acc);
			} else if constexpr (simd256_enable && requires(const simd256_type a) {
				                     simd256_type::add(a, a); simd256_type::reduce_add(a); }) {
				simd256_type acc = in1.v256[0];
				for (i = nr_limbs_in_simd256; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
					acc = simd256_type::add(acc, in1.v256[i / nr_limbs_in_simd256]);
				}
				ret = simd256_type::reduce_add(acc);
			}
		}
		for (; i < N; ++i) {
			ret = T(ret + in1.d[i]);
		}
		return ret;
	}

	/// \param a
	/// \param b
	/// \return lane-wise minimum
	[[nodiscard]] constexpr static inline TxN_t min(const TxN_t &a,
	                                                const TxN_t &b) noexcept {
		TxN_t ret;
		uint32_t i = 0;
		if (!std::is_constant_evaluated()) {
			if constexpr (simd512_enable && requires(const simd512_type x) { simd512_type::min(x, x); }) {
				for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
					ret.v512[i / nr_limbs_in_simd512] = simd512_type::min(a.v512[i / nr_limbs_in_simd512], b.v512[i / nr_limbs_in_simd512]);
				}
			}
			if constexpr (simd256_enable && requires(const simd256_type x) { simd256_type::min(x, x); }) {
				for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
					ret.v256[i / nr_limbs_in_simd256] = simd256_type::min(a.v256[i / nr_limbs_in_simd256], b.v256[i / nr_limbs_in_simd256]);
				}
			}
		}
		for (; i < N; ++i) {
			ret.d[i] = a.d[i] < b.d[i] ? a.d[i] : b.d[i];
		}
		return ret;
	}

	/// \param a
	/// \param b
	/// \return lane-wise maximum
	[[nodiscard]] constexpr static inline TxN_t max(const TxN_t &a,
	                                                const TxN_t &b) noexcept {
		TxN_t ret;
		uint32_t i = 0;
		if (!std::is_constant_evaluated()) {
			if constexpr (simd512_enable && requires(const simd512_type x) { simd512_type::max(x, x); }) {
				for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
					ret.v512[i / nr_limbs_in_simd512] = simd512_type::max(a.v512[i / nr_limbs_in_simd512], b.v512[i / nr_limbs_in_simd512]);
				}
			}
			if constexpr (simd256_enable && requires(const simd256_type x) { simd256_type::max(x, x); }) {
				for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
					ret.v256[i / nr_limbs_in_simd256] = simd256_type::max(a.v256[i / nr_limbs_in_simd256], b.v256[i / nr_limbs_in_simd256]);
				}
			}
		}
		for (; i < N; ++i) {
			ret.d[i] = a.d[i] > b.d[i] ? a.d[i] : b.d[i];
		}
		return ret;
	}

	/// horizontal add of the unsigned bytes of each limb
	/// \param in1
	/// \return sum of the bytes, per limb
	[[nodiscard]] constexpr static inline TxN_t hadd_epu8(const TxN_t &in1) noexcept {
		static_assert(sizeof(T) >= 2);
		// sums of neighbouring 8, 16, 32 bit fields
		constexpr U m8 = U(U(~U(0)) / U(0xFFFFu) * U(0xFFu));
		TxN_t x = add(and_(in1, set1(T(m8))), and_(srli(in1, 8), set1(T(m8))));
		if constexpr (sizeof(T) >= 4) {
			constexpr U m16 = U(U(~U(0)) / U(0xFFFFFFFFu) * U(0xFFFFu));
			x = add(and_(x, set1(T(m16))), and_(srli(x, 16), set1(T(m16))));
		}
		if constexpr (sizeof(T) == 8) {
			x = add(and_(x, set1(T(0xFFFFFFFFull))), srli(x, 32));
		}
		return x;
	}

	/// horizontal add of the unsigned 16 bit values of each limb
	/// \param in1
	/// \return sum, per limb
	[[nodiscard]] constexpr static inline TxN_t hadd_epu16(const TxN_t &in1) noexcept {
		static_assert(sizeof(T) >= 4);
		// sums of neighbouring 16, 32 bit fields
		constexpr U m16 = U(U(~U(0)) / U(0xFFFFFFFFu) * U(0xFFFFu));
		TxN_t x = add(and_(in1, set1(T(m16))), and_(srli(in1, 16), set1(T(m16))));
		if constexpr (sizeof(T) == 8) {
			x = add(and_(x, set1(T(0xFFFFFFFFull))), srli(x, 32));
		}
		return x;
	}

	/// horizontal add of the 2 unsigned 32 bit values of each 64 bit limb
	/// \param in1
	/// \return sum, per limb
	[[nodiscard]] constexpr static inline TxN_t hadd_epu32(const TxN_t &in1) noexcept {
		static_assert(sizeof(T) == 8);
		return add(and_(in1, set1(T(0xFFFFFFFFull))), srli(in1, 32));
	}

	/// number of 4 bit values (nibbles) equal to `in2` in each limb
	/// \param in1
	/// \param in2 nibble to count, < 16
	/// \return per limb: #{i : nibble i of the limb == in2}
	[[nodiscard]] constexpr static inline TxN_t histogram_epi4(const TxN_t &in1,
	                                                           const uint8_t in2) noexcept {
		assert(in2 < 16);
		// bit 0 of a nibble is set, iff the nibble != in2
		constexpr U ones = U(U(~U(0)) / U(0xFu));
		const TxN_t z = xor_(in1, set1(T(U(ones * in2))));
		const TxN_t nz = and_(or_(or_(z, srli(z, 1)), or_(srli(z, 2), srli(z, 3))), set1(T(ones)));
		return sub(set1(T(limb_bits / 4u)), popcnt(nz));
	}

	/// the indices of the lanes selected by `mask` in front, the other lanes
	/// are zero, e.g. 0b1010 -> [1, 3, 0, ..., 0] (as the AVX2 32 bit version)
	/// \param mask
	/// \return
	[[nodiscard]] constexpr static inline TxN_t pack(const mask_type mask) noexcept {
		TxN_t idx;
		for (uint32_t i = 0; i < N; ++i) {
			idx.d[i] = T(i);
		}
		return compress(idx, mask);
	}

	/// zero extends the first `N` bytes of `in`
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline TxN_t cvtepu8(const cryptanalysislib::_uint8x16_t in) noexcept {
		static_assert(N <= 16u);
		TxN_t ret;
		uint32_t i = 0;
		if (!std::is_constant_evaluated()) {
			if constexpr (simd256_enable && requires(const cryptanalysislib::_uint8x16_t b) { simd256_type::cvtepu8(b); }) {
				// each chunk zero extends the first bytes of its own 16 byte window
				alignas(16) uint8_t buf[32] = {0};
				__builtin_memcpy(buf, in.d, 16);
				for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
					ret.v256[i / nr_limbs_in_simd256] = simd256_type::cvtepu8(cryptanalysislib::_uint8x16_t::unaligned_load(buf + i));
				}
			}
		}
		for (; i < N; ++i) {
			ret.d[i] = T(uint8_t(in.d[i]));
		}
		return ret;
	}

	/// transposes the 8x8 bit matrix in each group of 8 byte limbs, as the
	/// AVX-512 version: bit i of output byte j is bit j of input byte (7 - i)
	/// \param input
	/// \return
	[[nodiscard]] constexpr static inline TxN_t transpose(const TxN_t &input) noexcept {
		static_assert((sizeof(T) == 1) && ((N % 8u) == 0));
		TxN_t ret;
		uint32_t i = 0;
		if (!std::is_constant_evaluated()) {
			if constexpr (simd512_enable && requires(const simd512_type a) { simd512_type::transpose(a); }) {
				for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
					ret.v512[i / nr_limbs_in_simd512] = simd512_type::transpose(input.v512[i / nr_limbs_in_simd512]);
				}
			}
			if constexpr (simd256_enable && requires(const simd256_type a) { simd256_type::transpose(a); }) {
				for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
					ret.v256[i / nr_limbs_in_simd256] = simd256_type::transpose(input.v256[i / nr_limbs_in_simd256]);
				}
			}
		}
		// reverse the bytes of each 64 bit word, then transpose with the
		// delta swaps of Hacker's Delight `transpose8`
		for (; i < N; i += 8) {
			uint64_t x = 0;
			for (uint32_t j = 0; j < 8; ++j) {
				x |= uint64_t(uint8_t(input.d[i + j])) << (8u * (7u - j));
			}
			uint64_t t;
			t = (x ^ (x >> 7u)) & 0x00AA00AA00AA00AAull;
			x ^= t ^ (t << 7u);
			t = (x ^ (x >> 14u)) & 0x0000CCCC0000CCCCull;
			x ^= t ^ (t << 14u);
			t = (x ^ (x >> 28u)) & 0x00000000F0F0F0F0ull;
			x ^= t ^ (t << 28u);
			for (uint32_t j = 0; j < 8; ++j) {
				ret.d[i + j] = T(uint8_t(x >> (8u * j)));
			}
		}
		return ret;
	}

	/// as `_mm256_sign_epi8`
	/// \return per lane: 0 if b == 0, -a if b < 0, a if b > 0
	[[nodiscard]] constexpr static inline TxN_t comp_sign(const TxN_t &a,
	                                                      const TxN_t &b) noexcept {
		// neg = -1 iff b < 0, then (a ^ neg) - neg is -a or a
		const TxN_t neg = sub(set1(T(0)), srli(b, uint32_t(limb_bits - 1u)));
		const TxN_t r = sub(xor_(a, neg), neg);
		return and_(r, cmp_(b, set1(T(0))));
	}

	/// permutes the 4 limbs of each group of 4, like `_mm256_permute4x64_epi64`:
	/// ret[i] = in1[(i & ~3) + ((in2 >> 2(i % 4)) & 3)]
	/// \param in1
	/// \param in2 8 bit, 2 bits per lane
	/// \return
	[[nodiscard]] constexpr static inline TxN_t permute(const TxN_t &in1,
	                                                    const uint32_t in2) noexcept {
		static_assert((N % 4u) == 0);
		TxN_t ret;
		uint32_t i = 0;
		if (!std::is_constant_evaluated()) {
			// NOTE: the backends have this runtime permutation only for 64 bit lanes
			if constexpr (sizeof(T) == 8u) {
				if constexpr (simd512_enable && requires(const simd512_type a, const uint32_t k) { simd512_type::permute(a, k); }) {
					for (; i + nr_limbs_in_simd512 <= N; i += nr_limbs_in_simd512) {
						ret.v512[i / nr_limbs_in_simd512] = simd512_type::permute(in1.v512[i / nr_limbs_in_simd512], in2);
					}
				}
				if constexpr (simd256_enable && requires(const simd256_type a, const uint32_t k) { simd256_type::permute(a, k); }) {
					for (; i + nr_limbs_in_simd256 <= N; i += nr_limbs_in_simd256) {
						ret.v256[i / nr_limbs_in_simd256] = simd256_type::permute(in1.v256[i / nr_limbs_in_simd256], in2);
					}
				}
			}
		}
		for (; i < N; ++i) {
			ret.d[i] = in1.d[(i & ~3u) + ((in2 >> (2u * (i % 4u))) & 3u)];
		}
		return ret;
	}

	/// compile time version of `permute(in1, in2)`
	template<const uint32_t in2>
	[[nodiscard]] constexpr static inline TxN_t permute(const TxN_t &in1) noexcept {
		return permute(in1, in2);
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
