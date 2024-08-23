#ifndef CRYPTANALYSISLIB_SIMD_AVX2_H
#define CRYPTANALYSISLIB_SIMD_AVX2_H

#ifndef CRYPTANALYSISLIB_SIMD_H
#error "dont include this file directly. Use `#include <simd/simd.h>`"
#endif

#ifndef USE_AVX2
#error "no avx2 enabled."
#endif

#include <emmintrin.h>
#include <type_traits>


#include <cstdint>
#include <cstdio>
#include <immintrin.h>

#include "helper.h"
#include "popcount/popcount.h"
#include "random.h"


/// justification for the unsinged comparison
/// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIAMxcpK4AMngMmAByPgBGmMQgAGwapAAOqAqETgwe3r4BQemZjgJhEdEscQnJtpj2JQxCBEzEBLk%2BfoG19dlNLQRlUbHxSSkKza3t%2BV3j/YMVVaMAlLaoXsTI7Bzm/uHI3lgA1Cb%2BbngsLOEExOEAdAgn2CYaAILPLwD6HyxmAKyJeEOCloBAAHBA0Axxocvj9/oCmKRDpDobC/gDDjElscAOxWV7PACcxEwBHWDEOEDR8OxUC%2BADd/GYAI4KJZMY6nSkMpmspZYk74t44gAi72pGK8IPBKIIMO%2B6IRSNl8rhGKxuKFRJJZOIFKpCpplINjJZXnZnLc3I%2BpuZ5oF/i1ove4sNGIUwDBEIEqLdSuRPrlEsBGpMeJdGmJpPJ1rVeFpJt5bI5j2ttrZDqdYtewcOXk9MsDqsVh0RAahQb9mOxYa1kZ1MYNcYTPLNFtTibbmZdzoJfbeOarwIIXES3orxfhpeVRdzofDfajuv1wZbNtHCHjKa5Jo38e7BN7nyrUpHY5VubLF6r87rS8bq%2BNDI35u3Vt3iQQ9qWgp72YHx5xkCnqjuOvpAVes43jWC4Dveeqxoqa70nu7b%2BNgaaoQewr/rm%2BZnmBlYQTOE5zjBd4NghTZIU%2B66fq%2BnIYR%2BX78j%2Bjp/hGAG5sOTKEZOGKQaR0GahG8Ern6yGgpkaHvgyUn7mxWaukBp68dexHluBJa3qJlHic2tH0lJDEnLJNrGaxv6Hv%2Byklh6BBqVBGnqdp5G6dGVGPnS5nSduTFydJ2FhrhJ6eo5QnOU5rkiYuemIUa3lGQoJnoWmFlBUenHcSCiSSHxl4kVpU46bFHn6TRiWSPgMlplVClWThtlTqeuX5VWglFeqbmlcu8UAshVUpWZ9KDZZ7HWVlQ6eq1LlTh1RHRbW7m9dRCUmnVaH%2BTaG0ZSFKnTXls0CYVC3Fd1cFxat/WGaNfm1eg34NcFHArLQnC/LwfjcLwqCcG41jWECawbJgxxmIEvAEJoL0rAA1iAvwpG9HCSJ90OkL9HC8AoIApFDHBaCscCwEgaAsKkdDxOQlBkxT9AJPshjAFwoJcCkNAgvEOMQDE6MxOELQAJ6cDwpD88wxCCwA8jE2iYA4Iu8GTbCCFLDC0MLBO8FgMT5m4Yi0Dj32kFgLBM%2BIWsm3gJIOHg9KYEbWjBKo8teAQWyi1cdTo7QeAxMQQseFg6PXOciukPbxAxBkmAipgZtGL7RjQysVAGMACgAGp4JgADuUupIw4f8IIIhiOwUgyIIigqOolu6EEBjJ6YljWPofs45AKyoKkDRGwAtGb9KqIcg9MMPZij5UlxTzElyT/3Uv%2BD9kc3FgncQCsdjyw0LgMO4ngdHooThEMlQjNIRRZAIUx%2BIUGTXww8zDAk0jb7bAh9JMh/5EE78NF/AYp8FgX1sBMNoP875gLmMAl%2BIBJBb2BpsCQr13po0tpjQ4qhQSJH7rlZETdgCHBZrcLgtwNCUlwIQEgYNAhLEhinOGCMkacFRqQL6TtMbY1xqQfGhNUEcDMOgzhnAGFayWCsSOmRnCSCAA%3D%3D%3D

using namespace cryptanalysislib::popcount::internal;

namespace internal {
	/// helper function. This enforces the compiler to emit a `vmovdqu` instruction
	/// \param ptr pointer to data.
	///				No alignment needed
	/// 			but 32 bytes should be readable
	/// \return unaligned `__m256i`
	constexpr static inline __m256i_u unaligned_load_wrapper(const __m256i_u *ptr) {
		return *ptr;
	}

	/// helper function. This enforces the compiler to emit a unaligned instruction
	/// \param ptr pointer to data
	/// \param data data to store
	/// \return nothing
	constexpr static inline void unaligned_store_wrapper(__m256i_u *ptr, __m256i_u data) {
		*ptr = data;
	}


	constexpr static inline __m128i_u unaligned_load_wrapper_128(__m128i_u const *ptr) {
		return *ptr;
	}

	constexpr static inline void unaligned_store_wrapper_128(__m128i_u *ptr, __m128i_u data) {
		*ptr = data;
	}
}// namespace internal


namespace cryptanalysislib {
	struct _uint16x8_t;
	struct _uint32x4_t;
	struct _uint64x2_t;

	struct _uint8x16_t {
		constexpr static uint32_t LIMBS = 16;
		using limb_type = uint8_t;

		constexpr inline _uint8x16_t &operator=(const _uint16x8_t &b) noexcept;
		constexpr inline _uint8x16_t &operator=(const _uint32x4_t &b) noexcept;
		constexpr inline _uint8x16_t &operator=(const _uint64x2_t &b) noexcept;

		constexpr _uint8x16_t() noexcept = default;
		constexpr _uint8x16_t(const _uint16x8_t &b) noexcept;
		constexpr _uint8x16_t(const _uint32x4_t &b) noexcept;
		constexpr _uint8x16_t(const _uint64x2_t &b) noexcept;

		union {
			// compatibility to `TxN_t`
			uint8_t d[16];

			uint8_t v8[16];
			uint16_t v16[8];
			uint32_t v32[4];
			uint64_t v64[2];

			__m128i v128;
		};

		[[nodiscard]] constexpr inline uint32_t size() const noexcept {
			return 16;
		}


		[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
			ASSERT(i < LIMBS);
			return d[i];
		}

		[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
			ASSERT(i < LIMBS);
			return d[i];
		}

		///
		/// \return
		static inline _uint8x16_t random() noexcept {
			_uint8x16_t ret;
			for (uint32_t i = 0; i < 2; i++) {
				ret.v64[i] = fastrandombytes_uint64();
			}

			return ret;
		}

		[[nodiscard]] constexpr static inline _uint8x16_t set1(const uint8_t i) noexcept {
			_uint8x16_t ret;
			for (uint32_t j = 0; j < 16u; ++j) {
				ret.v8[j] = i;
			}
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint8x16_t set(
		        uint32_t a, uint32_t b, uint32_t c, uint32_t d) noexcept {
			_uint8x16_t ret;
			ret.v32[0] = d;
			ret.v32[1] = c;
			ret.v32[2] = b;
			ret.v32[3] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint8x16_t setr(
		        uint32_t a, uint32_t b, uint32_t c, uint32_t d) noexcept {
			_uint8x16_t ret;
			ret.v32[0] = a;
			ret.v32[1] = b;
			ret.v32[2] = c;
			ret.v32[3] = d;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint8x16_t set(
		        uint8_t a, uint8_t b, uint8_t c, uint8_t d,
		        uint8_t e, uint8_t f, uint8_t g, uint8_t h,
		        uint8_t i, uint8_t j, uint8_t k, uint8_t l,
		        uint8_t m, uint8_t n, uint8_t o, uint8_t p) noexcept {
			_uint8x16_t ret;
			ret.v8[0] = p;
			ret.v8[1] = o;
			ret.v8[2] = n;
			ret.v8[3] = m;
			ret.v8[4] = l;
			ret.v8[5] = k;
			ret.v8[6] = j;
			ret.v8[7] = i;
			ret.v8[8] = h;
			ret.v8[9] = g;
			ret.v8[10] = f;
			ret.v8[11] = e;
			ret.v8[12] = d;
			ret.v8[13] = c;
			ret.v8[14] = b;
			ret.v8[15] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint8x16_t setr(
		        uint8_t a, uint8_t b, uint8_t c, uint8_t d,
		        uint8_t e, uint8_t f, uint8_t g, uint8_t h,
		        uint8_t i, uint8_t j, uint8_t k, uint8_t l,
		        uint8_t m, uint8_t n, uint8_t o, uint8_t p) noexcept {
			_uint8x16_t ret;
			ret.v8[0] = a;
			ret.v8[1] = b;
			ret.v8[2] = c;
			ret.v8[3] = d;
			ret.v8[4] = e;
			ret.v8[5] = f;
			ret.v8[6] = g;
			ret.v8[7] = h;
			ret.v8[8] = i;
			ret.v8[9] = j;
			ret.v8[10] = k;
			ret.v8[11] = l;
			ret.v8[12] = m;
			ret.v8[13] = n;
			ret.v8[14] = o;
			ret.v8[15] = p;
			return ret;
		}

		///
		/// \tparam aligned
		/// \param ptr
		/// \return
		template<const bool aligned = false>
		[[nodiscard]] constexpr static inline _uint8x16_t load(const void *ptr) noexcept {
			if constexpr (aligned) {
				return aligned_load(ptr);
			}

			return unaligned_load(ptr);
		}

		///
		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline _uint8x16_t aligned_load(const void *ptr) noexcept {
			auto *ptr128 = (__m128i *) ptr;
			_uint8x16_t out;
			out.v128 = *ptr128;
			return out;
		}


		///
		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline _uint8x16_t unaligned_load(const void *ptr) noexcept {
			__m128i_u const *ptr128 = (__m128i_u const *) ptr;
			const __m128i_u tmp = internal::unaligned_load_wrapper_128(ptr128);
			_uint8x16_t out;
			out.v128 = tmp;
			return out;
		}

		///
		/// \tparam aligned
		/// \param ptr
		/// \param in
		template<const bool aligned = false>
		constexpr static inline void store(void *ptr, const _uint8x16_t in) noexcept {
			if constexpr (aligned) {
				aligned_store(ptr, in);
				return;
			}

			unaligned_store(ptr, in);
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void aligned_store(void *ptr, const _uint8x16_t in) noexcept {
			auto *ptr128 = (__m128i *) ptr;
			*ptr128 = in.v128;
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void unaligned_store(void *ptr, const _uint8x16_t in) noexcept {
			auto *ptr128 = (__m128i_u *) ptr;
			internal::unaligned_store_wrapper_128(ptr128, in.v128);
		}

		/// \param in1
		/// \param in2
		/// \return in1 > in2 compressed
		[[nodiscard]] constexpr static inline uint32_t gt(const _uint8x16_t in1,
		                                                  const _uint8x16_t in2) noexcept {
			const __m128i tmp = (__m128i) ((__v16qu) in1.v128 > (__v16qu) in2.v128);
			return __builtin_ia32_pmovmskb128((__v16qi) tmp);
		}

		/// NOTE: signed comparison
		/// \param in1
		/// \param in2
		/// \return in1 > in2 compressed
		[[nodiscard]] constexpr static inline uint32_t lt(const _uint8x16_t in1,
		                                                  const _uint8x16_t in2) noexcept {
			const __m128i tmp = (__m128i) ((__v16qu) in1.v128 < (__v16qu) in2.v128);
			return __builtin_ia32_pmovmskb128((__v16qi) tmp);
		}

		///
		/// \param in1
		/// \param in2
		/// \return in1 == in2 compressed
		[[nodiscard]] constexpr static inline uint32_t cmp(const _uint8x16_t in1,
		                                                   const _uint8x16_t in2) noexcept {
			const __m128i tmp = (__m128i) ((__v16qu) in1.v128 == (__v16qu) in2.v128);
			return __builtin_ia32_pmovmskb128((__v16qi) tmp);
		}
	};

	struct _uint16x8_t {
		constexpr static uint32_t LIMBS = 8;
		using limb_type = uint16_t;

		constexpr inline _uint16x8_t &operator=(const _uint8x16_t &b) noexcept;
		constexpr inline _uint16x8_t &operator=(const _uint32x4_t &b) noexcept;
		constexpr inline _uint16x8_t &operator=(const _uint64x2_t &b) noexcept;

		constexpr _uint16x8_t() noexcept = default;
		constexpr _uint16x8_t(const _uint8x16_t &b) noexcept;
		constexpr _uint16x8_t(const _uint32x4_t &b) noexcept;
		constexpr _uint16x8_t(const _uint64x2_t &b) noexcept;

		union {
			// compatibility to `TxN_t`
			uint16_t d[8];

			uint8_t v8[16];
			uint16_t v16[8];
			uint32_t v32[4];
			uint64_t v64[2];

			__m128i v128;
		};

		[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
			ASSERT(i < LIMBS);
			return d[i];
		}

		[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
			ASSERT(i < LIMBS);
			return d[i];
		}

		///
		/// \return
		static inline _uint16x8_t random() noexcept {
			_uint16x8_t ret;
			for (uint32_t i = 0; i < 2; i++) {
				ret.v64[i] = fastrandombytes_uint64();
			}

			return ret;
		}

		[[nodiscard]] constexpr static inline _uint16x8_t set(
		        uint32_t a, uint32_t b, uint32_t c, uint32_t d) noexcept {
			_uint16x8_t ret;
			ret.v32[0] = d;
			ret.v32[1] = c;
			ret.v32[2] = b;
			ret.v32[3] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint16x8_t setr(
		        uint32_t a, uint32_t b, uint32_t c, uint32_t d) noexcept {
			_uint16x8_t ret;
			ret.v32[0] = a;
			ret.v32[1] = b;
			ret.v32[2] = c;
			ret.v32[3] = d;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint16x8_t set(
		        uint16_t a, uint16_t b, uint16_t c, uint16_t d,
		        uint16_t e, uint16_t f, uint16_t g, uint16_t h) noexcept {
			_uint16x8_t ret;
			ret.v16[0] = h;
			ret.v16[1] = g;
			ret.v16[2] = f;
			ret.v16[3] = e;
			ret.v16[4] = d;
			ret.v16[5] = c;
			ret.v16[6] = b;
			ret.v16[7] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint16x8_t setr(
		        uint8_t a, uint8_t b, uint8_t c, uint8_t d,
		        uint8_t e, uint8_t f, uint8_t g, uint8_t h) noexcept {
			_uint16x8_t ret;
			ret.v8[0] = a;
			ret.v8[1] = b;
			ret.v8[2] = c;
			ret.v8[3] = d;
			ret.v8[4] = e;
			ret.v8[5] = f;
			ret.v8[6] = g;
			ret.v8[7] = h;
			return ret;
		}

		///
		/// \tparam aligned
		/// \param ptr
		/// \return
		template<const bool aligned = false>
		[[nodiscard]] constexpr static inline _uint16x8_t load(const void *ptr) noexcept {
			if constexpr (aligned) {
				return aligned_load(ptr);
			}

			return unaligned_load(ptr);
		}

		///
		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline _uint16x8_t aligned_load(const void *ptr) noexcept {
			auto *ptr128 = (__m128i *) ptr;
			_uint16x8_t out;
			out.v128 = *ptr128;
			return out;
		}


		///
		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline _uint16x8_t unaligned_load(const void *ptr) noexcept {
			__m128i_u const *ptr128 = (__m128i_u const *) ptr;
			const __m128i_u tmp = internal::unaligned_load_wrapper_128(ptr128);
			_uint16x8_t out;
			out.v128 = tmp;
			return out;
		}

		///
		/// \tparam aligned
		/// \param ptr
		/// \param in
		template<const bool aligned = false>
		constexpr static inline void store(void *ptr, const _uint16x8_t in) noexcept {
			if constexpr (aligned) {
				aligned_store(ptr, in);
				return;
			}

			unaligned_store(ptr, in);
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void aligned_store(void *ptr, const _uint16x8_t in) noexcept {
			auto *ptr128 = (__m128i *) ptr;
			*ptr128 = in.v128;
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void unaligned_store(void *ptr, const _uint16x8_t in) noexcept {
			auto *ptr128 = (__m128i_u *) ptr;
			internal::unaligned_store_wrapper_128(ptr128, in.v128);
		}
	};

	struct _uint32x4_t {
		constexpr static uint32_t LIMBS = 4;
		using limb_type = uint32_t;

		constexpr inline _uint32x4_t &operator=(const _uint8x16_t &b) noexcept;
		constexpr inline _uint32x4_t &operator=(const _uint16x8_t &b) noexcept;
		constexpr inline _uint32x4_t &operator=(const _uint64x2_t &b) noexcept;

		constexpr _uint32x4_t() noexcept = default;
		constexpr _uint32x4_t(const _uint8x16_t &b) noexcept;
		constexpr _uint32x4_t(const _uint16x8_t &b) noexcept;
		constexpr _uint32x4_t(const _uint64x2_t &b) noexcept;

		union {
			uint8_t v8[16];
			uint16_t v16[8];
			uint32_t v32[4];
			uint64_t v64[2];
			__m128i v128;
		};

		[[nodiscard]] constexpr static inline _uint32x4_t set1(uint32_t a) {
			_uint32x4_t ret;
			ret.v32[0] = a;
			ret.v32[1] = a;
			ret.v32[2] = a;
			ret.v32[3] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint32x4_t set(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
			_uint32x4_t ret;
			ret.v32[0] = d;
			ret.v32[1] = c;
			ret.v32[2] = b;
			ret.v32[3] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint32x4_t setr(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
			_uint32x4_t ret;
			ret.v32[0] = a;
			ret.v32[1] = b;
			ret.v32[2] = c;
			ret.v32[3] = d;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint32x4_t set(uint64_t a, uint64_t b) {
			_uint32x4_t ret;
			ret.v64[0] = b;
			ret.v64[1] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint32x4_t setr(uint64_t a, uint64_t b) {
			_uint32x4_t ret;
			ret.v64[0] = a;
			ret.v64[1] = b;
			return ret;
		}
	};

	struct _uint64x2_t {
		constexpr static uint32_t LIMBS = 2;
		using limb_type = uint64_t;

		constexpr inline _uint64x2_t &operator=(const _uint8x16_t &b) noexcept;
		constexpr inline _uint64x2_t &operator=(const _uint16x8_t &b) noexcept;
		constexpr inline _uint64x2_t &operator=(const _uint32x4_t &b) noexcept;

		constexpr _uint64x2_t() noexcept = default;
		constexpr _uint64x2_t(const _uint8x16_t &b) noexcept;
		constexpr _uint64x2_t(const _uint16x8_t &b) noexcept;
		constexpr _uint64x2_t(const _uint32x4_t &b) noexcept;

		union {
			uint8_t v8[16];
			uint16_t v16[8];
			uint32_t v32[4];
			uint64_t v64[2];
			__m128i v128;
		};

		[[nodiscard]] constexpr static inline _uint64x2_t set1(uint64_t a) {
			_uint64x2_t ret;
			ret.v64[0] = a;
			ret.v64[1] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint64x2_t set(uint64_t a, uint64_t b) {
			_uint64x2_t ret;
			ret.v64[0] = b;
			ret.v64[1] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint64x2_t setr(uint64_t a, uint64_t b) {
			_uint64x2_t ret;
			ret.v64[0] = a;
			ret.v64[1] = b;
			return ret;
		}

		///
		/// \tparam aligned
		/// \param ptr
		/// \return
		template<const bool aligned = false>
		[[nodiscard]] constexpr static inline _uint64x2_t load(const void *ptr) noexcept {
			if constexpr (aligned) {
				return aligned_load(ptr);
			}

			return unaligned_load(ptr);
		}

		///
		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline _uint64x2_t aligned_load(const void *ptr) noexcept {
			auto *ptr128 = (__m128i *) ptr;
			_uint64x2_t out;
			out.v128 = *ptr128;
			return out;
		}


		///
		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline _uint64x2_t unaligned_load(const void *ptr) noexcept {
			__m128i_u const *ptr128 = (__m128i_u const *) ptr;
			const __m128i_u tmp = internal::unaligned_load_wrapper_128(ptr128);
			_uint64x2_t out;
			out.v128 = tmp;
			return out;
		}

		///
		/// \tparam aligned
		/// \param ptr
		/// \param in
		template<const bool aligned = false>
		constexpr static inline void store(void *ptr, const _uint64x2_t in) noexcept {
			if constexpr (aligned) {
				aligned_store(ptr, in);
				return;
			}

			unaligned_store(ptr, in);
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void aligned_store(void *ptr, const _uint64x2_t in) noexcept {
			auto *ptr128 = (__m128i *) ptr;
			*ptr128 = in.v128;
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void unaligned_store(void *ptr, const _uint64x2_t in) noexcept {
			auto *ptr128 = (__m128i_u *) ptr;
			internal::unaligned_store_wrapper_128(ptr128, in.v128);
		}
	};
}// namespace cryptanalysislib


constexpr static __m256i u8tom256(const uint8_t t[32]) noexcept {
	long long __t[4];
	__t[0] = (long long) t[0] | (((long long) t[1]) << 8) | ((long long) t[2] << 16) | ((long long) t[3] << 24) | ((long long) t[4] << 32) | ((long long) t[5] << 40) | ((long long) t[6] << 48) | ((long long) t[7] << 56);
	__t[1] = (long long) t[8] | (((long long) t[9]) << 8) | ((long long) t[10] << 16) | ((long long) t[11] << 24) | ((long long) t[12] << 32) | ((long long) t[13] << 40) | ((long long) t[14] << 48) | ((long long) t[15] << 56);
	__t[2] = (long long) t[16] | (((long long) t[17]) << 8) | ((long long) t[18] << 16) | ((long long) t[19] << 24) | ((long long) t[20] << 32) | ((long long) t[21] << 40) | ((long long) t[22] << 48) | ((long long) t[23] << 56);
	__t[3] = (long long) t[24] | (((long long) t[25]) << 8) | ((long long) t[26] << 16) | ((long long) t[27] << 24) | ((long long) t[28] << 32) | ((long long) t[29] << 40) | ((long long) t[30] << 48) | ((long long) t[31] << 56);
	__m256i tmp = {__t[0], __t[1], __t[2], __t[3]};
	return tmp;
}

constexpr static __m256i u16tom256(const uint16_t t[16]) noexcept {
	long long __t[4];
	__t[0] = (long long) t[0] | (((long long) t[1]) << 16) | ((long long) t[2] << 32) | ((long long) t[3] << 48);
	__t[1] = (long long) t[4] | (((long long) t[5]) << 16) | ((long long) t[6] << 32) | ((long long) t[7] << 48);
	__t[2] = (long long) t[8] | (((long long) t[9]) << 16) | ((long long) t[10] << 32) | ((long long) t[11] << 48);
	__t[3] = (long long) t[12] | (((long long) t[13]) << 16) | ((long long) t[14] << 32) | ((long long) t[15] << 48);
	__m256i tmp = {__t[0], __t[1], __t[2], __t[3]};
	return tmp;
}

constexpr static __m256i u32tom256(const uint32_t t[8]) noexcept {
	long long __t[4];
	__t[0] = (long long) t[0] | (((long long) t[1]) << 32);
	__t[1] = (long long) t[2] | (((long long) t[3]) << 32);
	__t[2] = (long long) t[4] | (((long long) t[5]) << 32);
	__t[3] = (long long) t[6] | (((long long) t[7]) << 32);
	__m256i tmp = {__t[0], __t[1], __t[2], __t[3]};
	return tmp;
}

constexpr static __m256i u64tom256(const uint64_t t[4]) noexcept {
	__m256i tmp = {(long long) t[0], (long long) t[1], (long long) t[2], (long long) t[3]};
	return tmp;
}

/// TODO not working
constexpr static void m256tou16(uint16_t t[16], const __m256i m) noexcept {
	const __v4di mm = m;
	long long d0 = mm[0], d1 = 1, d2 = 2, d3 = 3;
	t[0] = d0;
	t[1] = d0 >> 16;
	t[2] = d0 >> 32;
	t[3] = d0 >> 48;
	t[4] = d1;
	t[5] = d1 >> 16;
	t[6] = d1 >> 32;
	t[7] = d1 >> 48;
	t[8] = d2;
	t[9] = d2 >> 16;
	t[10] = d2 >> 32;
	t[11] = d2 >> 48;
	t[12] = d3;
	t[13] = d3 >> 16;
	t[14] = d3 >> 32;
	t[15] = d3 >> 48;
}

// needed forward decl
template<typename T, const uint32_t N>
#if __cplusplus > 201709L
    requires std::is_integral_v<T>
#endif
class TxN_t;


struct uint8x32_t {
	constexpr static uint32_t LIMBS = 32;
	using limb_type = uint8_t;
	using S = uint8x32_t;

	union {
		// compatibility with TxN_t
		uint8_t d[32];

		uint8_t v8[32];
		uint16_t v16[16];
		uint32_t v32[8];
		uint64_t v64[4];
		__m256i v256;
	};

	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const {
		ASSERT(i < LIMBS);
		return d[i];
	}

	[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) {
		ASSERT(i < LIMBS);
		return d[i];
	}

	/// Example of how the constexpr implementation works:
	/// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGEgBykrgAyeAyYAHI%2BAEaYxCCSZqQADqgKhE4MHt6%2BASlpGQKh4VEssfGJtpj2jgJCBEzEBNk%2BflyBdpgOmfWNBMWRMXEJSQoNTS257bbj/WGDZcOJAJS2qF7EyOwc5gDMYcjeWADUJrtuY/iCAHQIZ9gmGgCCewdHmKfneCwsYQTEYVu90eLzM%2BwYhy8JzObmQlycwOeILGxC8DmOXj%2B/lUuzMAH0CKcAOxWJEaACcXgYmWJpJeFMelKxBOOxwAbv5TgBWCy4kxcgAiZzpjMZmMEXAAbCy2VL%2BRY5YLhSDRRTxQRcTK%2BTzjv5%2BULdiKGWq/pLJDKzfLjpJ9cqyeTGXi8SwzFzJXh2a7JXb6Q6iQa6azWSCg8c0AwxphVMliMcxkxHMhjmFaPMMVicfjCaoSHiIOGxunBNjNYSwlxSCqKaGa7W6/WC4T1SWs8mGGZlrSqw6TcXMyz1gQfaryYPrmyvZ8BccIE6XW68J2oE62ZJ0F5luXx5P%2BdgZyu1xuwmZt27lsPjeSAPRXscTt1T47O%2BfSnPEPHpL0QLf3yWkNsnr%2B56Gt2jLEJgBAbAwxyDhefpCmSvo3le3bIdybjJI0rDHMk/yoTe6HgZBxAMN2jZRjGcYNImbapuERYEC2LJUmIeDAOE6B4rQqBMOg%2BYCIWbKoHg6DHAAVLhxCdiYJKgRSTBeEQ4mSTuuzTrOzpeh6YnLJJcFihmpYwYp%2BkUneqnThJ/xeqZ163opp6So%2Bz5elxPHrh%2BeBfhpL7abp/zAUaPbkkRUHGUOIH2jJCFPKGIZBsh%2BFXuhmHENhen2mh/IYVhLBtmRAkEBRsbxjRKZpkJIkYswqbsZgnFjCQmAQJVolWWQYaFQxTFlgw0myVFFJzlpynWQ%2BZzqcNC7if5xC2e1Flto5Ppxf6KprZF9JPOR0axq1xxFWMEDNiyYmDv%2Bjbdad5YXV1J2EmJx63RGTZ/EZfVds8ob8LGx1va2HoTccGjCsm1i4p8bjHKRhpg5YE24v1gZ1pdzb9oSTCPmjpYgCALG1Rxbm8d%2BDBcKcljJmJiMrSjd2Ga20RY/TBK4/jbGE9xxPHuTFiU9Tm0NnTfZGUmQPY1muNvnmTD/tEgXxbW4ss3jNXs/VH5EOBEDnWG8tfUG0XrTFIJ/McLBMGEEBIwrnUvcVDHvY%2BXBmHqAtBjtlH3eF8qkYKj4yRYIMbcjrIe7GXvlj7%2Br%2BySQcBjbYdXb1ZhR37QMB3HF6xUGh0EFAXs6TrN0AaQfU06yMZ/FQEDmCnrroNlpFmEk3s8iDgp6y8/ocKstCcFyvB%2BBwWikKgnBuNY1hxusmwfHsPCkAQmg96sADWIC7Ls1waP4kiSkSkhEv4GhH0ff59xwkiD8vo%2BcLwCggBoi/L6scCwEgaAsMkdBxOQlCf9/eg8RgBcFxKQLAbI8BbAAGp4EwAAdwAPLJEYJwBeNBaBFWIA/CA0Qb7RDCI0AAnmg3gBDmDECIYg6I2guhL24LwT%2BbBBCIIYLQEhw9eBYGiF4YAbgxC0Afgw8BmBzZGHEJw8BeBwLdDZJgIRI8oxdEUtsBefxqg31TNENKlCPBYBvv8b4pDVhUAMMABQsCEHINQcI/gggRBiHYFIGQghFAqHUJI3QFYDBGBQJPSw%2Bg8DRAfpAVYqBcKZCEQAWnNmyVQZhjhRMQbsXgqA5HEABFgEJVsqg1EyC4Bg7hPCtAkESIIhSBilHKBIJ%2BqR0i1CyMUqYZS6mFAYJUoY8QuBP06N0OosxJhtDKb0hpvQmgdMWF0npAymlDJmH0CZ1TumrAUDPLYEhe792vpIseHBjiqH8JKKJZpjjAGQEmUB1wEkQFwIQEg5NdhcGWLwehWhlhrw3lvLk/guDknJO0fw3yj4/OkBfK%2Bpcb67Pvo/Z%2BnDX4wEQCAQcyRFJ/34l/H%2BxAIisG2Aco5JyzkXK3mYXg9U7mZL0HY4QohxDOKpW4tQN8vGkHgWlZIxj9BbIhTszgiDFIosJKgKg%2BzDnHMkKc85xxLnXI8BioBDynkvJfh8ze1xN7qo1Zqzll9tkjyhbYGFryV7apJdyvVd9YVvNWOk9IzhJBAA%3D
	constexpr inline uint8x32_t() noexcept = default;

	/// NOTE: currently cannot be constexpr
	/// \return a uniform random element
	[[nodiscard]] static inline uint8x32_t random() noexcept {
		uint8x32_t ret;
		for (size_t i = 0; i < 4; ++i) {
			ret.v64[i] = fastrandombytes_uint64();
		}
		return ret;
	}

	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false,
	                            bool hex = false) const;

	///
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t set(char __q31, char __q30, char __q29, char __q28,
	                                                     char __q27, char __q26, char __q25, char __q24,
	                                                     char __q23, char __q22, char __q21, char __q20,
	                                                     char __q19, char __q18, char __q17, char __q16,
	                                                     char __q15, char __q14, char __q13, char __q12,
	                                                     char __q11, char __q10, char __q09, char __q08,
	                                                     char __q07, char __q06, char __q05, char __q04,
	                                                     char __q03, char __q02, char __q01, char __q00) noexcept {
		uint8x32_t out;
		out.v256 = __extension__(__m256i)(__v32qi){
		        __q00, __q01, __q02, __q03, __q04, __q05, __q06, __q07,
		        __q08, __q09, __q10, __q11, __q12, __q13, __q14, __q15,
		        __q16, __q17, __q18, __q19, __q20, __q21, __q22, __q23,
		        __q24, __q25, __q26, __q27, __q28, __q29, __q30, __q31};

		return out;
	}

	///
	[[nodiscard]] constexpr static inline uint8x32_t setr(char __q31, char __q30, char __q29, char __q28,
	                                                      char __q27, char __q26, char __q25, char __q24,
	                                                      char __q23, char __q22, char __q21, char __q20,
	                                                      char __q19, char __q18, char __q17, char __q16,
	                                                      char __q15, char __q14, char __q13, char __q12,
	                                                      char __q11, char __q10, char __q09, char __q08,
	                                                      char __q07, char __q06, char __q05, char __q04,
	                                                      char __q03, char __q02, char __q01, char __q00) noexcept {
		return set(__q00, __q01, __q02, __q03, __q04, __q05, __q06, __q07, __q08,
		           __q09, __q10, __q11, __q12, __q13, __q14, __q15, __q16, __q17,
		           __q18, __q19, __q20, __q21, __q22, __q23, __q24, __q25, __q26,
		           __q27, __q28, __q29, __q30, __q31);
	}

	/// sets all 32 8bit limbs to `a`
	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t set1(const uint8_t a) noexcept {
		uint8x32_t out;
		out = uint8x32_t::set(a, a, a, a, a, a, a, a,
		                      a, a, a, a, a, a, a, a,
		                      a, a, a, a, a, a, a, a,
		                      a, a, a, a, a, a, a, a);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline uint8x32_t load(const uint8_t *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	/// the is `is_constant_evaluated()' is removed with `-O3`
	/// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIM6SuADJ4DJgAcj4ARpjEIACsABykAA6oCoRODB7evv6p6ZkCIWGRLDFxSbaY9o4CQgRMxAQ5Pn4BdpgOWQ1NBCUR0bEJyQqNza15HeP9oYPlw0kAlLaoXsTI7BzmAMyhyN5YANQmO254LCyhBMShAHQIp9gmGgCCu/uHmCdnY/iCDyeL3eZj2DAOXmOpzcBAAnilMAB9G5MQgKIFvYFoBhjTCqFLEI5jJiOZBHRGIlhmeIANjwRy8iSIVNpEGxYwZ10SyKOBBM8QsOzM/IAIksjgxUHjNikCCcAOxWTEaACctAEwCO6qM5OR/IsklFpyV71VFL5Ao0op%2BIqOEG1modSwtFiOVvitpM8rcdqgDq1Gud%2BqOXFF4uh0KOiXD3t9/qdLqOwo9Pzcka4NJjPr9GoDRiDAqOO2tEbOSckWbjuYTwcNKdLPqFlZzOprhfiJbOkckGmb9urgcTNM7abLkmjCuz/dbg%2BD8pHkdpS2NwJeKvN%2BtD9Z2tunjtnhcSJdjUD3eeABddKrDqcjE69U/jB4sXHdnq7ZYzfaf%2BZdXC376jj6ZgVpOVYzr%2Bm7JoBkZNmBLb7pBApcMW25AUcPbfgOSEvnWMFjveJ5nm2L4dmhi6ZiuyrrnqArQTado/hef7DvWRHEc%2BXDzh64Yfj6hGPthzGbke5Gfpm8EcThXDXmJwGgQ%2B4GIcJdFvreZZwYpCHnpeZgAepPqYZJTG6fRDYYQJSk6S6oILmWS5UaaNEuqh767iZNl4ZJUkqRY1I3uZlnaSRZisfhPpfsZQm6dx4XllhEG%2BWYolxZpREefqZiyXFRlaT5l47Gp5njglykFfp5kOTsJprhSLJ0ryLApAxXoWBulqiqQ7Uvp13XQV1tGCka8oio5a7EJgBDrAwjUpGNI2rsqrxjMQXgOJygiJKoQo8q1i0quyBB4gSRKNKSG0EDtcpBAAkgAshYQgMUKY2ql4GSthcUTIvC3ynLaXhcnq1X7e9oSak9/0XVtV2vU5XgMFkCo1aqa4APRo0caBNSSeBRHQhCwkcADuhAIEcAAqqjhHq1FroDm08ug%2BovR6cNrvTQNygAbqJgrQa915vdcGY8tzGabmFguc4IV1HNzrMWCl0vC4INKSGL6v6nhKs0fV9Lc9Sw4g3TC0m6aTkY2j%2B1Wyc8RuCkTSsEcsrEDbGN224E1TcQDD7fq%2BqSvgCiiMQzNsymh3HYSxLnaEtBzND21mDyYh4MAYToIi6pMOgbICByDMENycoAFSu%2BKkrSpgsrI/7qp4FQdp/CAIB4AoiKHYYBCIpg3NiF4JKYHnSwxijQsTx7oS8gg3xR/ihKiEoRIXCktBE%2BnkoTTPJIz986AkkwRzt0cacZ8P9eT5jhAk3QtASqgcoIEw3PfIYRPmGYq2CBcmCfy7jAxCOEwOiOmqNVSHV1PrWaDFGTMiNhACuus1xPTWHyc2HNwEqjQXcQ2tIGIECasg1U3tppHDQcgkaRwajLz2mAieTAvBECOOXG4RsGIQDqkbekpdxSu2ISqVBTCBE4LwTSBirDiBGwEaQ325DhEYKwV6UapsVHvDNijV4RcYYpzlEdMYXB844jlNonkkix7AiOFYo4siZraOTsiVuZ9M7Z1QLnRBNxlwYIWkteeJ17Fy30QQMwRjC5cxYRXOubxrE2MmmQgJuinEJ3PlnHOeckHeJFBwFYtBODxF4H4bgvBUCcDTJYawRI1gbD%2BqCHgpACCaGySsAA1iAHYKo7g7HlBoGkKpEgoS4jSDQGhki5I4JIApjTSAlI4LwBQIAND1MaSsOAsAkDYxSHQWI5BKAbK2XEYAUguB8DoEdYg8yIBRCmfjZgxBYScDqTcposIADyURtBdAaUU0g2M2CCBeQwdeUysBRC8MANwYhaDzO%2BVgFghhgDiA4FoUg%2BAJrdFftC5FeIuhMK2HU64NQpkJyiMQZ5HgsBTJuBcB5vBX7ECiOkTAIpMBwqMAnIwyy%2BAGGAAoAAangTAxMXkIkKXU/gggRBiHYFIGQghFAqHUEinQegDActMOUyw%2Bg8bzMgCsVAsosjQoALQvJ2EcI1cLuaqHNSwKgcKbVRCuGYG1L9VDOvNX8f6GcvDWGsGYZFqA6W3CwDqiAKxOjdGcBAVwkw/DHOCHMMoFQ9BpAyHUbIng2gpsKOmgYSbhjHIjem3oExM15ELTUT5PQZh5qGHEQtMxY16GJM0WtCx63hqqZsCQOS8mTKVdMzgRxVCJBpEa9WRxgDIDJFIO4XA7S4EICQE4tSli8C%2BVoUepBWlmHlHcLK8QVQ0kSPEHYkgVQ9MPQEMZEzSCFIDZwOZCyllKpWTARAIA0EpCYTs/OTV9nhFYFsEdY6J1TpnZIOdvBh5LuDXocVwhRDiBlQh%2BVagpm6GOcTUlKQaW9o4Pku9UyZkvKYd%2BuUqAm4gfHZISd06QyQfnRADw/76CEl2FwNdL7N0tLaTsTpOxBNCeE0J/QnBb33uKY%2B2wz6N1NLExwMw/aH2zO4/JulGRnCSCAA%3D
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t aligned_load(const uint8_t *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			// in the constexpr case simply ignore that the data is aligned
			// it will not have any "runtime" penalties
			const __m256i tmp = u8tom256(ptr);
			S out;
			out.v256 = tmp;
			return out;
		} else {
			auto *ptr256 = (__m256i *) ptr;
			S out;
			out.v256 = *ptr256;
			return out;
		}
	}

	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t unaligned_load(const uint8_t *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			const __m256i tmp = u8tom256(ptr);
			S out;
			out.v256 = tmp;
			return out;
		} else {
			S out;
			out.v256 = internal::unaligned_load_wrapper((__m256i_u *) ptr);
			return out;
		}
	}

	/// NOTE: the store can never be constexpr ans its needs to access
	/// given memory
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	static inline void store(void *ptr, const uint8x32_t in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		unaligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	static inline void aligned_store(void *ptr, const uint8x32_t in) noexcept {
		auto *ptr256 = (__m256i *) ptr;
		*ptr256 = in.v256;
	}

	///
	/// \param ptr
	/// \param in
	static inline void unaligned_store(void *ptr, const uint8x32_t in) noexcept {
		auto *ptr256 = (__m256i_u *) ptr;
		internal::unaligned_store_wrapper(ptr256, in.v256);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t xor_(const uint8x32_t in1,
	                                                      const uint8x32_t in2) noexcept {
		uint8x32_t out;
		out.v256 = (__m256i) ((__v4du) in1.v256 ^ (__v4du) in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t and_(const uint8x32_t in1,
	                                                      const uint8x32_t in2) noexcept {
		uint8x32_t out;
		out.v256 = (__m256i) ((__v4du) in1.v256 & (__v4du) in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t or_(const uint8x32_t in1,
	                                                     const uint8x32_t in2) noexcept {
		uint8x32_t out;
		out.v256 = (__m256i) ((__v4du) in1.v256 | (__v4du) in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t andnot(const uint8x32_t in1,
	                                                        const uint8x32_t in2) noexcept {
		uint8x32_t out;
#ifndef __clang__
		out.v256 = (__m256i) __builtin_ia32_andnotsi256((__v4di) in1.v256, (__v4di) in2.v256);
#else
		out.v256 = (__m256i) (~(__v4du) in1.v256 & (__v4du) in2.v256);
#endif
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t not_(const uint8x32_t in1) noexcept {
		uint8x32_t out;
		const uint8x32_t minus_one = set1(-1);
		out.v256 = (__m256i) ((__v4du) in1.v256 ^ (__v4du) minus_one.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t add(const uint8x32_t in1,
	                                                     const uint8x32_t in2) noexcept {
		uint8x32_t out;
		out.v256 = (__m256i) ((__v32qu) in1.v256 + (__v32qu) in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t sub(const uint8x32_t in1,
	                                                     const uint8x32_t in2) noexcept {
		uint8x32_t out;
		out.v256 = (__m256i) ((__v32qu) in1.v256 - (__v32qu) in2.v256);
		return out;
	}

	/// 8 bit mul lo
	/// \param in1 first input
	/// \param in2
	/// \return in1*in2
	[[nodiscard]] constexpr static inline uint8x32_t mullo(const uint8x32_t in1,
	                                                       const uint8x32_t in2) noexcept {
		uint8x32_t out;
		out.v256 = ((__m256i) ((__v32qu) in1.v256 * (__v32qu) in2.v256));
		return out;
		const __m256i maskl = __extension__(__m256i)(__v16hi){0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
		const __m256i maskh = __extension__(__m256i)(__v16hi){(short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00};

		const __m256i in1l = (__m256i) ((__v4du) in1.v256 & (__v4du) maskl);
		const __m256i in2l = (__m256i) ((__v4du) in2.v256 & (__v4du) maskl);
		const __m256i in1h = (__m256i) ((__v4du) in1.v256 & (__v4du) maskh);
		const __m256i in2h = (__m256i) ((__v4du) in2.v256 & (__v4du) maskh);


		out.v256 = ((__m256i) ((__v16hu) in1l * (__v16hu) in2l)) & maskl;
		out.v256 ^= ((__m256i) ((__v16hu) in1h * (__v16hu) in2h)) & maskh;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t mullo(const uint8x32_t in1,
	                                                       const uint8_t in2) noexcept {
		const uint8x32_t rs = uint8x32_t::set1(in2);
		return uint8x32_t::mullo(in1, rs);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t slli(const uint8x32_t in1,
	                                                      const uint8_t in2) noexcept {
		ASSERT(in2 <= 8);
		uint8x32_t out;
		const uint8x32_t mask = set1((1u << in2) - 1u);
		out = uint8x32_t::and_(in1, mask);
		// if (std::is_constant_evaluated()) {
		// 	out.v256 = (__m256i)((__v32qi)out.v256) << in2;
		// 	return out;
		// }
		// out.v256 = (__m256i) __builtin_ia32_psllwi256((__v16hi) out.v256, in2);

		out.v256 = (__m256i) ((__v32qi) out.v256) << in2;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t srli(const uint8x32_t in1,
	                                                      const uint8_t in2) noexcept {
		ASSERT(in2 <= 8);
		const uint8x32_t mask1 = set1(((1u << (8u - in2)) - 1u) << in2);
		const uint8x32_t mask2 = set1((1u << (8u - in2)) - 1u);
		uint8x32_t out = uint8x32_t::and_(in1, mask1);
		if (std::is_constant_evaluated()) {
			out.v256 = (__m256i) ((__v32qi) out.v256) >> in2;
			return out;
		}
		out.v256 = (__m256i) __builtin_ia32_psrlwi256((__v16hi) out.v256, in2);
		out = uint8x32_t::and_(out, mask2);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return in1 > in2 uncompressed
	[[nodiscard]] constexpr static inline uint8x32_t gt_(const uint8x32_t in1,
	                                                     const uint8x32_t in2) noexcept {
		uint8x32_t ret;
		ret.v256 = (__m256i) ((__v32qu) in1.v256 > (__v32qu) in2.v256);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return in1 > in2 compressed
	[[nodiscard]] constexpr static inline uint32_t gt(const uint8x32_t in1,
	                                                  const uint8x32_t in2) noexcept {
		const __m256i tmp = (__m256i) ((__v32qu) in1.v256 > (__v32qu) in2.v256);
		return __builtin_ia32_pmovmskb256((__v32qi) tmp);
	}

	/// \param in1
	/// \param in2
	/// \return in1 > in2 uncompressed
	[[nodiscard]] constexpr static inline uint8x32_t lt_(const uint8x32_t in1,
	                                                     const uint8x32_t in2) noexcept {
		uint8x32_t ret;
		ret.v256 = (__m256i) ((__v32qu) in1.v256 < (__v32qu) in2.v256);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return in1 > in2 compressed
	[[nodiscard]] constexpr static inline uint32_t lt(const uint8x32_t in1,
	                                                  const uint8x32_t in2) noexcept {
		const __m256i tmp = (__m256i) ((__v32qu) in1.v256 < (__v32qu) in2.v256);
		return __builtin_ia32_pmovmskb256((__v32qi) tmp);
	}

	///
	/// \param in1
	/// \param in2
	/// \return in1 == in2 compressed
	[[nodiscard]] constexpr static inline uint8x32_t cmp_(const uint8x32_t in1,
	                                                      const uint8x32_t in2) noexcept {
		uint8x32_t ret;
		ret.v256 = (__m256i) ((__v32qs) in1.v256 == (__v32qs) in2.v256);
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t cmp(const uint8x32_t in1,
	                                                   const uint8x32_t in2) noexcept {
		const __m256i tmp = (__m256i) ((__v32qi) in1.v256 == (__v32qi) in2.v256);
		return __builtin_ia32_pmovmskb256((__v32qi) tmp);
	}

	///
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t popcnt(const uint8x32_t in) noexcept {
		uint8x32_t ret;
		ret.v256 = popcount_avx2_8(in.v256);
		return ret;
	}

	/// checks if all bytes are equal
	/// source: https://github.com/WojciechMula/toys/tree/master/simd-all-bytes-equal
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline bool all_equal(const uint8x32_t in) noexcept {
#ifdef __clang__
		// no cost, 0th lane is mapped to an XMM reg
		const __m128i lane0 = __builtin_shufflevector((__v4di) in.v256, (__v4di) in.v256, 0, 1);
		const __m128i tmp = (__m128i) __builtin_ia32_pshufb128((__v16qi) lane0,
		                                                       (__v16qi) __extension__(__m128i)(__v4si){0, 0, 0, 0});
		const __m256i populated_0th_byte = (__m256i) __builtin_shufflevector((__v2di) tmp, (__v2di) tmp, 0, 1, 2, 3);
		const __m256i eq = (__m256i) ((__v32qi) in.v256 == (__v32qi) populated_0th_byte);
		return (uint32_t) __builtin_ia32_pmovmskb256((__v32qi) eq) == 0xffffffff;
#else
		const __m128i lane0 = (__m128i) __builtin_ia32_si_si256((__v8si) in.v256);
		const __m128i tmp = (__m128i) __builtin_ia32_pshufb128((__v16qi) lane0,
		                                                       (__v16qi) __extension__(__m128i)(__v4si){0, 0, 0, 0});
		const __m256i populated_0th_byte = ((__m256i) __builtin_ia32_vinsertf128_si256(
		        (__v8si) (__m256i) (__builtin_ia32_si256_si((__v4si) tmp)),
		        (__v4si) (__m128i) (tmp),
		        (int) (1)));
		const __m256i eq = (__m256i) ((__v32qi) in.v256 == (__v32qi) populated_0th_byte);
		return (uint32_t) __builtin_ia32_pmovmskb256((__v32qi) eq) == 0xffffffff;
#endif
	}

	/// only reverses the u8 limbs
	/// source:  https://github.com/WojciechMula/toys/blob/master/simd-basic/reverse-bytes/reverse.avx2.cpp
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t reverse(const uint8x32_t in) noexcept {
		// extract 128-bit lanes
		const __m128i lo = ((__m128i) __builtin_ia32_extract128i256((__v4di) (__m256i) (in.v256), (int) (0)));
		const __m128i hi = ((__m128i) __builtin_ia32_extract128i256((__v4di) (__m256i) (in.v256), (int) (1)));

		// reverse them using SSE instructions
		const __m128i indices = __extension__(__m128i)(__v16qi){15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
		const __m128i lo_rev = (__m128i) __builtin_ia32_pshufb128((__v16qi) lo, (__v16qi) indices);
		const __m128i hi_rev = (__m128i) __builtin_ia32_pshufb128((__v16qi) hi, (__v16qi) indices);

		// build the new AVX2 vector
#ifdef __clang__
		__m256i ret = __builtin_shufflevector((__v2di) hi_rev, (__v2di) hi_rev, 0, 1, 2, 3);
#else
		__m256i ret = (__m256i) __builtin_ia32_si256_si((__v4si) hi_rev);
#endif
		ret = ((__m256i) __builtin_ia32_insert128i256((__v4di) (__m256i) (ret),
		                                              (__v2di) (__m128i) (lo_rev), (int) (1)));
		uint8x32_t ret2;
		ret2.v256 = ret;
		return ret2;
	}

	/// TODO not optimized
	/// kmoves the msb into each bit
	[[nodiscard]] constexpr static inline uint32_t move(const uint8x32_t in) noexcept {
		return __builtin_ia32_pmovmskb256((__v32qi) in.v256);
	}
};

struct uint16x16_t {
	constexpr static uint32_t LIMBS = 16;
	using limb_type = uint16_t;
	using S = uint16x16_t;

	union {
		// compatibility with TxN_t
		uint16_t d[16];

		uint8_t v8[32];
		uint16_t v16[16];
		uint32_t v32[8];
		uint64_t v64[4];
		__m256i v256;
	};

	constexpr inline uint16x16_t() noexcept = default;
	[[nodiscard]] constexpr inline static bool is_unsigned() { return true; }


	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const {
		ASSERT(i < LIMBS);
		return d[i];
	}

	[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) {
		ASSERT(i < LIMBS);
		return d[i];
	}

	///
	/// \return
	[[nodiscard]] static inline uint16x16_t random() noexcept {
		uint16x16_t ret;
		for (size_t i = 0; i < 4; ++i) {
			ret.v64[i] = fastrandombytes_uint64();
		}
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	///
	[[nodiscard]] constexpr static inline uint16x16_t set(short __q15, short __q14, short __q13, short __q12,
	                                                      short __q11, short __q10, short __q09, short __q08,
	                                                      short __q07, short __q06, short __q05, short __q04,
	                                                      short __q03, short __q02, short __q01, short __q00) noexcept {
		uint16x16_t out;
		out.v256 = __extension__(__m256i)(__v16hi){
		        __q00, __q01, __q02, __q03, __q04, __q05, __q06, __q07,
		        __q08, __q09, __q10, __q11, __q12, __q13, __q14, __q15};
		return out;
	}

	///
	[[nodiscard]] constexpr static inline uint16x16_t setr(short __q15, short __q14, short __q13, short __q12,
	                                                       short __q11, short __q10, short __q09, short __q08,
	                                                       short __q07, short __q06, short __q05, short __q04,
	                                                       short __q03, short __q02, short __q01, short __q00) noexcept {
		return uint16x16_t::set(__q00, __q01, __q02, __q03, __q04, __q05, __q06,
		                        __q07, __q08, __q09, __q10, __q11, __q12, __q13, __q14, __q15);
	}

	///
	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t set1(const uint16_t a) noexcept {
		return uint16x16_t::set(a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a);
	}

	///
	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t set(const uint16_t a) noexcept {
		return uint16x16_t::set1(a);
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline uint16x16_t load(const uint16_t *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t aligned_load(const uint16_t ptr[16]) noexcept {
		if (std::is_constant_evaluated()) {
			// in the constexpr case simply ignore that the data is aligned
			// it will not have any "runtime" penalties
			const __m256i tmp = u16tom256(ptr);
			S out;
			out.v256 = tmp;
			return out;
		} else {
			auto *ptr256 = (__m256i *) ptr;
			S out;
			out.v256 = *ptr256;
			return out;
		}
	}

	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t unaligned_load(const uint16_t *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			const __m256i tmp = u16tom256(ptr);
			S out;
			out.v256 = tmp;
			return out;
		} else {
			S out;
			out.v256 = internal::unaligned_load_wrapper((__m256i_u *) ptr);
			return out;
		}
	}

	/// NOTE: can never be constexpr
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	static inline void store(void *ptr, const uint16x16_t in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	static inline void aligned_store(void *ptr, const uint16x16_t in) noexcept {
		auto *ptr256 = (__m256i *) ptr;
		*ptr256 = in.v256;
	}

	///
	/// \param ptr
	/// \param in
	static inline void unaligned_store(void *ptr, const uint16x16_t in) noexcept {
		auto *ptr256 = (__m256i_u *) ptr;
		internal::unaligned_store_wrapper(ptr256, in.v256);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t xor_(const uint16x16_t in1,
	                                                       const uint16x16_t in2) noexcept {
		uint16x16_t out;
		out.v256 = in1.v256 ^ in2.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t and_(const uint16x16_t in1,
	                                                       const uint16x16_t in2) noexcept {
		uint16x16_t out;
		out.v256 = in1.v256 & in2.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t or_(const uint16x16_t in1,
	                                                      const uint16x16_t in2) noexcept {
		uint16x16_t out;
		out.v256 = in1.v256 | in2.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t andnot(const uint16x16_t in1,
	                                                         const uint16x16_t in2) noexcept {
		uint16x16_t out;
		out.v256 = ~(in1.v256 & in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t not_(const uint16x16_t in1) noexcept {
		uint16x16_t out;
		const uint16x16_t minus_one = uint16x16_t::set1(-1);
		out.v256 = in1.v256 ^ minus_one.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t add(const uint16x16_t in1,
	                                                      const uint16x16_t in2) noexcept {
		uint16x16_t out;
		out.v256 = (__m256i) ((__v16hu) in1.v256 + (__v16hu) in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t sub(const uint16x16_t in1,
	                                                      const uint16x16_t in2) noexcept {
		uint16x16_t out;
		out.v256 = (__m256i) ((__v16hu) in1.v256 - (__v16hu) in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t mullo(const uint16x16_t in1,
	                                                        const uint16x16_t in2) noexcept {
		uint16x16_t out;
		out.v256 = (__m256i) ((__v16hu) in1.v256 * (__v16hu) in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t mullo(const uint16x16_t in1,
	                                                        const uint8_t in2) noexcept {
		auto rs = uint16x16_t::set1(in2);
		return mullo(in1, rs);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t slli(const uint16x16_t in1,
	                                                       const uint8_t in2) noexcept {
		ASSERT(in2 <= 16);
		uint16x16_t out;
		uint16x16_t mask = set1((1u << ((16u - in2) & 15u)) - 1u);
		out = uint16x16_t::and_(in1, mask);
#ifndef __clang__
		out.v256 = (__m256i) __builtin_ia32_psllwi256((__v16hi) out.v256, in2);
#else
		out.v256 = (__m256i) (((__v16hu) out.v256) << in2);
#endif
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t srli(const uint16x16_t in1,
	                                                       const uint8_t in2) noexcept {
		ASSERT(in2 <= 16);
		const uint16x16_t mask = set1(~((1u << in2) - 1u));
		uint16x16_t out;
		out = uint16x16_t::and_(in1, mask);
#ifndef __clang__
		out.v256 = (__m256i) __builtin_ia32_psrlwi256((__v16hi) out.v256, in2);
#else
		out.v256 = (__m256i) (((__v16hu) out.v256) >> in2);
#endif
		return out;
	}


	/// \param in1
	/// \param in2
	/// \return in1 > in2 uncompressed
	[[nodiscard]] constexpr static inline uint16x16_t gt_(const uint16x16_t in1,
	                                                      const uint16x16_t in2) noexcept {
		uint16x16_t ret;
		ret.v256 = (__m256i) ((__v16hu) in1.v256 > (__v16hu) in2.v256);
		return ret;
	}

	/// NOTE: this is a function which cannot be vectorized
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t gt(const uint16x16_t in1,
	                                                  const uint16x16_t in2) noexcept {
		uint16x16_t tmp;
		tmp.v256 = (__m256i) ((__v16hu) in1.v256 > (__v16hu) in2.v256);
		return S::move(tmp);
	}

	/// \param in1
	/// \param in2
	/// \return in1 > in2 uncompressed
	[[nodiscard]] constexpr static inline uint16x16_t lt_(const uint16x16_t in1,
	                                                      const uint16x16_t in2) noexcept {
		uint16x16_t ret;
		ret.v256 = (__m256i) ((__v16hu) in1.v256 < (__v16hu) in2.v256);
		return ret;
	}

	/// NOTE: this is a function which cannot be vectorized
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t lt(const uint16x16_t in1,
	                                                  const uint16x16_t in2) noexcept {
		uint16x16_t tmp;
		tmp.v256 = (__m256i) ((__v16hu) in1.v256 < (__v16hu) in2.v256);
		return S::move(tmp);
	}

	/// \param in1
	/// \param in2
	/// \return in1 == in2 uncompressed
	[[nodiscard]] constexpr static inline uint16x16_t cmp_(const uint16x16_t in1,
	                                                       const uint16x16_t in2) noexcept {
		uint16x16_t ret;
		ret.v256 = (__m256i) ((__v16hi) in1.v256 == (__v16hi) in2.v256);
		return ret;
	}

	/// NOTE: this is a function which cannot be vectorized
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline int cmp(const uint16x16_t in1,
	                                              const uint16x16_t in2) noexcept {
		uint16x16_t tmp;
		tmp.v256 = (__m256i) ((__v16hi) in1.v256 == (__v16hi) in2.v256);

		int ret = 0;
		for (uint16_t i = 0; i < 16; i++) {
			ret ^= (tmp.v16[i] != 0) << i;
		}

		return ret;
	}

	///
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t popcnt(const uint16x16_t in) noexcept {
		uint16x16_t ret;
		ret.v256 = popcount_avx2_16(in.v256);
		return ret;
	}

	/// TODO not optimized
	/// checks if all bytes are equal
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline bool all_equal(const uint16x16_t in) noexcept {
		for (uint32_t i = 1; i < LIMBS; i++) {
			if (in.d[i - 1] != in.d[i]) {
				return false;
			}
		}

		return true;
	}

	// TODO not optimized
	[[nodiscard]] constexpr static inline uint16x16_t reverse(const uint16x16_t in) noexcept {
		S ret;
		for (uint32_t i = 0; i < LIMBS; i++) {
			ret.d[LIMBS - 1 - i] = in.d[i];
		}

		return ret;
	}


	/// TODO not optimized
	/// kmoves the msb into each bit
	[[nodiscard]] constexpr static inline uint16_t move(const uint16x16_t in) noexcept {
		uint16_t ret = 0;
		constexpr limb_type mask = 1u << ((sizeof(limb_type) * 8) - 1u);

		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret ^= ((in.d[i] & mask) > 0) << i;
		}
		return ret;
	}
};

struct uint32x8_t {
	constexpr static uint32_t LIMBS = 8;
	using limb_type = uint32_t;
	using S = uint32x8_t;

	union {
		// compatibility to TxN_t
		uint32_t d[8];

		uint8_t v8[32];
		uint16_t v16[16];
		uint32_t v32[8];
		uint64_t v64[4];
		cryptanalysislib::_uint32x4_t v128[2];
		__m256i v256;
	};

	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
		ASSERT(i < LIMBS);
		return d[i];
	}

	[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
		ASSERT(i < LIMBS);
		return d[i];
	}

	///
	/// \return
	[[nodiscard]] static inline uint32x8_t random() noexcept {
		uint32x8_t ret{};
		for (size_t i = 0; i < 4; ++i) {
			ret.v64[i] = fastrandombytes_uint64();
		}
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	///
	/// \param a0
	/// \param a1
	/// \param a2
	/// \param a3
	/// \param a4
	/// \param a5
	/// \param a6
	/// \param a7
	/// \return
	[[nodiscard]] constexpr inline static uint32x8_t set(const uint32_t a0,
	                                                     const uint32_t a1,
	                                                     const uint32_t a2,
	                                                     const uint32_t a3,
	                                                     const uint32_t a4,
	                                                     const uint32_t a5,
	                                                     const uint32_t a6,
	                                                     const uint32_t a7) noexcept {
		uint32x8_t out{};
		out.v256 = __extension__(__m256i)(__v8si){(int) a7, (int) a6, (int) a5, (int) a4, (int) a3, (int) a2, (int) a1, (int) a0};
		return out;
	}

	///
	/// \param a0
	/// \param a1
	/// \param a2
	/// \param a3
	/// \param a4
	/// \param a5
	/// \param a6
	/// \param a7
	/// \return
	[[nodiscard]] constexpr inline static uint32x8_t setr(const uint32_t a0,
	                                                      const uint32_t a1,
	                                                      const uint32_t a2,
	                                                      const uint32_t a3,
	                                                      const uint32_t a4,
	                                                      const uint32_t a5,
	                                                      const uint32_t a6,
	                                                      const uint32_t a7) noexcept {
		return set(a7, a6, a5, a4, a3, a2, a1, a0);
	}

	///
	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t set1(const uint32_t a) noexcept {
		return set(a, a, a, a, a, a, a, a);
	}

	///
	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t set(const uint32_t a) noexcept {
		return uint32x8_t::set1(a);
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline uint32x8_t load(const uint32_t *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t aligned_load(const uint32_t *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			// in the constexpr case simply ignore that the data is aligned
			// it will not have any "runtime" penalties
			const __m256i tmp = u32tom256(ptr);
			S out;
			out.v256 = tmp;
			return out;
		} else {
			auto *ptr256 = (__m256i *) ptr;
			S out;
			out.v256 = *ptr256;
			return out;
		}
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t unaligned_load(const uint32_t *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			const __m256i tmp = u32tom256(ptr);
			S out;
			out.v256 = tmp;
			return out;
		} else {
			S out;
			out.v256 = internal::unaligned_load_wrapper((__m256i_u *) ptr);
			return out;
		}
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(void *ptr, const uint32x8_t in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		unaligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(void *ptr, const uint32x8_t in) noexcept {
		auto *ptr256 = (__m256i *) ptr;
		*ptr256 = in.v256;
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const uint32x8_t in) noexcept {
		auto *ptr256 = (__m256i_u *) ptr;
		internal::unaligned_store_wrapper(ptr256, in.v256);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t xor_(const uint32x8_t in1,
	                                                      const uint32x8_t in2) noexcept {
		uint32x8_t out{};
		out.v256 = in1.v256 ^ in2.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t and_(const uint32x8_t in1,
	                                                      const uint32x8_t in2) noexcept {
		uint32x8_t out{};
		out.v256 = in1.v256 & in2.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t or_(const uint32x8_t in1,
	                                                     const uint32x8_t in2) noexcept {
		uint32x8_t out{};
		out.v256 = in1.v256 | in2.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t andnot(const uint32x8_t in1,
	                                                        const uint32x8_t in2) noexcept {
		uint32x8_t out{};
		out.v256 = (__m256i) (~(__v4du) in1.v256 & (__v4du) in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t not_(const uint32x8_t in1) noexcept {
		uint32x8_t out{};
		constexpr uint32x8_t minus_one = uint32x8_t::set1(-1);
		out.v256 = in1.v256 ^ minus_one.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t add(const uint32x8_t in1,
	                                                     const uint32x8_t in2) noexcept {
		uint32x8_t out{};
		out.v256 = (__m256i) ((__v8su) in1.v256 + (__v8su) in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t sub(const uint32x8_t in1,
	                                                     const uint32x8_t in2) noexcept {
		uint32x8_t out{};
		out.v256 = (__m256i) ((__v8su) in1.v256 - (__v8su) in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t mullo(const uint32x8_t in1,
	                                                       const uint32x8_t in2) noexcept {
		uint32x8_t out{};
		out.v256 = (__m256i) ((__v8su) in1.v256 * (__v8su) in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t mullo(const uint32x8_t in1,
	                                                       const uint32_t in2) noexcept {
		auto m = uint32x8_t::set1(in2);
		return mullo(in1, m);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t slli(const uint32x8_t in1,
	                                                      const uint8_t in2) noexcept {
		ASSERT(in2 <= 8);
		uint32x8_t out{};
#ifndef __clang__
		out.v256 = (__m256i) __builtin_ia32_psllwi256((__v16hi) in1.v256, in2);
#else
		out.v256 = (__m256i) ((__v8si) in1.v256 << in2);
#endif
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t srli(const uint32x8_t in1,
	                                                      const uint8_t in2) noexcept {
		ASSERT(in2 <= 8);
		uint32x8_t out{};
#ifndef __clang__
		out.v256 = (__m256i) __builtin_ia32_psrldi256((__v8si) in1.v256, in2);
#else
		out.v256 = (__m256i) ((__v8si) in1.v256 >> in2);
#endif
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return in1 > in2 uncompress
	[[nodiscard]] constexpr static inline uint32x8_t gt_(const uint32x8_t in1,
	                                                     const uint32x8_t in2) noexcept {
		uint32x8_t ret{};
		ret.v256 = (__m256i) ((__v8su) in1.v256 > (__v8su) in2.v256);
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t gt(const uint32x8_t in1, const uint32x8_t in2) noexcept {
		const __m256i tmp = (__m256i) ((__v8su) in1.v256 > (__v8su) in2.v256);
		return __builtin_ia32_movmskps256((__v8sf) tmp);
	}

	///
	/// \param in1
	/// \param in2
	/// \return in1 > in2 uncompress
	[[nodiscard]] constexpr static inline uint32x8_t lt_(const uint32x8_t in1,
	                                                     const uint32x8_t in2) noexcept {
		uint32x8_t ret{};
		ret.v256 = (__m256i) ((__v8su) in1.v256 < (__v8su) in2.v256);
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t lt(const uint32x8_t in1,
	                                                  const uint32x8_t in2) noexcept {
		const __m256i tmp = (__m256i) ((__v8su) in1.v256 < (__v8su) in2.v256);
		return __builtin_ia32_movmskps256((__v8sf) tmp);
	}

	///
	/// \param in1
	/// \param in2
	/// \return in1 == in2 uncompress
	[[nodiscard]] constexpr static inline uint32x8_t cmp_(const uint32x8_t in1,
	                                                      const uint32x8_t in2) noexcept {
		uint32x8_t ret{};
		ret.v256 = (__m256i) ((__v8si) in1.v256 == (__v8si) in2.v256);
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline int cmp(const uint32x8_t in1, const uint32x8_t in2) noexcept {
		const __m256i tmp = (__m256i) ((__v8si) in1.v256 == (__v8si) in2.v256);
#ifndef __clang__
		return __builtin_ia32_movmskps256((__v8sf) tmp);
#else
		return _mm256_movemask_ps((__m256) tmp);
#endif
	}

	///
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t popcnt(const uint32x8_t in) noexcept {
		uint32x8_t ret{};

#ifdef USE_AVX512
#ifndef __clang__
		ret.v256 = (__m256i) __builtin_ia32_vpopcountd_v8si((__v8si) in.v256);
#else
		ret.v256 = __builtin_ia32_vpopcntd_256((__v8si) in.v256);
#endif
#else
		ret.v256 = popcount_avx2_32(in.v256);
#endif
		return ret;
	}

	/// TODO not optimized
	/// checks if all bytes are equal
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline bool all_equal(const uint32x8_t in) noexcept {
		for (uint32_t i = 1; i < LIMBS; i++) {
			if (in.d[i - 1] != in.d[i]) {
				return false;
			}
		}

		return true;
	}

	// TODO not optimized
	[[nodiscard]] constexpr static inline uint32x8_t reverse(const uint32x8_t in) noexcept {
		S ret;
		for (uint32_t i = 0; i < LIMBS; i++) {
			ret.d[LIMBS - 1 - i] = in.d[i];
		}

		return ret;
	}
	/// \tparam scale
	/// \param ptr
	/// \param data
	/// \return
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline uint32x8_t gather(const void *ptr,
	                                                        const uint32x8_t data) noexcept {
		uint32x8_t ret{};

#ifndef __clang__
		ret.v256 = (__m256i) __builtin_ia32_gathersiv8si((__v8si) _mm256_setzero_si256(),
		                                                 (int const *) (ptr),
		                                                 (__v8si) (__m256i) (data.v256),
		                                                 (__v8si) _mm256_set1_epi32(-1),
		                                                 (int) (scale));
#else
		ret.v256 = _mm256_i32gather_epi32((int *) ptr, data.v256, scale);
#endif
		return ret;
	}

	///
	/// \tparam scale
	/// \param ptr
	/// \param offset
	/// \param data
	/// \return
	template<const uint32_t scale = 1>
	constexpr static inline void scatter(const void *ptr,
	                                     const uint32x8_t offset,
	                                     const uint32x8_t data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		const uint8_t *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < 8; i++) {
			*(uint32_t *) (ptr8 + offset.v32[i] * scale) = data.v32[i];
		}
	}

	///
	/// \param in
	/// \param perm
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t permute(const uint32x8_t in,
	                                                         const uint32x8_t perm) noexcept {
		uint32x8_t ret{};
		ret.v256 = (__m256i) __builtin_ia32_permvarsi256((__v8si) in.v256, (__v8si) perm.v256);
		return ret;
	}

	/// moves the msb into each bit
	[[nodiscard]] constexpr static inline uint8_t move(const uint32x8_t in) noexcept {
		return __builtin_ia32_movmskps256((__v8sf) in.v256);
	}

	/// needs BMI2
	/// src: https://stackoverflow.com/questions/36932240/avx2-what-is-the-most-efficient-way-to-pack-left-based-on-a-mask
	/// input:
	/// 	mask: 0b010101010
	/// output: a permutation mask s.t, applied on in =  [ x0, x1, x2, x3, x4, x5, x6, x7 ],
	/// 			uint32x8_t::permute(in, permutation_mask) will result int
	///  	[x1, x3, x5, x7, 0, 0, 0, 0]
	[[nodiscard]] constexpr static inline uint32x8_t pack(const uint32_t mask) noexcept {
		uint32x8_t ret{};
#ifdef USE_BMI2
		uint64_t expanded_mask = __builtin_ia32_pdep_di(mask, 0x0101010101010101);
		expanded_mask *= 0xFFU;
		const uint64_t identity_indices = 0x0706050403020100;
		uint64_t wanted_indices = __builtin_ia32_pext_di(identity_indices, expanded_mask);
		const __m128i bytevec = __extension__(__m128i)(__v2di){0, (long long int) wanted_indices};
#ifdef __clang__
		ret.v256 = (__m256i) __builtin_convertvector((__v8hi) bytevec, __v8si);
#else
		ret.v256 = (__m256i) __builtin_ia32_pmovzxbd256((__v16qi) bytevec);
#endif
#else
		ASSERT(false);
#endif
		return ret;
	}


	///
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t cvtepu8(const cryptanalysislib::_uint8x16_t in) noexcept {
		uint32x8_t ret{};
#ifdef __clang__
		ret.v256 = (__m256i) __builtin_convertvector((__v8hi) in.v128, __v8si);
#else
		ret.v256 = (__m256i) __builtin_ia32_pmovzxbd256((__v16qi) in.v128);
#endif
		return ret;
	}
};

struct uint64x4_t {
	constexpr static uint32_t LIMBS = 4;
	using limb_type = uint64_t;
	using S = uint64x4_t;

	union {
		// compatibility with TxN_t
		uint64_t d[4];

		uint8_t v8[32];
		uint16_t v16[16];
		uint32_t v32[8];
		uint64_t v64[4];
		__m256i v256;
	};

	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
		ASSERT(i < LIMBS);
		return d[i];
	}

	[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
		ASSERT(i < LIMBS);
		return d[i];
	}

	///
	/// \return
	[[nodiscard]] static inline uint64x4_t random() noexcept {
		uint64x4_t ret;
		for (size_t i = 0; i < 4; ++i) {
			ret.v64[i] = fastrandombytes_uint64();
		}
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	///
	/// \param a
	/// \param b
	/// \param c
	/// \param d
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t set(const uint64_t a,
	                                                     const uint64_t b,
	                                                     const uint64_t c,
	                                                     const uint64_t d) noexcept {
		uint64x4_t out;
		out.v256 = __extension__(__m256i)(__v4di){(long long) d,
		                                          (long long) c,
		                                          (long long) b,
		                                          (long long) a};
		return out;
	}

	///
	/// \param a
	/// \param b
	/// \param c
	/// \param d
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t setr(const uint64_t a, const uint64_t b, const uint64_t c, const uint64_t d) noexcept {
		return set(d, c, b, a);
	}

	///
	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t set1(const uint64_t a) noexcept {
		return set(a, a, a, a);
	}

	///
	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t set(const uint64_t a) noexcept {
		return uint64x4_t::set1(a);
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline uint64x4_t load(const uint64_t *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t aligned_load(const uint64_t *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			// in the constexpr case simply ignore that the data is aligned
			// it will not have any "runtime" penalties
			const __m256i tmp = u64tom256(ptr);
			S out;
			out.v256 = tmp;
			return out;
		} else {
			auto *ptr256 = (__m256i *) ptr;
			S out;
			out.v256 = *ptr256;
			return out;
		}
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t unaligned_load(const uint64_t *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			const __m256i tmp = u64tom256(ptr);
			S out;
			out.v256 = tmp;
			return out;
		} else {
			S out;
			out.v256 = internal::unaligned_load_wrapper((const __m256i_u *) ptr);
			return out;
		}
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	static inline void store(void *ptr, const uint64x4_t in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		unaligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	static inline void aligned_store(void *ptr, const uint64x4_t in) noexcept {
		auto *ptr256 = (__m256i *) ptr;
		*ptr256 = in.v256;
	}

	///
	/// \param ptr
	/// \param in
	static inline void unaligned_store(void *ptr, const uint64x4_t in) noexcept {
		auto *ptr256 = (__m256i_u *) ptr;
		internal::unaligned_store_wrapper(ptr256, in.v256);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t xor_(const uint64x4_t in1,
	                                                      const uint64x4_t in2) noexcept {
		uint64x4_t out;
		out.v256 = in1.v256 ^ in2.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t and_(const uint64x4_t in1,
	                                                      const uint64x4_t in2) noexcept {
		uint64x4_t out;
		out.v256 = in1.v256 & in2.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t or_(const uint64x4_t in1,
	                                                     const uint64x4_t in2) noexcept {
		uint64x4_t out;
		out.v256 = in1.v256 | in2.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t andnot(const uint64x4_t in1,
	                                                        const uint64x4_t in2) noexcept {
		uint64x4_t out;
		out.v256 = (__m256i) (~(__v4du) in1.v256 & (__v4du) in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t not_(const uint64x4_t in1) noexcept {
		uint64x4_t out;
		constexpr uint64x4_t minus_one = uint64x4_t::set1(-1);
		out.v256 = in1.v256 ^ minus_one.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t add(const uint64x4_t in1,
	                                                     const uint64x4_t in2) noexcept {
		uint64x4_t out;
		out.v256 = (__m256i) ((__v4du) in1.v256 + (__v4du) in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t sub(const uint64x4_t in1,
	                                                     const uint64x4_t in2) noexcept {
		uint64x4_t out;
		out.v256 = (__m256i) ((__v4du) in1.v256 - (__v4du) in2.v256);
		return out;
	}

	/// TODO
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t mullo(const uint64x4_t in1,
	                                                       const uint64x4_t in2) noexcept {
		uint64x4_t out;
#ifdef USE_AVX512
		out.v256 = (__m256i) ((__v4du) in1.v256 * (__v4du) in2.v256);
#else

		if (std::is_constant_evaluated()) {
			// TODO
		} else {
			for (uint32_t i = 0; i < 4; i++) {
				out.v64[i] = in1.v64[i] * in2.v64[i];
			}
		}
#endif
		return out;
	}
	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t mullo(const uint64x4_t in1,
	                                                       const uint64_t in2) noexcept {
		auto m = uint64x4_t::set1(in2);
		return mullo(in1, m);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t slli(const uint64x4_t in1,
	                                                      const uint8_t in2) noexcept {
		ASSERT(in2 <= 8);
		uint64x4_t out;
#ifndef __clang__
		out.v256 = (__m256i) __builtin_ia32_psllqi256((__v4di) in1.v256, in2);
#else
		out.v256 = (__m256i) ((__v4di) in1.v256 << in2);
#endif
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t srli(const uint64x4_t in1,
	                                                      const uint8_t in2) noexcept {
		ASSERT(in2 <= 8);
		//const uint64x4_t mask = set1(((1u << (8u - in2)) - 1u) << in2);
		uint64x4_t out;
		//out = uint64x4_t::and_(in1, mask);
#ifndef __clang__
		out.v256 = (__m256i) __builtin_ia32_psrlqi256((__v4di) in1.v256, in2);
#else
		out.v256 = (__m256i) ((__v4di) in1.v256 >> in2);
#endif
		return out;
	}

	///
	/// \tparam in2
	/// \param in1
	/// \return
	template<const uint32_t in2>
	[[nodiscard]] constexpr static inline uint64x4_t permute(const uint64x4_t in1) noexcept {
		uint64x4_t ret;
		ret.v256 = ((__m256i) __builtin_ia32_permdi256((__v4di) (__m256i) (in1.v256), (int) (in2)));
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return in1 > in2 uncompressed
	[[nodiscard]] constexpr static inline uint64x4_t gt_(const uint64x4_t in1,
	                                                     const uint64x4_t in2) noexcept {
		uint64x4_t ret;
		ret.v256 = (__m256i) ((__v4du) in1.v256 > (__v4du) in2.v256);
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t gt(const uint64x4_t in1,
	                                                  const uint64x4_t in2) noexcept {
		const auto tmp = (__m256i) ((__v4du) in1.v256 > (__v4du) in2.v256);
		return __builtin_ia32_movmskpd256((__v4df) tmp);
	}

	///
	/// \param in1
	/// \param in2
	/// \return in1 > in2 uncompressed
	[[nodiscard]] constexpr static inline uint64x4_t lt_(const uint64x4_t in1,
	                                                     const uint64x4_t in2) noexcept {
		uint64x4_t ret;
		ret.v256 = (__m256i) ((__v4du) in1.v256 < (__v4du) in2.v256);
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t lt(const uint64x4_t in1,
	                                                  const uint64x4_t in2) noexcept {
		const auto tmp = (__m256i) ((__v4du) in1.v256 < (__v4du) in2.v256);
		return __builtin_ia32_movmskpd256((__v4df) tmp);
	}

	///
	/// \param in1
	/// \param in2
	/// \return in1 == in2 uncompressed
	[[nodiscard]] constexpr static inline uint64x4_t cmp_(const uint64x4_t in1,
	                                                      const uint64x4_t in2) noexcept {
		uint64x4_t ret;
		ret.v256 = (__m256i) ((__v4di) in1.v256 == (__v4di) in2.v256);
		return ret;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint32_t cmp(const uint64x4_t in1,
	                                                   const uint64x4_t in2) noexcept {
#ifndef __clang__
		const __m256i tmp = (__m256i) ((__v4di) in1.v256 == (__v4di) in2.v256);
		return __builtin_ia32_movmskpd256((__v4df) tmp);
#else
		const __m256i tmp = _mm256_cmpeq_epi64(in1.v256, in2.v256);
		return _mm256_movemask_pd((__m256d) tmp);
#endif
	}

	///
	/// \param in
	/// \return
	constexpr static inline uint64x4_t popcnt(const uint64x4_t in) noexcept {
		uint64x4_t ret;
#ifdef USE_AVX512
#ifdef __clang__
		ret.v256 = (__m256i) __builtin_ia32_vpopcntq_256((__v4di) in.v256);
#else
		ret.v256 = (__m256i) __builtin_ia32_vpopcountq_v4di((__v4di) in.v256);
#endif
#else
		ret.v256 = popcount_avx2_64(in.v256);
#endif
		return ret;
	}

	/// TODO not optimized
	/// checks if all bytes are equal
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline bool all_equal(const uint64x4_t in) noexcept {
		for (uint32_t i = 1; i < LIMBS; i++) {
			if (in.d[i - 1] != in.d[i]) {
				return false;
			}
		}

		return true;
	}

	// TODO not optimized
	[[nodiscard]] constexpr static inline uint64x4_t reverse(const uint64x4_t in) noexcept {
		S ret;
		for (uint32_t i = 0; i < LIMBS; i++) {
			ret.d[LIMBS - 1 - i] = in.d[i];
		}

		return ret;
	}
	/// \tparam scale
	/// \param ptr
	/// \param data
	/// \return
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline uint64x4_t gather(const void *ptr, const cryptanalysislib::_uint32x4_t data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		uint64x4_t ret;
#ifndef __clang__
		ret.v256 = (__m256i) __builtin_ia32_gathersiv4di((__v4di) _mm256_setzero_si256(),
		                                                 (long long const *) (ptr),
		                                                 (__v4si) (__m128i) (data.v128),
		                                                 (__v4di) _mm256_set1_epi64x(-1),
		                                                 (int) (scale));
#else
		ret.v256 = _mm256_i32gather_epi64((long long *) ptr, data.v128, scale);
#endif
		return ret;
	}

	/// \tparam scale
	/// \param ptr
	/// \param data
	/// \return
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline uint64x4_t gather(const void *ptr, const uint64x4_t data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		uint64x4_t ret;
#ifndef __clang__
		ret.v256 = (__m256i) __builtin_ia32_gatherdiv4di((__v4di) _mm256_setzero_si256(),
		                                                 (long long const *) (ptr),
		                                                 (__v4di) (__m256i) (data.v256),
		                                                 (__v4di) _mm256_set1_epi64x(-1),
		                                                 (int) (scale));
#else
		ret.v256 = _mm256_i64gather_epi64((long long *) ptr, data.v256, scale);
#endif
		return ret;
	}

	///
	/// \tparam scale
	/// \param ptr
	/// \param offset
	/// \param data
	/// \return
	template<const uint32_t scale = 1>
	constexpr static inline void scatter(const void *ptr, const uint64x4_t offset, const uint64x4_t data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		const uint8_t *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			*(uint64_t *) (ptr8 + offset.v64[i] * scale) = data.v64[i];
		}
	}

	[[nodiscard]] constexpr static inline uint8_t move(const uint64x4_t in1) noexcept {
#ifndef __clang__
		return __builtin_ia32_movmskpd256((__v4df) in1.v256);
#else
		return _mm256_movemask_pd((__m256d) in1.v256);
#endif
	}
};


inline void sse_prefixsum_u32(uint32_t *in) noexcept {
	__m128i x = _mm_loadu_si128((__m128i *) in);
	// x = 1, 2, 3, 4
	x = _mm_add_epi32(x, _mm_slli_si128(x, 4));
	// x = 1, 2, 3, 4
	//   + 0, 1, 2, 3
	//   = 1, 3, 5, 7
	x = _mm_add_epi32(x, _mm_slli_si128(x, 8));
	// x = 1, 3, 5, 7
	//   + 0, 0, 1, 3
	//   = 1, 3, 6, 10
	_mm_storeu_si128((__m128i *) in, x);
	// return x;
}

inline void avx_prefix_prefixsum_u32(uint32_t *p) noexcept {
	__m256i x = _mm256_loadu_si256((__m256i *) p);
	x = _mm256_add_epi32(x, _mm256_slli_si256(x, 4));
	x = _mm256_add_epi32(x, _mm256_slli_si256(x, 8));
	_mm256_storeu_si256((__m256i *) p, x);
}

inline __m128i sse_prefixsum_accumulate_u32(uint32_t *p, const __m128i s) {
	__m128i d = (__m128i) _mm_broadcast_ss((float *) &p[3]);
	__m128i x = _mm_loadu_si128((__m128i *) p);
	x = _mm_add_epi32(s, x);
	_mm_storeu_si128((__m128i *) p, x);
	return _mm_add_epi32(s, d);
}

// TODO use L1 cache size
constexpr size_t prefixsum_block_size = 64;
// TODO move to `simd.h` as this is a generic algorithm, which only needs
// specialized sub routines.
//
// PrefixSum:
// 	a[0] = a[0]
//	a[1] = a[0] + a[1]
//		...
__m128i avx2_local_prefixsum_u32(uint32_t *a, __m128i s) {
	for (uint32_t i = 0; i < prefixsum_block_size; i += 8) {
		avx_prefix_prefixsum_u32(&a[i]);
	}

	for (uint32_t i = 0; i < prefixsum_block_size; i += 4) {
		s = sse_prefixsum_accumulate_u32(&a[i], s);
	}

	return s;
}

void avx2_prefixsum_u32(uint32_t *a, const size_t n) {
	// simple version for small inputs
	if (n < prefixsum_block_size) {
		for (uint32_t i = 1; i < n; i++) {
			a[i] += a[i - 1];
		}
		return;
	}

	__m128i s = _mm_setzero_si128();
	uint32_t i = 0;
	for (; i + prefixsum_block_size <= n; i += prefixsum_block_size) {
		s = avx2_local_prefixsum_u32(a + i, s);
	}

	// tail mngt.
	for (; i < n; i++) {
		a[i] += a[i - 1];
	}
	// slow version
	// for (uint32_t i = 0; i < n; i += 8) {
	//     avx_prefix_prefixsum_u32(&a[i]);
	// }
	//
	// __m128i s = (__m128i) _mm_broadcast_ss((float*) &a[3]);
	// for (uint32_t i = 4; i < n; i += 4) {
	//     s = sse_prefixsum_accumulate_u32(&a[i], s);
	// }
}

/// loads `element_count` f32 elements from array + index*8
/// \param array base pointer to the data
/// \param index number of __m256 already loaded
/// \param element_count  number of f32 to load
/// \return __m256 register with the first `element_count` f32
/// 		fields loaded, the rest is set to inf.
static inline __m256 avx2_load_f32x8(const float *array,
                                     const uint32_t index,
                                     const uint32_t element_count) noexcept {
	if (element_count == 8) {
		return _mm256_loadu_ps(array + index * 8);
	}

	__m256 inf_mask = _mm256_cvtepi32_ps(_mm256_set_epi32(0x7F800000,
	                                                      (element_count > 6) ? 0 : 0x7F800000,
	                                                      (element_count > 5) ? 0 : 0x7F800000,
	                                                      (element_count > 4) ? 0 : 0x7F800000,
	                                                      (element_count > 3) ? 0 : 0x7F800000,
	                                                      (element_count > 2) ? 0 : 0x7F800000,
	                                                      (element_count > 1) ? 0 : 0x7F800000,
	                                                      (element_count > 0) ? 0 : 0x7F800000));

	__m256i loadstoremask = _mm256_set_epi32(0,
	                                         (element_count > 6) ? 0xffffffff : 0,
	                                         (element_count > 5) ? 0xffffffff : 0,
	                                         (element_count > 4) ? 0xffffffff : 0,
	                                         (element_count > 3) ? 0xffffffff : 0,
	                                         (element_count > 2) ? 0xffffffff : 0,
	                                         (element_count > 1) ? 0xffffffff : 0,
	                                         (element_count > 0) ? 0xffffffff : 0);
	__m256 a = _mm256_maskload_ps(array + index * 8, loadstoremask);
	return _mm256_or_ps(a, inf_mask);
}

/// \param array base pointer to the data
/// \param a data to store
/// \param index number of `__m256` already stored
/// \param element_count numbe of f32 to store from `a`
static inline void avx2_store_f32x8(float *array,
                                    const __m256 a,
                                    const uint32_t index,
                                    const uint32_t element_count) {
	if (element_count == 8) {
		_mm256_storeu_ps(array + index * 8, a);
	} else {
		__m256i loadstoremask = _mm256_set_epi32(0,
		                                         (element_count > 6) ? 0xffffffff : 0,
		                                         (element_count > 5) ? 0xffffffff : 0,
		                                         (element_count > 4) ? 0xffffffff : 0,
		                                         (element_count > 3) ? 0xffffffff : 0,
		                                         (element_count > 2) ? 0xffffffff : 0,
		                                         (element_count > 1) ? 0xffffffff : 0,
		                                         (element_count > 0) ? 0xffffffff : 0);
		_mm256_maskstore_ps(array + index * 8, loadstoremask, a);
	}
}

#endif
