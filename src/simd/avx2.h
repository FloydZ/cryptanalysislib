#pragma once

#ifndef USE_AVX2
#error "no avx"
#endif

#include <cstdint>
#include <cstdio>
#include <immintrin.h>

#include "helper.h"
#include "popcount/popcount.h"
#include "random.h"

using namespace cryptanalysislib::popcount::internal;

namespace internal {
	/// helper function. This enforces the compiler to emit a `vmovdqu` instruction
	/// \param ptr pointer to data.
	///				No alignment needed
	/// 			but 32 bytes should be readable
	/// \return unaligned `__m256i`
	constexpr static inline __m256i_u unaligned_load_wrapper(__m256i_u const *ptr) {
		return *ptr;
	}

	/// helper function. This enforces the compiler to emite a unaligned instruction
	/// \param ptr pointer to data
	/// \param data data to store
	/// \return nothing
	constexpr static inline void unaligned_store_wrapper(__m256i_u *ptr, __m256i_u data) {
		*ptr = data;
	}
}// namespace internal


struct uint32x4_t {
	union {
		uint8_t v8[16];
		uint16_t v16[8];
		uint32_t v32[4];
		uint64_t v64[2];
		__m128i v128;
	};

	[[nodiscard]] constexpr static inline uint32x4_t set1(uint32_t a) {
		uint32x4_t ret;
		ret.v32[0] = a;
		ret.v32[1] = a;
		ret.v32[2] = a;
		ret.v32[3] = a;
		return ret;
	}

	[[nodiscard]] constexpr static inline uint32x4_t set(uint32_t a, uint32_t b, uint32_t c, uint64_t d) {
		uint32x4_t ret;
		ret.v32[0] = d;
		ret.v32[1] = c;
		ret.v32[2] = b;
		ret.v32[3] = a;
		return ret;
	}

	[[nodiscard]] constexpr static inline uint32x4_t setr(uint32_t a, uint32_t b, uint32_t c, uint64_t d) {
		uint32x4_t ret;
		ret.v32[0] = a;
		ret.v32[1] = b;
		ret.v32[2] = c;
		ret.v32[3] = d;
		return ret;
	}

	[[nodiscard]] constexpr static inline uint32x4_t set(uint64_t a, uint64_t b) {
		uint32x4_t ret;
		ret.v64[0] = b;
		ret.v64[1] = a;
		return ret;
	}

	[[nodiscard]] constexpr static inline uint32x4_t setr(uint64_t a, uint64_t b) {
		uint32x4_t ret;
		ret.v64[0] = a;
		ret.v64[1] = b;
		return ret;
	}
};

struct uint64x2_t {
	union {
		uint8_t v8[16];
		uint16_t v16[8];
		uint32_t v32[4];
		uint64_t v64[2];
		__m128i v128;
	};

	[[nodiscard]] constexpr static inline uint64x2_t set1(uint64_t a) {
		uint64x2_t ret;
		ret.v64[0] = a;
		ret.v64[1] = a;
		return ret;
	}

	[[nodiscard]] constexpr static inline uint64x2_t set(uint64_t a, uint64_t b) {
		uint64x2_t ret;
		ret.v64[0] = b;
		ret.v64[1] = a;
		return ret;
	}

	[[nodiscard]] constexpr static inline uint64x2_t setr(uint64_t a, uint64_t b) {
		uint64x2_t ret;
		ret.v64[0] = a;
		ret.v64[1] = b;
		return ret;
	}
};


struct uint8x32_t {
	union {
		uint8_t v8[32];
		uint16_t v16[16];
		uint32_t v32[8];
		uint64_t v64[4];
		__m256i v256;
	};

	/// Example of how the constexpr implementation works:
	/// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGEgBykrgAyeAyYAHI%2BAEaYxCCSZqQADqgKhE4MHt6%2BASlpGQKh4VEssfGJtpj2jgJCBEzEBNk%2BflyBdpgOmfWNBMWRMXEJSQoNTS257bbj/WGDZcOJAJS2qF7EyOwc5gDMYcjeWADUJrtuY/iCAHQIZ9gmGgCCewdHmKfneCwsYQTEYVu90eLzM%2BwYhy8JzObmQlycwOeILGxC8DmOXj%2B/lUuzMAH0CKcAOxWJEaACcXgYmWJpJeFMelKxBOOxwAbv5TgBWCy4kxcgAiZzpjMZmMEXAAbCy2VL%2BRY5YLhSDRRTxQRcTK%2BTzjv5%2BULdiKGWq/pLJDKzfLjpJ9cqyeTGXi8SwzFzJXh2a7JXb6Q6iQa6azWSCg8c0AwxphVMliMcxkxHMhjmFaPMMVicfjCaoSHiIOGxunBNjNYSwlxSCqKaGa7W6/WC4T1SWs8mGGZlrSqw6TcXMyz1gQfaryYPrmyvZ8BccIE6XW68J2oE62ZJ0F5luXx5P%2BdgZyu1xuwmZt27lsPjeSAPRXscTt1T47O%2BfSnPEPHpL0QLf3yWkNsnr%2B56Gt2jLEJgBAbAwxyDhefpCmSvo3le3bIdybjJI0rDHMk/yoTe6HgZBxAMN2jZRjGcYNImbapuERYEC2LJUmIeDAOE6B4rQqBMOg%2BYCIWbKoHg6DHAAVLhxCdiYJKgRSTBeEQ4mSTuuzTrOzpeh6YnLJJcFihmpYwYp%2BkUneqnThJ/xeqZ163opp6So%2Bz5elxPHrh%2BeBfhpL7abp/zAUaPbkkRUHGUOIH2jJCFPKGIZBsh%2BFXuhmHENhen2mh/IYVhLBtmRAkEBRsbxjRKZpkJIkYswqbsZgnFjCQmAQJVolWWQYaFQxTFlgw0myVFFJzlpynWQ%2BZzqcNC7if5xC2e1Flto5Ppxf6KprZF9JPOR0axq1xxFWMEDNiyYmDv%2Bjbdad5YXV1J2EmJx63RGTZ/EZfVds8ob8LGx1va2HoTccGjCsm1i4p8bjHKRhpg5YE24v1gZ1pdzb9oSTCPmjpYgCALG1Rxbm8d%2BDBcKcljJmJiMrSjd2Ga20RY/TBK4/jbGE9xxPHuTFiU9Tm0NnTfZGUmQPY1muNvnmTD/tEgXxbW4ss3jNXs/VH5EOBEDnWG8tfUG0XrTFIJ/McLBMGEEBIwrnUvcVDHvY%2BXBmHqAtBjtlH3eF8qkYKj4yRYIMbcjrIe7GXvlj7%2Br%2BySQcBjbYdXb1ZhR37QMB3HF6xUGh0EFAXs6TrN0AaQfU06yMZ/FQEDmCnrroNlpFmEk3s8iDgp6y8/ocKstCcFyvB%2BBwWikKgnBuNY1hxusmwfHsPCkAQmg96sADWIC7Ls1waP4kiSkSkhEv4GhH0ff59xwkiD8vo%2BcLwCggBoi/L6scCwEgaAsMkdBxOQlCf9/eg8RgBcFxKQLAbI8BbAAGp4EwAAdwAPLJEYJwBeNBaBFWIA/CA0Qb7RDCI0AAnmg3gBDmDECIYg6I2guhL24LwT%2BbBBCIIYLQEhw9eBYGiF4YAbgxC0Afgw8BmBzZGHEJw8BeBwLdDZJgIRI8oxdEUtsBefxqg31TNENKlCPBYBvv8b4pDVhUAMMABQsCEHINQcI/gggRBiHYFIGQghFAqHUJI3QFYDBGBQJPSw%2Bg8DRAfpAVYqBcKZCEQAWnNmyVQZhjhRMQbsXgqA5HEABFgEJVsqg1EyC4Bg7hPCtAkESIIhSBilHKBIJ%2BqR0i1CyMUqYZS6mFAYJUoY8QuBP06N0OosxJhtDKb0hpvQmgdMWF0npAymlDJmH0CZ1TumrAUDPLYEhe792vpIseHBjiqH8JKKJZpjjAGQEmUB1wEkQFwIQEg5NdhcGWLwehWhlhrw3lvLk/guDknJO0fw3yj4/OkBfK%2Bpcb67Pvo/Z%2BnDX4wEQCAQcyRFJ/34l/H%2BxAIisG2Aco5JyzkXK3mYXg9U7mZL0HY4QohxDOKpW4tQN8vGkHgWlZIxj9BbIhTszgiDFIosJKgKg%2BzDnHMkKc85xxLnXI8BioBDynkvJfh8ze1xN7qo1Zqzll9tkjyhbYGFryV7apJdyvVd9YVvNWOk9IzhJBAA%3D

	constexpr uint8x32_t() noexcept = default;

	/// NOTE: currently cannot be constexpr
	/// \return
	[[nodiscard]] static inline uint8x32_t random() noexcept {
		uint8x32_t ret;
		ret.v256 = fastrandombytes_m256i();
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	///
	/// \param __q31
	/// \param __q30
	/// \param __q29
	/// \param __q28
	/// \param __q27
	/// \param __q26
	/// \param __q25
	/// \param __q24
	/// \param __q23
	/// \param __q22
	/// \param __q21
	/// \param __q20
	/// \param __q19
	/// \param __q18
	/// \param __q17
	/// \param __q16
	/// \param __q15
	/// \param __q14
	/// \param __q13
	/// \param __q12
	/// \param __q11
	/// \param __q10
	/// \param __q09
	/// \param __q08
	/// \param __q07
	/// \param __q06
	/// \param __q05
	/// \param __q04
	/// \param __q03
	/// \param __q02
	/// \param __q01
	/// \param __q00
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

	[[nodiscard]] constexpr static inline uint8x32_t setr(char __q31, char __q30, char __q29, char __q28,
														  char __q27, char __q26, char __q25, char __q24,
														  char __q23, char __q22, char __q21, char __q20,
														  char __q19, char __q18, char __q17, char __q16,
														  char __q15, char __q14, char __q13, char __q12,
														  char __q11, char __q10, char __q09, char __q08,
														  char __q07, char __q06, char __q05, char __q04,
														  char __q03, char __q02, char __q01, char __q00) noexcept {
		return set(__q00,__q01,__q02,__q03,__q04,__q05,__q06,__q07,__q08,
		           __q09,__q10,__q11,__q12,__q13,__q14,__q15,__q16,__q17,
		           __q18,__q19,__q20,__q21,__q22,__q23,__q24,__q25,__q26,
		           __q27,__q28,__q29,__q30,__q31);
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
	[[nodiscard]] constexpr static inline uint8x32_t load(const void *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t aligned_load(const void *ptr) noexcept {
		auto *ptr256 = (__m256i *) ptr;
		uint8x32_t out;
		out.v256 = *ptr256;
		return out;
	}


	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t unaligned_load(const void *ptr) noexcept {
		__m256i_u const *ptr256 = (__m256i_u const *) ptr;
		const __m256i_u tmp = internal::unaligned_load_wrapper(ptr256);
		uint8x32_t out;
		out.v256 = tmp;
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(void *ptr, const uint8x32_t in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		unaligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(void *ptr, const uint8x32_t in) noexcept {
		auto *ptr256 = (__m256i *) ptr;
		*ptr256 = in.v256;
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const uint8x32_t in) noexcept {
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
		out.v256 = (__m256i)(~(__v4du)in1.v256 & (__v4du)in2.v256);
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
		__m256i tmp;
		const __m256i maskl = __extension__(__m256i)(__v16hi){0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
		const __m256i maskh = __extension__(__m256i)(__v16hi){(short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00};

		const __m256i in1l = (__m256i) ((__v4du) in1.v256 & (__v4du) maskl);
		const __m256i in2l = (__m256i) ((__v4du) in2.v256 & (__v4du) maskl);

		const __m256i in1h = (__m256i) __builtin_ia32_psrlwi256((__v16hi) ((__v4du) in1.v256 & (__v4du) maskh), 8);
		const __m256i in2h = (__m256i) __builtin_ia32_psrlwi256((__v16hi) ((__v4du) in2.v256 & (__v4du) maskh), 8);

		out.v256 = (__m256i) ((__v16hu) in1l * (__v16hu) in2l);
		tmp = (__m256i) ((__v16hu) in1h * (__v16hu) in2h);

		tmp = (__m256i) __builtin_ia32_psllwi256((__v16hi) tmp, 8u);
		out.v256 = (__m256i) ((__v4du) tmp ^ (__v4du) out.v256);
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
		out.v256 = (__m256i) __builtin_ia32_psllwi256((__v16hi) out.v256, in2);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t slri(const uint8x32_t in1,
	                                                      const uint8_t in2) noexcept {
		ASSERT(in2 <= 8);
		const uint8x32_t mask = set1(((1u << (8u - in2)) - 1u) << in2);
		uint8x32_t out;
		out = uint8x32_t::and_(in1, mask);
#ifndef __clang__
		out.v256 = (__m256i) __builtin_ia32_psrlwi256((__v16hi) out.v256, in2);
#else
		out.v256 = (__m256i) __builtin_ia32_psllwi256((__v16hi) out.v256, in2);
#endif
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline int gt(const uint8x32_t in1, const uint8x32_t in2) noexcept {
		const __m256i tmp = (__m256i) ((__v32qs) in1.v256 > (__v32qs) in2.v256);
#ifndef __clang__
		return __builtin_ia32_pmovmskb256((__v32qi) tmp);
#else
		return __builtin_ia32_pmovmskb256((__v32qi) tmp);
#endif
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline int cmp(const uint8x32_t in1, const uint8x32_t in2) noexcept {
		const __m256i tmp = (__m256i) ((__v32qi) in1.v256 == (__v32qi) in2.v256);
#ifndef __clang__
		return __builtin_ia32_pmovmskb256((__v32qi) tmp);
#else
		return __builtin_ia32_pmovmskb256((__v32qi) tmp);
#endif
	}

	///
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t popcnt(const uint8x32_t in) noexcept {
		uint8x32_t ret;
		ret.v256 = popcount_avx2_8(in.v256);
		return ret;
	}
};

struct uint16x16_t {
	union {
		uint8_t v8[32];
		uint16_t v16[16];
		uint32_t v32[8];
		uint64_t v64[4];
		__m256i v256;
	};

	///
	/// \return
	[[nodiscard]] static inline uint16x16_t random() noexcept {
		uint16x16_t ret;
		ret.v256 = fastrandombytes_m256i();
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
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline uint16x16_t load(const void *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t aligned_load(const void *ptr) noexcept {
		uint16x16_t out;
		auto *ptr256 = (__m256i *) ptr;
		out.v256 = *ptr256;
		return out;
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t unaligned_load(const void *ptr) noexcept {
		uint16x16_t out;
		__m256i_u const *ptr256 = (__m256i_u const *) ptr;
		__m256i_u tmp = internal::unaligned_load_wrapper(ptr256);
		out.v256 = tmp;
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(void *ptr, const uint16x16_t in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(void *ptr, const uint16x16_t in) noexcept {
		auto *ptr256 = (__m256i *) ptr;
		*ptr256 = in.v256;
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const uint16x16_t in) noexcept {
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
		ASSERT(in2 <= 8);
		uint16x16_t out;
		uint16x16_t mask = set1((1u << in2) - 1u);
		out = uint16x16_t::and_(in1, mask);
#ifndef __clang__
		out.v256 = (__m256i) __builtin_ia32_psllwi256((__v16hi) out.v256, in2);
#else
		out.v256 = _mm256_slli_epi16(out.v256, in2);
#endif
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t slri(const uint16x16_t in1,
	                                                       const uint8_t in2) noexcept {
		ASSERT(in2 <= 8);
		const uint16x16_t mask = set1(((1u << (8u - in2)) - 1u) << in2);
		uint16x16_t out;
		out = uint16x16_t::and_(in1, mask);
#ifndef __clang__
		out.v256 = (__m256i) __builtin_ia32_psrlwi256((__v16hi) out.v256, in2);
#else
		out.v256 = _mm256_srli_epi16(out.v256, in2);
#endif
		return out;
	}

	/// NOTE: this is a function which cannot be vectorized
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline int gt(const uint16x16_t in1, const uint16x16_t in2) noexcept {
		uint16x16_t tmp;
		tmp.v256 = (__m256i) ((__v16hi) in1.v256 > (__v16hi) in2.v256);

		int ret = 0;
		for (uint16_t i = 0; i < 16; i++) {
			ret ^= (tmp.v16[i] != 0) << i;
		}

		return ret;
	}

	/// NOTE: this is a function which cannot be vectorized
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline int cmp(const uint16x16_t in1, const uint16x16_t in2) noexcept {
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
};

struct uint32x8_t {
	union {
		uint8_t v8[32];
		uint16_t v16[16];
		uint32_t v32[8];
		uint64_t v64[4];
		__m256i v256;
	};

	///
	/// \return
	[[nodiscard]] static inline uint32x8_t random() noexcept {
		uint32x8_t ret;
		ret.v256 = fastrandombytes_m256i();
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
		uint32x8_t out;
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
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline uint32x8_t load(const void *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t aligned_load(const void *ptr) noexcept {
		uint32x8_t out;
		auto *ptr256 = (__m256i *) ptr;
		out.v256 = *ptr256;
		return out;
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t unaligned_load(const void *ptr) noexcept {
		uint32x8_t out;
		__m256i_u const *ptr256 = (__m256i_u const *) ptr;
		const __m256i_u tmp = internal::unaligned_load_wrapper(ptr256);
		out.v256 = tmp;
		return out;
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

		aligned_store(ptr, in);
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
		uint32x8_t out;
		out.v256 = in1.v256 ^ in2.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t and_(const uint32x8_t in1,
	                                                      const uint32x8_t in2) noexcept {
		uint32x8_t out;
		out.v256 = in1.v256 & in2.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t or_(const uint32x8_t in1,
	                                                     const uint32x8_t in2) noexcept {
		uint32x8_t out;
		out.v256 = in1.v256 | in2.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t andnot(const uint32x8_t in1,
	                                              const uint32x8_t in2) noexcept {
		uint32x8_t out;
		out.v256 = (__m256i) (~(__v4du) in1.v256 & (__v4du) in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t not_(const uint32x8_t in1) noexcept {
		uint32x8_t out;
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
		uint32x8_t out;
		out.v256 = (__m256i) ((__v8su) in1.v256 + (__v8su) in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t sub(const uint32x8_t in1,
	                                                     const uint32x8_t in2) noexcept {
		uint32x8_t out;
		out.v256 = (__m256i) ((__v8su) in1.v256 - (__v8su) in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t mullo(const uint32x8_t in1,
	                                                       const uint32x8_t in2) noexcept {
		uint32x8_t out;
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
		uint32x8_t out;
		uint32x8_t mask = set1((1u << in2) - 1u);
		out = uint32x8_t::and_(in1, mask);
#ifndef __clang__
		out.v256 = (__m256i)__builtin_ia32_psllwi256 ((__v16hi)out.v256, in2);
#else
		out.v256 = _mm256_slli_epi32(out.v256, in2);
#endif
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t slri(const uint32x8_t in1,
	                                            const uint8_t in2) noexcept {
		ASSERT(in2 <= 8);
		const uint32x8_t mask = set1(((1u << (8u - in2)) - 1u) << in2);
		uint32x8_t out;
		out = uint32x8_t::and_(in1, mask);
#ifndef __clang__
  		out.v256 = (__m256i)__builtin_ia32_psrldi256 ((__v8si)out.v256, in2);
#else
		out.v256 = _mm256_srli_epi32(out.v256, in2);
#endif
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline int gt(const uint32x8_t in1, const uint32x8_t in2) noexcept {
		const __m256i tmp = (__m256i) ((__v8si) in1.v256 > (__v8si) in2.v256);
#ifndef __clang__
		return __builtin_ia32_movmskps256((__v8sf) tmp);
#else
		return __builtin_ia32_movmskps256((__v8sf) tmp);
#endif
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline int cmp(const uint32x8_t in1, const uint32x8_t in2) noexcept {
  		const __m256i tmp = (__m256i)((__v8si)in1.v256 == (__v8si)in2.v256);
#ifndef __clang__
  		return __builtin_ia32_movmskps256 ((__v8sf)tmp);
#else
		return _mm256_movemask_ps((__m256) tmp);
#endif
	}

	///
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t popcnt(const uint32x8_t in) noexcept {
		uint32x8_t ret;

#ifdef USE_AVX512
#ifndef __clang__
  		ret.v256 = (__m256i) __builtin_ia32_vpopcountd_v8si ((__v8si)in.v256);
#else 
		ret.v256 = __builtin_ia32_vpopcntd_256((__v8si)in.v256);
#endif
#else
		ret.v256 = popcount_avx2_32(in.v256);
#endif
		return ret;
	}

	/// \tparam scale
	/// \param ptr
	/// \param data
	/// \return
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline uint32x8_t gather(const void *ptr, const uint32x8_t data) noexcept {
		uint32x8_t ret;

#ifndef __clang__
  		ret.v256 = (__m256i) __builtin_ia32_gathersiv8si ((__v8si) _mm256_setzero_si256 (), \
					 (int const *) (ptr),		   \
					 (__v8si)(__m256i) (data.v256),	   \
					 (__v8si)_mm256_set1_epi32 (-1),   \
					 (int) (scale));
#else
		ret.v256 = _mm256_i32gather_epi32((int *) ptr, data.v256, scale);
#endif
		return ret;
	}

	///
	/// \param in
	/// \param perm
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t permute(const uint32x8_t in, const uint32x8_t perm) noexcept {
		uint32x8_t ret;
		ret.v256 = (__m256i) __builtin_ia32_permvarsi256((__v8si) in.v256, (__v8si) perm.v256);
		return ret;
	}

	/// moves
	[[nodiscard]] constexpr static inline uint8_t move(const uint32x8_t in) noexcept {
#ifndef __clang__
  		return __builtin_ia32_movmskps256 ((__v8sf)in.v256);
#else
  		return __builtin_ia32_movmskps256 ((__v8sf)in.v256);
#endif
	}
};

struct uint64x4_t {
	union {
		uint8_t v8[32];
		uint16_t v16[16];
		uint32_t v32[8];
		uint64_t v64[4];
		__m256i v256;
	};

	///
	/// \return
	[[nodiscard]] static inline uint64x4_t random() noexcept {
		uint64x4_t ret;
		ret.v256 = fastrandombytes_m256i();
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
		out.v256 = __extension__(__m256i)(__v4di){(long long) d, (long long) c, (long long) b, (long long) a};
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
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline uint64x4_t load(const void *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t aligned_load(const void *ptr) noexcept {
		uint64x4_t out;
		auto *ptr256 = (__m256i *) ptr;
		out.v256 = *ptr256;
		return out;
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t unaligned_load(const void *ptr) noexcept {
		uint64x4_t out;
		__m256i_u const *ptr256 = (__m256i_u const *) ptr;
		const __m256i_u tmp = internal::unaligned_load_wrapper(ptr256);
		out.v256 = tmp;
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(void *ptr, const uint64x4_t in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(void *ptr, const uint64x4_t in) noexcept {
		auto *ptr256 = (__m256i *) ptr;
		*ptr256 = in.v256;
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const uint64x4_t in) noexcept {
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

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t mullo(const uint64x4_t in1,
	                                                       const uint64x4_t in2) noexcept {
		uint64x4_t out;
#ifdef USE_AVX512
		out.v256 = (__m256i) ((__v4du) in1.v256 * (__v4du) in2.v256);
#else
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = in1.v64[i] * in2.v64[i];
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
		uint64x4_t mask = set1((1u << in2) - 1u);
		out = uint64x4_t::and_(in1, mask);
#ifndef __clang__
  		out.v256 = (__m256i)__builtin_ia32_psllqi256 ((__v4di)out.v256, in2);
#else
		out.v256 = _mm256_slli_epi64(out.v256, in2);
#endif
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t slri(const uint64x4_t in1,
	                                            const uint8_t in2) noexcept {
		ASSERT(in2 <= 8);
		const uint64x4_t mask = set1(((1u << (8u - in2)) - 1u) << in2);
		uint64x4_t out;
		out = uint64x4_t::and_(in1, mask);
#ifndef __clang__
  		out.v256 = (__m256i)__builtin_ia32_psrlqi256 ((__v4di)out.v256, in2);
#else
		out.v256 = _mm256_srli_epi64(out.v256, in2);
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
#ifndef __clang__
  		ret.v256 = ((__m256i) __builtin_ia32_permdi256 ((__v4di)(__m256i)(in1.v256), (int)(in2)));
#else
		ret.v256 = _mm256_permute4x64_epi64(in1.v256, in2);
#endif
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline int gt(const uint64x4_t in1, const uint64x4_t in2) noexcept {
		const auto tmp = (__m256i) ((__v4di) in1.v256 > (__v4di) in2.v256);
#ifndef __clang__
		return __builtin_ia32_movmskpd256((__v4df) tmp);
#else
		return __builtin_ia32_movmskpd256((__v4df) tmp);
#endif
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline int cmp(const uint64x4_t in1, const uint64x4_t in2) noexcept {
#ifndef __clang__
  		const __m256i tmp = (__m256i) ((__v4di)in1.v256 == (__v4di)in2.v256);
  		return __builtin_ia32_movmskpd256 ((__v4df)tmp);
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
  		ret.v256 = (__m256i) __builtin_ia32_vpopcountq_v4di ((__v4di)in.v256);
#else
		ret.v256 = popcount_avx2_64(in.v256);
#endif
		return ret;
	}

	/// \tparam scale
	/// \param ptr
	/// \param data
	/// \return
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline uint64x4_t gather(const void *ptr, const uint32x4_t data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		uint64x4_t ret;
#ifndef __clang__
  		ret.v256 = (__m256i) __builtin_ia32_gathersiv4di ((__v4di) _mm256_setzero_si256 (), \
					 (long long const *) (ptr),	   \
					 (__v4si)(__m128i) (data.v128),	   \
					 (__v4di)_mm256_set1_epi64x (-1),  \
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
  		ret.v256 = (__m256i) __builtin_ia32_gatherdiv4di ((__v4di) _mm256_setzero_si256 (), \
					 (long long const *) (ptr),	   \
					 (__v4di)(__m256i) (data.v256),	   \
					 (__v4di)_mm256_set1_epi64x (-1),  \
					 (int) (scale));
#else
		ret.v256 = _mm256_i64gather_epi64((long long *) ptr, data.v256, scale);
#endif
		return ret;
	}


	[[nodiscard]] constexpr static inline uint8_t move(const uint64x4_t in1) noexcept {
#ifndef __clang__
  		return __builtin_ia32_movmskpd256 ((__v4df)in1.v256);
#else
		return _mm256_movemask_pd((__m256d) in1.v256);
#endif
	}
};
