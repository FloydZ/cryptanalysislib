#pragma once

#ifndef USE_AVX2
#error "no avx"
#endif

#include <immintrin.h>
#include <cstdint>
#include <cstdio>

#include "helper.h"

//
#if CUSTOM_ALIGNMENT
#define MM256_LOAD(x) _mm256_load_ps(x)
#define MM256_STORE(x, y) _mm256_store_ps(x, y)
#else
#define MM256_LOAD(x) _mm256_loadu_ps(x)
#define MM256_STORE(x, y) _mm256_storeu_ps(x, y)
#endif


#define MM256_LOAD_UNALIGNED(x) _mm256_loadu_ps(x)
#define MM256_STORE_UNALIGNED(x, y) _mm256_storeu_ps(x, y)

// Load instruction which do not touch the cache
#define MM256_STREAM_LOAD256(x) _mm256_stream_load_si256(x)
#define MM256_STREAM_LOAD128(x) _mm_stream_load_si128(x)
#define MM256_STREAM_LOAD64(x)  __asm volatile("movntdq %0, (%1)\n\t"       \
												:                           \
												: "x" (x), "r" (y)          \
												: "memory");
// Write instructions which do not touch the cache
#define MM256_STREAM256(x, y) _mm256_stream_load_si256(x, y)
#define MM256_STREAM128(x, y) _mm_stream_si128(x)
#define MM256_STREAM64(x, y)  __asm volatile("movntdq %0, (%1)\n\t"         \
												:                           \
												: "x" (y), "r" (x)          \
												: "memory");


namespace internal {
	/// helper function. This enforces the compiler to emit a `vmovdqu` instruction
	/// \param ptr pointer to data.
	///				No alignment needed
	/// 			but 32 bytes should be redeable
	/// \return unaligned `__m256i`
	constexpr static inline __m256i_u unaligned_load_wrapper(__m256i_u const *ptr) {
		return *ptr;
	}
}

union U256i {
	__m256i v;
	uint32_t a[8];
	uint64_t b[4];
};

struct uint8x32_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
		__m256i v256;
	};
	/// Example of how the constexpr implementation works:
	/// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGEgBykrgAyeAyYAHI%2BAEaYxCCSZqQADqgKhE4MHt6%2BASlpGQKh4VEssfGJtpj2jgJCBEzEBNk%2BflyBdpgOmfWNBMWRMXEJSQoNTS257bbj/WGDZcOJAJS2qF7EyOwc5gDMYcjeWADUJrtuY/iCAHQIZ9gmGgCCewdHmKfneCwsYQTEYVu90eLzM%2BwYhy8JzObmQlycwOeILGxC8DmOXj%2B/lUuzMAH0CKcAOxWJEaACcXgYmWJpJeFMelKxBOOxwAbv5TgBWCy4kxcgAiZzpjMZmMEXAAbCy2VL%2BRY5YLhSDRRTxQRcTK%2BTzjv5%2BULdiKGWq/pLJDKzfLjpJ9cqyeTGXi8SwzFzJXh2a7JXb6Q6iQa6azWSCg8c0AwxphVMliMcxkxHMhjmFaPMMVicfjCaoSHiIOGxunBNjNYSwlxSCqKaGa7W6/WC4T1SWs8mGGZlrSqw6TcXMyz1gQfaryYPrmyvZ8BccIE6XW68J2oE62ZJ0F5luXx5P%2BdgZyu1xuwmZt27lsPjeSAPRXscTt1T47O%2BfSnPEPHpL0QLf3yWkNsnr%2B56Gt2jLEJgBAbAwxyDhefpCmSvo3le3bIdybjJI0rDHMk/yoTe6HgZBxAMN2jZRjGcYNImbapuERYEC2LJUmIeDAOE6B4rQqBMOg%2BYCIWbKoHg6DHAAVLhxCdiYJKgRSTBeEQ4mSTuuzTrOzpeh6YnLJJcFihmpYwYp%2BkUneqnThJ/xeqZ163opp6So%2Bz5elxPHrh%2BeBfhpL7abp/zAUaPbkkRUHGUOIH2jJCFPKGIZBsh%2BFXuhmHENhen2mh/IYVhLBtmRAkEBRsbxjRKZpkJIkYswqbsZgnFjCQmAQJVolWWQYaFQxTFlgw0myVFFJzlpynWQ%2BZzqcNC7if5xC2e1Flto5Ppxf6KprZF9JPOR0axq1xxFWMEDNiyYmDv%2Bjbdad5YXV1J2EmJx63RGTZ/EZfVds8ob8LGx1va2HoTccGjCsm1i4p8bjHKRhpg5YE24v1gZ1pdzb9oSTCPmjpYgCALG1Rxbm8d%2BDBcKcljJmJiMrSjd2Ga20RY/TBK4/jbGE9xxPHuTFiU9Tm0NnTfZGUmQPY1muNvnmTD/tEgXxbW4ss3jNXs/VH5EOBEDnWG8tfUG0XrTFIJ/McLBMGEEBIwrnUvcVDHvY%2BXBmHqAtBjtlH3eF8qkYKj4yRYIMbcjrIe7GXvlj7%2Br%2BySQcBjbYdXb1ZhR37QMB3HF6xUGh0EFAXs6TrN0AaQfU06yMZ/FQEDmCnrroNlpFmEk3s8iDgp6y8/ocKstCcFyvB%2BBwWikKgnBuNY1hxusmwfHsPCkAQmg96sADWIC7Ls1waP4kiSkSkhEv4GhH0ff59xwkiD8vo%2BcLwCggBoi/L6scCwEgaAsMkdBxOQlCf9/eg8RgBcFxKQLAbI8BbAAGp4EwAAdwAPLJEYJwBeNBaBFWIA/CA0Qb7RDCI0AAnmg3gBDmDECIYg6I2guhL24LwT%2BbBBCIIYLQEhw9eBYGiF4YAbgxC0Afgw8BmBzZGHEJw8BeBwLdDZJgIRI8oxdEUtsBefxqg31TNENKlCPBYBvv8b4pDVhUAMMABQsCEHINQcI/gggRBiHYFIGQghFAqHUJI3QFYDBGBQJPSw%2Bg8DRAfpAVYqBcKZCEQAWnNmyVQZhjhRMQbsXgqA5HEABFgEJVsqg1EyC4Bg7hPCtAkESIIhSBilHKBIJ%2BqR0i1CyMUqYZS6mFAYJUoY8QuBP06N0OosxJhtDKb0hpvQmgdMWF0npAymlDJmH0CZ1TumrAUDPLYEhe792vpIseHBjiqH8JKKJZpjjAGQEmUB1wEkQFwIQEg5NdhcGWLwehWhlhrw3lvLk/guDknJO0fw3yj4/OkBfK%2Bpcb67Pvo/Z%2BnDX4wEQCAQcyRFJ/34l/H%2BxAIisG2Aco5JyzkXK3mYXg9U7mZL0HY4QohxDOKpW4tQN8vGkHgWlZIxj9BbIhTszgiDFIosJKgKg%2BzDnHMkKc85xxLnXI8BioBDynkvJfh8ze1xN7qo1Zqzll9tkjyhbYGFryV7apJdyvVd9YVvNWOk9IzhJBAA%3D

	constexpr uint8x32_t() noexcept = default;

	///
	/// \return
	static inline uint8x32_t random(){
		uint8x32_t ret;
		ret.v256 = fastrandombytes_m256i();
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary=false, bool hex=false);

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
		                char __q03, char __q02, char __q01, char __q00){
		uint8x32_t out;
		out.v256 = __extension__ (__m256i)(__v32qi){
		        __q00, __q01, __q02, __q03, __q04, __q05, __q06, __q07,
		        __q08, __q09, __q10, __q11, __q12, __q13, __q14, __q15,
		        __q16, __q17, __q18, __q19, __q20, __q21, __q22, __q23,
		        __q24, __q25, __q26, __q27, __q28, __q29, __q30, __q31
		};

		return out;
	}

	///
	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t set1(const uint8_t __A) {
		uint8x32_t out;
		out =  uint8x32_t::set(__A, __A, __A, __A, __A, __A, __A, __A,
		                       __A, __A, __A, __A, __A, __A, __A, __A,
		                       __A, __A, __A, __A, __A, __A, __A, __A,
		                       __A, __A, __A, __A, __A, __A, __A, __A);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned>
	constexpr static inline uint8x32_t load(const void *ptr) {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	constexpr static inline uint8x32_t aligned_load(const void *ptr) {
		auto *ptr256 = (__m256i *)ptr;
		uint8x32_t out;
		out.v256 = *ptr256;
		return out;
	}


	///
	/// \param ptr
	/// \return
	constexpr static inline uint8x32_t unaligned_load(const void *ptr) {
		__m256i_u const *ptr256 = (__m256i_u const *)ptr;
		__m256i_u tmp = internal::unaligned_load_wrapper(ptr256);
		uint8x32_t out;
		out.v256 = tmp;
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned>
	constexpr static inline void store(void *ptr, const uint8x32_t in) {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	static inline void aligned_store(void *ptr, const uint8x32_t in) {
		auto *ptr256 = (__m256i *)ptr;
		*ptr256 = in.v256;
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const uint8x32_t in) {
		auto *ptr256 = (__m256i_u *)ptr;
		*ptr256 = in.v256;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t xor_(const uint8x32_t in1,
	                              const uint8x32_t in2) {
		uint8x32_t out;
		out.v256 = (__m256i) ((__v4du)in1.v256 ^ (__v4du)in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t and_(const uint8x32_t in1,
	                              const uint8x32_t in2) {
		uint8x32_t out;
		out.v256 = (__m256i) ((__v4du)in1.v256 & (__v4du)in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t or_(const uint8x32_t in1,
						  const uint8x32_t in2) {
		uint8x32_t out;
		out.v256 = (__m256i) ((__v4du)in1.v256 | (__v4du)in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t andnot(const uint8x32_t in1,
	                                const uint8x32_t in2) {
		uint8x32_t out;
		// TODO only valid in gcc
		//out.v256 = (__m256i) __builtin_ia32_andnotsi256 ((__v4di)in1.v256, (__v4di)in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \return
	constexpr static inline uint8x32_t not_(const uint8x32_t in1) {
		uint8x32_t out;
		const uint8x32_t minus_one = set1(-1);
		out.v256 = (__m256i) ((__v4du)in1.v256 ^ (__v4du)minus_one.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t add(const uint8x32_t in1,
	                             const uint8x32_t in2) {
		uint8x32_t out;
		out.v256 = (__m256i) ((__v32qu)in1.v256 + (__v32qu)in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t sub(const uint8x32_t in1,
	                             const uint8x32_t in2) {
		uint8x32_t out;
		out.v256 = (__m256i) ((__v32qu)in1.v256 - (__v32qu)in2.v256);
		return out;
	}

	/// 8 bit mul lo
	/// \param in1
	/// \param in2
	/// \return
	static inline uint8x32_t mullo(const uint8x32_t in1,
	                               const uint8x32_t in2) {
		uint8x32_t out;
		const __m256i maskl = _mm256_set1_epi16(0x000f);
		const __m256i maskh = _mm256_set1_epi16(0x0f00);

		const __m256i in1l = _mm256_and_si256(in1.v256, maskl);
		const __m256i in2l = _mm256_and_si256(in2.v256, maskl);
		const __m256i in1h = _mm256_srli_epi16(_mm256_and_si256(in1.v256, maskh), 8u);
		const __m256i in2h = _mm256_srli_epi16(_mm256_and_si256(in2.v256, maskh), 8u);

		out.v256 = _mm256_mullo_epi16(in1l, in2l);
		const __m256i tho = _mm256_slli_epi16(_mm256_mullo_epi16(in1h, in2h), 8u);
		out.v256 = _mm256_xor_si256(tho, out.v256);
		return out;
	}
};

struct uint16x16_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
		__m256i v256;
	};

	///
	/// \return
	inline uint16x16_t random(){
		uint16x16_t ret;
		ret.v256 = fastrandombytes_m256i();
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	inline void print(bool binary=false, bool hex=false);

	///
	/// \param a
	/// \return
	static inline uint16x16_t set1(const uint16_t a) {
		uint16x16_t out;
		out.v256 = _mm256_set1_epi16(a);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned>
	static inline uint16x16_t load(const void *ptr) {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	static inline uint16x16_t aligned_load(const void *ptr) {
		uint16x16_t out;
		out.v256 = _mm256_load_si256((__m256i *)ptr);
		return out;
	}

	///
	/// \param ptr
	/// \return
	static inline uint16x16_t unaligned_load(const void *ptr) {
		uint16x16_t out;
		out.v256 = _mm256_loadu_si256((__m256i *)ptr);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned>
	static inline void store(void *ptr, const uint16x16_t in) {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	static inline void aligned_store(void *ptr, const uint16x16_t in) {
		_mm256_store_si256((__m256i *)ptr, in.v256);
	}

	///
	/// \param ptr
	/// \param in
	static inline void unaligned_store(void *ptr, const uint16x16_t in) {
		_mm256_storeu_si256((__m256i *)ptr, in.v256);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint16x16_t xor_(const uint16x16_t in1,
							const uint16x16_t in2) {
		uint16x16_t out;
		out.v256 = _mm256_xor_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint16x16_t and_(const uint16x16_t in1,
							const uint16x16_t in2) {
		uint16x16_t out;
		out.v256 = _mm256_and_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint16x16_t or_(const uint16x16_t in1,
						   const uint16x16_t in2) {
		uint16x16_t out;
		out.v256 = _mm256_or_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint16x16_t andnot(const uint16x16_t in1,
							  const uint16x16_t in2) {
		uint16x16_t out;
		out.v256 = _mm256_andnot_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \return
	inline uint16x16_t not_(const uint16x16_t in1) {
		uint16x16_t out;
		const __m256i minus_one = _mm256_set1_epi8(-1);
		out.v256 = _mm256_xor_si256(in1.v256, minus_one);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint16x16_t add(const uint16x16_t in1,
						   const uint16x16_t in2) {
		uint16x16_t out;
		out.v256 = _mm256_add_epi16(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint16x16_t sub(const uint16x16_t in1,
						   const uint16x16_t in2) {
		uint16x16_t out;
		out.v256 = _mm256_sub_epi16(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint16x16_t mullo(const uint16x16_t in1,
							 const uint16x16_t in2) {
		uint16x16_t out;
		out.v256 = _mm256_mullo_epi16(in1.v256, in2.v256);
		return out;
	}
};

struct uint32x8_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
		__m256i v256;
	};

	///
	/// \return
	inline uint32x8_t random(){
		uint32x8_t ret;
		ret.v256 = fastrandombytes_m256i();
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	inline void print(bool binary=false, bool hex=false);

	///
	/// \param a
	/// \return
	static inline uint32x8_t set1(const uint32_t a) {
		uint32x8_t out;
		out.v256 = _mm256_set1_epi32(a);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned>
	static inline uint32x8_t load(const void *ptr) {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	static inline uint32x8_t aligned_load(const void *ptr) {
		uint32x8_t out;
		out.v256 = _mm256_load_si256((__m256i *)ptr);
		return out;
	}

	///
	/// \param ptr
	/// \return
	static inline uint32x8_t unaligned_load(const void *ptr) {
		uint32x8_t out;
		out.v256 = _mm256_loadu_si256((__m256i *)ptr);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned>
	static inline void store(void *ptr, const uint32x8_t in) {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	static inline void aligned_store(void *ptr, const uint32x8_t in) {
		_mm256_store_si256((__m256i *)ptr, in.v256);
	}

	///
	/// \param ptr
	/// \param in
	static inline void unaligned_store(void *ptr, const uint32x8_t in) {
		_mm256_storeu_si256((__m256i *)ptr, in.v256);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint32x8_t xor_(const uint32x8_t in1,
						   const uint32x8_t in2) {
		uint32x8_t out;
		out.v256 = _mm256_xor_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint32x8_t and_(const uint32x8_t in1,
						   const uint32x8_t in2) {
		uint32x8_t out;
		out.v256 = _mm256_and_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint32x8_t or_(const uint32x8_t in1,
						  const uint32x8_t in2) {
		uint32x8_t out;
		out.v256 = _mm256_or_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint32x8_t andnot(const uint32x8_t in1,
							 const uint32x8_t in2) {
		uint32x8_t out;
		out.v256 = _mm256_andnot_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \return
	inline uint32x8_t not_(const uint32x8_t in1) {
		uint32x8_t out;
		const __m256i minus_one = _mm256_set1_epi8(-1);
		out.v256 = _mm256_xor_si256(in1.v256, minus_one);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint32x8_t add(const uint32x8_t in1,
						  const uint32x8_t in2) {
		uint32x8_t out;
		out.v256 = _mm256_add_epi32(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint32x8_t sub(const uint32x8_t in1,
						  const uint32x8_t in2) {
		uint32x8_t out;
		out.v256 = _mm256_sub_epi32(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint32x8_t mullo(const uint32x8_t in1,
							const uint32x8_t in2) {
		uint32x8_t out;
		out.v256 = _mm256_mullo_epi16(in1.v256, in2.v256);
		return out;
	}
};

struct uint64x4_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
		__m256i v256;
	};

	///
	/// \return
	inline uint64x4_t random(){
		uint64x4_t ret;
		ret.v256 = fastrandombytes_m256i();
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	inline void print(bool binary=false, bool hex=false);

	///
	/// \param a
	/// \return
	static inline uint64x4_t set1(const uint64_t a) {
		uint64x4_t out;
		out.v256 = _mm256_set1_epi64x(a);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned>
	static inline uint64x4_t load(const void *ptr) {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	static inline uint64x4_t aligned_load(const void *ptr) {
		uint64x4_t out;
		out.v256 = _mm256_load_si256((__m256i *)ptr);
		return out;
	}

	///
	/// \param ptr
	/// \return
	static inline uint64x4_t unaligned_load(const void *ptr) {
		uint64x4_t out;
		out.v256 = _mm256_loadu_si256((__m256i *)ptr);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned>
	static inline void store(void *ptr, const uint64x4_t in) {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	static inline void aligned_store(void *ptr, const uint64x4_t in) {
		_mm256_store_si256((__m256i *)ptr, in.v256);
	}

	///
	/// \param ptr
	/// \param in
	static inline void unaligned_store(void *ptr, const uint64x4_t in) {
		_mm256_storeu_si256((__m256i *)ptr, in.v256);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint64x4_t xor_(const uint64x4_t in1,
						   const uint64x4_t in2) {
		uint64x4_t out;
		out.v256 = _mm256_xor_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint64x4_t and_(const uint64x4_t in1,
						   const uint64x4_t in2) {
		uint64x4_t out;
		out.v256 = _mm256_and_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint64x4_t or_(const uint64x4_t in1,
						  const uint64x4_t in2) {
		uint64x4_t out;
		out.v256 = _mm256_or_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint64x4_t andnot(const uint64x4_t in1,
							 const uint64x4_t in2) {
		uint64x4_t out;
		out.v256 = _mm256_andnot_si256(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \return
	inline uint64x4_t not_(const uint64x4_t in1) {
		uint64x4_t out;
		const __m256i minus_one = _mm256_set1_epi8(-1);
		out.v256 = _mm256_xor_si256(in1.v256, minus_one);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint64x4_t add(const uint64x4_t in1,
						  const uint64x4_t in2) {
		uint64x4_t out;
		out.v256 = _mm256_add_epi64(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint64x4_t sub(const uint64x4_t in1,
						  const uint64x4_t in2) {
		uint64x4_t out;
		out.v256 = _mm256_sub_epi64(in1.v256, in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint64x4_t mullo(const uint64x4_t in1,
							const uint64x4_t in2) {
		ASSERT(false);
		uint64x4_t out;
		return out;
	}
};

bool operator==(const uint8x32_t& a, const uint8x32_t& b){
	const __m256i tmp = _mm256_cmpeq_epi8(a.v256, b.v256);
	const int mask = _mm256_movemask_epi8(tmp);
	return mask == 0xffffffff;
}

