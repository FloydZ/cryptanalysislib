#ifndef CRYPTANALYSISLIB_SIMD_H
#define CRYPTANALYSISLIB_SIMD_H

#include <cstdint>
#include <cinttypes>

#include "helper.h"
#include "random.h"



#if defined(USE_AVX2)

#include "simd/avx2.h"

#elif defined(USE_ARM)

#include "simd/neon.h"

#elif defined(USE_RISCV)

#include "simd/riscv.h"

#else


struct uint32x4_t {
	union {
		uint8_t  v8 [16];
		uint16_t v16[ 8];
		uint32_t v32[ 4];
		uint64_t v64[ 2];
		__m128i  v128;
	};

	constexpr static inline uint32x4_t set(uint64_t a, uint64_t b) {
		uint32x4_t ret;
		ret.v64[0] = a;
		ret.v64[1] = b;
		return ret;
	}
};

struct uint64x2_t {
	union {
		uint8_t  v8 [16];
		uint16_t v16[ 8];
		uint32_t v32[ 4];
		uint64_t v64[ 2];
		__m128i  v128;
	};
};
struct uint8x32_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
	};

	///
	/// \return
	static inline uint8x32_t random() {
		uint8x32_t ret;
		for (uint32_t i=0; i<4; i++) {
			ret.v64[i] = fastrandombytes_uint64();
		}

		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary=false, bool hex=false) const;

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
		uint8x32_t out = {0};
		out.v8[ 0] = __q31;
		out.v8[ 1] = __q30;
		out.v8[ 2] = __q29;
		out.v8[ 3] = __q28;
		out.v8[ 4] = __q27;
		out.v8[ 5] = __q26;
		out.v8[ 6] = __q25;
		out.v8[ 7] = __q24;
		out.v8[ 8] = __q23;
		out.v8[ 9] = __q22;
		out.v8[10] = __q21;
		out.v8[11] = __q20;
		out.v8[12] = __q19;
		out.v8[13] = __q18;
		out.v8[14] = __q17;
		out.v8[15] = __q16;
		out.v8[16] = __q15;
		out.v8[17] = __q14;
		out.v8[18] = __q13;
		out.v8[19] = __q12;
		out.v8[20] = __q11;
		out.v8[21] = __q10;
		out.v8[22] = __q09;
		out.v8[23] = __q08;
		out.v8[24] = __q07;
		out.v8[25] = __q06;
		out.v8[26] = __q05;
		out.v8[27] = __q04;
		out.v8[28] = __q03;
		out.v8[29] = __q02;
		out.v8[30] = __q01;
		out.v8[31] = __q00;
		return out;
	}

	/// sets all 32 8bit limbs to `a`
	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t set1(const uint8_t a) {
		uint8x32_t out = {0};
		out =  uint8x32_t::set(a, a, a, a, a, a, a, a,
		                       a, a, a, a, a, a, a, a,
		                       a, a, a, a, a, a, a, a,
		                       a, a, a, a, a, a, a, a);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned=false>
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
		uint8x32_t out = {0};
		const uint64_t *ptr64 = (uint64_t *)ptr;
		for (uint32_t i = 0; i < 3; i++) {
			out.v64[i] = ptr64[i];
		}
		return out;
	}


	///
	/// \param ptr
	/// \return
	constexpr static inline uint8x32_t unaligned_load(const void *ptr) {
		uint8x32_t out = {0};
		const uint64_t *ptr64 = (uint64_t *)ptr;
		for (uint32_t i = 0; i < 3; i++) {
			out.v64[i] = ptr64[i];
		}
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned=false>
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
		uint64_t *ptr64 = (uint64_t *)ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ptr64[i] = in.v64[i];
		}
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const uint8x32_t in) {
		uint64_t *ptr64 = (uint64_t *)ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ptr64[i] = in.v64[i];
		}
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t xor_(const uint8x32_t in1,
	                              const uint8x32_t in2) {
		uint8x32_t out = {0};
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = in1.v64[i] ^ in2.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t and_(const uint8x32_t in1,
	                              const uint8x32_t in2) {
		uint8x32_t out = {0};
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = in1.v64[i] & in2.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t or_(const uint8x32_t in1,
						  const uint8x32_t in2) {
		uint8x32_t out = {0};
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = in1.v64[i] | in2.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t andnot(const uint8x32_t in1,
	                                const uint8x32_t in2) {
		uint8x32_t out = {0};
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = ~(in1.v64[i] & in2.v64[i]);
		}
		return out;
	}

	///
	/// \param in1
	/// \return
	constexpr static inline uint8x32_t not_(const uint8x32_t in1) {
		uint8x32_t out = {0};
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = ~in1.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t add(
						const uint8x32_t in1,
	                    const uint8x32_t in2) {
		uint8x32_t out = {0};
		for (uint32_t i = 0; i < 32; i++) {
			out.v8[i] = in1.v8[i] + in2.v8[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t sub(const uint8x32_t in1,
	                             const uint8x32_t in2) {
		uint8x32_t out = {0};
		for (uint32_t i = 0; i < 32; i++) {
			out.v8[i] = in1.v8[i] - in2.v8[i];
		}
		return out;
	}

	/// 8 bit mul lo
	/// \param in1
	/// \param in2
	/// \return
	static inline uint8x32_t mullo(const uint8x32_t in1,
	                               const uint8x32_t in2) {
		uint8x32_t out = {0};
		for (uint32_t i = 0; i < 8; i++) {
			out.v8[i] = in1.v8[i] * in2.v8[i];
		}
		return out;
	}

	///
	static inline uint8x32_t mullo(const uint8x32_t in1,
	                               const uint8_t in2) {
		uint8x32_t rs = uint8x32_t::set1(in2);
		return uint8x32_t::mullo(in1, rs);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t slli(const uint8x32_t in1,
														  const uint8_t in2) {
		ASSERT(in2 <= 8);
		uint8x32_t out = {0};
		for (uint32_t i = 0; i < 32; i++) {
			out.v8[i] = in1.v8[i] << in2;
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t slri(const uint8x32_t in1,
														  const uint8_t in2) {
		ASSERT(in2 <= 8);
		uint8x32_t out = {0};
		for (uint32_t i = 0; i < 32; i++) {
			out.v8[i] = in1.v8[i] >> in2;
		}
		return out;
	}


	static inline int cmp(const uint8x32_t in1, const uint8x32_t in2) {
		int ret = 0;
		for (uint32_t i = 0; i < 32; i++){
			ret ^= (in1.v8[i] == in2.v8[i]) << i;
		}

		return ret;
	}

	static inline int gt(const uint8x32_t in1, const uint8x32_t in2) {
		int ret = 0;
		for (uint32_t i = 0; i < 32; i++){
			ret ^= (in1.v8[i] < in2.v8[i]) << i;
		}

		return ret;
	}

	static inline uint8x32_t popcnt(const uint8x32_t in) {
		uint8x32_t ret;

		for(uint32_t i = 0; i < 32; i++) {
			ret.v8[i] = __builtin_popcount(in.v8[i]);
		}
		return ret;
	}

};

struct uint16x16_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
	};

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary=false, bool hex=false) const;

	static inline int cmp(const uint16x16_t in1, const uint16x16_t in2) {
		int ret = 0;
		for (uint32_t i = 0; i < 16; i++){
			ret ^= (in1.v16[i] == in2.v16[i]) << i;
		}

		return ret;
	}

	static inline int gt(const uint16x16_t in1, const uint16x16_t in2) {
		int ret = 0;
		for (uint32_t i = 0; i < 16; i++){
			ret ^= (in1.v16[i] < in2.v16[i]) << i;
		}

		return ret;
	}

	static inline uint16x16_t popcnt(const uint8x32_t in) {
		uint8x32_t ret;

		for(uint32_t i = 0; i < 16; i++) {
			ret.v16[i] = __builtin_popcount(in.v16[i]);
		}
		return ret;
	}
};

struct uint32x8_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
	};
	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary=false, bool hex=false) const;

	static inline int cmp(const uint32x8_t in1, const uint32x8_t in2) {
		int ret = 0;
		for (uint32_t i = 0; i < 8; i++){
			ret ^= (in1.v32[i] == in2.v32[i]) << i;
		}

		return ret;
	}

	static inline int gt(const uint32x8_t in1, const uint32x8_t in2) {
		int ret = 0;
		for (uint32_t i = 0; i < 8; i++){
			ret ^= (in1.v32[i] < in2.v32[i]) << i;
		}

		return ret;
	}

	static inline uint32x8_t popcnt(const uint32x8_t in) {
		uint32x8_t ret;

		for(uint32_t i = 0; i < 8; i++) {
			ret.v32[i] = __builtin_popcount(in.v32[i]);
		}
		return ret;
	}
};

struct uint64x4_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
	};

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary=false, bool hex=false) const;

	static inline int permute(const uint64x4_t in1, const uint32_t in2) {
		uint64x4_t ret;
		const int mask = 0b11;
		for (uint64x4_t i = 0; i < 4; i++) {
			const int pos = (in2 >> (2*i)) & mask;
			in1.v64[pos] == in2.v64[i];
		}

		return ret;
	}

	static inline int cmp(const uint64x4_t in1, const uint64x4_t in2) {
		int ret = 0;
		for (uint64x4_t i = 0; i < 4; i++){
			ret ^= (in1.v64[i] == in2.v64[i]) << i;
		}

		return ret;
	}

	static inline int gt(const uint64x4_t in1, const uint64x4_t in2) {
		int ret = 0;
		for (uint32_t i = 0; i < 4; i++){
			ret ^= (in1.v64[i] < in2.v64[i]) << i;
		}

		return ret;
	}

	static inline uint64x4_t popcnt(const uint64x4_t in) {
		uint32x8_t ret;

		for(uint32_t i = 0; i < i; i++) {
			ret.v64[i] = __builtin_popcountll(in.v64[i]);
		}
		return ret;
	}
};



#endif // no SIMD uinit available

///
inline uint8x32_t operator* (const uint8x32_t& lhs, const uint8x32_t& rhs) {
	return uint8x32_t::mullo(lhs, rhs);
}
inline uint8x32_t operator* (const uint8x32_t& lhs, const uint8_t & rhs) {
	return uint8x32_t::mullo(lhs, rhs);
}
inline uint8x32_t operator* (const uint8_t & lhs, const uint8x32_t & rhs) {
	return uint8x32_t::mullo(rhs, lhs);
}
inline uint8x32_t operator+ (const uint8x32_t& lhs, const uint8x32_t& rhs) {
	return uint8x32_t::add(lhs, rhs);
}
inline uint8x32_t operator- (const uint8x32_t& lhs, const uint8x32_t& rhs) {
	return uint8x32_t::sub(lhs, rhs);
}
inline uint8x32_t operator& (const uint8x32_t& lhs, const uint8x32_t& rhs) {
	return uint8x32_t::and_(lhs, rhs);
}
inline uint8x32_t operator^ (const uint8x32_t& lhs, const uint8x32_t& rhs) {
	return uint8x32_t::xor_(lhs, rhs);
}
inline uint8x32_t operator| (const uint8x32_t& lhs, const uint8x32_t& rhs) {
	return uint8x32_t::or_(lhs, rhs);
}
inline uint8x32_t operator~ (const uint8x32_t& lhs) {
	return uint8x32_t::not_(lhs);
}
inline uint8x32_t operator>> (const uint8x32_t& lhs, const uint32_t rhs) {
	return uint8x32_t::slri(lhs, rhs);
}
inline uint8x32_t operator<< (const uint8x32_t& lhs, const uint32_t rhs) {
	return uint8x32_t::slli(lhs, rhs);
}


inline uint16x16_t operator* (const uint16x16_t& lhs, const uint16x16_t& rhs) {
	return uint16x16_t::mullo(lhs, rhs);
}
inline uint16x16_t operator* (const uint16x16_t& lhs, const uint8_t & rhs) {
	return uint16x16_t::mullo(lhs, rhs);
}
inline uint16x16_t operator* (const uint8_t & lhs, const uint16x16_t & rhs) {
	return uint16x16_t::mullo(rhs, lhs);
}
inline uint16x16_t operator+ (const uint16x16_t& lhs, const uint16x16_t& rhs) {
	return uint16x16_t::add(lhs, rhs);
}
inline uint16x16_t operator- (const uint16x16_t& lhs, const uint16x16_t& rhs) {
	return uint16x16_t::sub(lhs, rhs);
}
inline uint16x16_t operator& (const uint16x16_t& lhs, const uint16x16_t& rhs) {
	return uint16x16_t::and_(lhs, rhs);
}
inline uint16x16_t operator^ (const uint16x16_t& lhs, const uint16x16_t& rhs) {
	return uint16x16_t::xor_(lhs, rhs);
}
inline uint16x16_t operator| (const uint16x16_t& lhs, const uint16x16_t& rhs) {
	return uint16x16_t::or_(lhs, rhs);
}
inline uint16x16_t operator~ (const uint16x16_t& lhs) {
	return uint16x16_t::not_(lhs);
}
inline uint16x16_t operator>> (const uint16x16_t& lhs, const uint32_t rhs) {
	return uint16x16_t::slri(lhs, rhs);
}
inline uint16x16_t operator<< (const uint16x16_t& lhs, const uint32_t rhs) {
	return uint16x16_t::slli(lhs, rhs);
}


inline uint32x8_t operator* (const uint32x8_t& lhs, const uint32x8_t& rhs) {
	return uint32x8_t::mullo(lhs, rhs);
}
inline uint32x8_t operator* (const uint32x8_t& lhs, const uint8_t & rhs) {
	return uint32x8_t::mullo(lhs, rhs);
}
inline uint32x8_t operator* (const uint8_t & lhs, const uint32x8_t & rhs) {
	return uint32x8_t::mullo(rhs, lhs);
}
inline uint32x8_t operator+ (const uint32x8_t& lhs, const uint32x8_t& rhs) {
	return uint32x8_t::add(lhs, rhs);
}
inline uint32x8_t operator- (const uint32x8_t& lhs, const uint32x8_t& rhs) {
	return uint32x8_t::sub(lhs, rhs);
}
inline uint32x8_t operator& (const uint32x8_t& lhs, const uint32x8_t& rhs) {
	return uint32x8_t::and_(lhs, rhs);
}
inline uint32x8_t operator^ (const uint32x8_t& lhs, const uint32x8_t& rhs) {
	return uint32x8_t::xor_(lhs, rhs);
}
inline uint32x8_t operator| (const uint32x8_t& lhs, const uint32x8_t& rhs) {
	return uint32x8_t::or_(lhs, rhs);
}
inline uint32x8_t operator~ (const uint32x8_t& lhs) {
	return uint32x8_t::not_(lhs);
}
inline uint32x8_t operator>> (const uint32x8_t& lhs, const uint32_t rhs) {
	return uint32x8_t::slri(lhs, rhs);
}
inline uint32x8_t operator<< (const uint32x8_t& lhs, const uint32_t rhs) {
	return uint32x8_t::slli(lhs, rhs);
}


inline uint64x4_t operator* (const uint64x4_t& lhs, const uint64x4_t& rhs) {
	return uint64x4_t::mullo(lhs, rhs);
}
inline uint64x4_t operator* (const uint64x4_t& lhs, const uint64_t & rhs) {
	return uint64x4_t::mullo(lhs, rhs);
}
inline uint64x4_t operator* (const uint8_t & lhs, const uint64x4_t & rhs) {
	return uint64x4_t::mullo(rhs, lhs);
}
inline uint64x4_t operator+ (const uint64x4_t& lhs, const uint64x4_t& rhs) {
	return uint64x4_t::add(lhs, rhs);
}
inline uint64x4_t operator- (const uint64x4_t& lhs, const uint64x4_t& rhs) {
	return uint64x4_t::sub(lhs, rhs);
}
inline uint64x4_t operator& (const uint64x4_t& lhs, const uint64x4_t& rhs) {
	return uint64x4_t::and_(lhs, rhs);
}
inline uint64x4_t operator^ (const uint64x4_t& lhs, const uint64x4_t& rhs) {
	return uint64x4_t::xor_(lhs, rhs);
}
inline uint64x4_t operator| (const uint64x4_t& lhs, const uint64x4_t& rhs) {
	return uint64x4_t::or_(lhs, rhs);
}
inline uint64x4_t operator~ (const uint64x4_t& lhs) {
	return uint64x4_t::not_(lhs);
}
inline uint64x4_t operator>> (const uint64x4_t& lhs, const uint32_t rhs) {
	return uint64x4_t::slri(lhs, rhs);
}
inline uint64x4_t operator<< (const uint64x4_t& lhs, const uint32_t rhs) {
	return uint64x4_t::slli(lhs, rhs);
}
//////////////////////////////////////////////////////////\

int operator==(const uint8x32_t& a, const uint8x32_t& b){
	return uint8x32_t::cmp(a, b);
}
int operator!=(const uint8x32_t& a, const uint8x32_t& b){
	return 0xffffffff ^ uint8x32_t::cmp(a, b);
}
int operator<(const uint8x32_t& a, const uint8x32_t& b){
	return uint8x32_t::gt(b, a);
}
int operator>(const uint8x32_t& a, const uint8x32_t& b){
	return uint8x32_t::gt(b, a);
}


int operator==(const uint16x16_t& a, const uint16x16_t& b){
	return uint16x16_t::cmp(a, b);
}
int operator!=(const uint16x16_t& a, const uint16x16_t& b){
	return 0xffff ^ uint16x16_t::cmp(a, b);
}
int operator<(const uint16x16_t& a, const uint16x16_t& b){
	return uint16x16_t::gt(b, a);
}
int operator>(const uint16x16_t& a, const uint16x16_t& b){
	return uint16x16_t::gt(a, b);
}


int operator==(const uint32x8_t& a, const uint32x8_t& b){
	return uint32x8_t::cmp(a, b);
}
int operator!=(const uint32x8_t& a, const uint32x8_t& b){
	return 0xff ^ uint32x8_t::cmp(a, b);
}
int operator<(const uint32x8_t& a, const uint32x8_t& b){
	return uint32x8_t::gt(b, a);
}
int operator>(const uint32x8_t& a, const uint32x8_t& b){
	return uint32x8_t::gt(a, b);
}

int operator==(const uint64x4_t & a, const uint64x4_t& b){
	return uint64x4_t::cmp(a, b);
}
int operator!=(const uint64x4_t& a, const uint64x4_t& b){
	return 0xf ^ uint64x4_t::cmp(a, b);
}
int operator<(const uint64x4_t& a, const uint64x4_t& b){
	return uint64x4_t::gt(b, a);
}
int operator>(const uint64x4_t& a, const uint64x4_t& b){
	return uint64x4_t::gt(a, b);
}





////////////////////////////////////////////////////////////


/// functions which are shared among all implementations.
constexpr inline void uint8x32_t::print(bool binary, bool hex) const {
	/// make sure that only one is defined
	ASSERT(binary + hex < 2);

	if (binary) {
		for (uint32_t i = 0; i < 32; i++) {
			printbinary(this->v8[i]);
		}

		return;
	}

	if (hex) {
		for (uint32_t i = 0; i < 32; i++) {
			printf("%hhx ", this->v8[i]);
		}

		return;
	}

	for (uint32_t i = 0; i < 32; i++) {
		printf("%u ", this->v8[i]);
	}
	printf("\n");
}

constexpr inline void uint16x16_t::print(bool binary, bool hex) const {
	/// make sure that only one is defined
	ASSERT(binary + hex < 2);

	if (binary) {
		for (uint32_t i = 0; i < 16; i++) {
			printbinary(this->v16[i]);
		}

		return;
	}

	if (hex) {
		for (uint32_t i = 0; i < 16; i++) {
			printf("%hx ", this->v16[i]);
		}

		return;
	}

	for (uint32_t i = 0; i < 16; i++) {
		printf("%u ", this->v16[i]);
	}
	printf("\n");
}

constexpr inline void uint32x8_t::print(bool binary, bool hex) const {
	/// make sure that only one is defined
	ASSERT(binary + hex < 2);

	if (binary) {
		for (uint32_t i = 0; i < 8; i++) {
			printbinary(this->v32[i]);
		}

		return;
	}

	if (hex) {
		for (uint32_t i = 0; i < 8; i++) {
			printf("%x ", this->v32[i]);
		}

		return;
	}

	for (uint32_t i = 0; i < 8; i++) {
		printf("%u ", this->v32[i]);
	}
	printf("\n");
}

constexpr inline void uint64x4_t::print(bool binary, bool hex) const {
	/// make sure that only one is defined
	ASSERT(binary + hex < 2);

	if (binary) {
		for (uint32_t i = 0; i < 4; i++) {
			printbinary(this->v64[i]);
		}

		return;
	}

	if (hex) {
		for (uint32_t i = 0; i < 4; i++) {
			printf("%" PRIu64 " ", this->v64[i]);
		}

		return;
	}

	for (uint32_t i = 0; i < 4; i++) {
		printf("%" PRIu64 " ", this->v64[i]);
	}
	printf("\n");
}

#endif//CRYPTANALYSISLIB_SIMD_H
