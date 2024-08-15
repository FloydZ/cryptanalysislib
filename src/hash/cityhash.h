#ifndef CRYPTANALYSISLIB_HASH_CITYHASH_H
#define CRYPTANALYSISLIB_HASH_CITYHASH_H

// SOURCE:https://github.com/google/cityhash/

#ifndef CRYPTANALYSISLIB_HASH_H
#error "do not include this file directly. Use `#inluce <cryptanalysislib/hash/hash.h>`"
#endif

#ifdef __SSE4_2__
#include <immintrin.h>
#endif
#include <cstdint>

#include "helper.h"
#include "binary.h"

#include "memory/memory.h"

using namespace cryptanalysislib;

namespace cryptanalysislib::hash {
	// Some primes between 2^63 and 2^64 for various uses.
	constexpr static uint64_t k0 = 0xc3a5c85c97cb3127ULL;
	constexpr static uint64_t k1 = 0xb492b66fbe98f273ULL;
	constexpr static uint64_t k2 = 0x9ae16a3b2f90404fULL;

	// Magic numbers for 32-bit hashing.  Copied from Murmur3.
	constexpr inline static const uint32_t c1 = 0xcc9e2d51;
	static const uint32_t c2 = 0x1b873593;

	constexpr inline uint64_t bswap64(const uint64_t x) {
		return ((x << 56) & 0xff00000000000000UL) |
		       ((x << 40) & 0x00ff000000000000UL) |
		       ((x << 24) & 0x0000ff0000000000UL) |
		       ((x <<  8) & 0x000000ff00000000UL) |
		       ((x >>  8) & 0x00000000ff000000UL) |
		       ((x >> 24) & 0x0000000000ff0000UL) |
		       ((x >> 40) & 0x000000000000ff00UL) |
		       ((x >> 56) & 0x00000000000000ffUL);
	}

	// Hash 128 input bits down to 64 bits of output.
	// This is intended to be a reasonably good hash function.
	constexpr inline uint64_t Hash128to64(const __uint128_t& x) {
		// Murmur-inspired hashing.
		const uint64_t kMul = 0x9ddfea08eb382d69ULL;
		uint64_t a = ((x) ^ (x >> 64)) * kMul;
		a ^= (a >> 47);
		uint64_t b = ((x >> 64) ^ a) * kMul;
		b ^= (b >> 47);
		b *= kMul;
		return b;
	}
	// A 32-bit to 32-bit integer hash copied from Murmur3.
	constexpr inline static uint32_t fmix(uint32_t h) noexcept {
		h ^= h >> 16;
		h *= 0x85ebca6b;
		h ^= h >> 13;
		h *= 0xc2b2ae35;
		h ^= h >> 16;
		return h;
	}


#undef PERMUTE3
#define PERMUTE3(a, b, c) do { std::swap(a, b); std::swap(a, c); } while (0)

	constexpr static inline uint32_t Mur(uint32_t a, uint32_t h) noexcept {
		// Helper from Murmur3 for combining two 32-bit values.
		a *= c1;
		a = Rotate32(a, 17);
		a *= c2;
		h ^= a;
		h = Rotate32(h, 19);
		return h * 5 + 0xe6546b64;
	}

	constexpr static inline uint32_t Hash32Len13to24(const char *s, size_t len) noexcept {
		uint32_t a = fetch32(s - 4 + (len >> 1));
		uint32_t b = fetch32(s + 4);
		uint32_t c = fetch32(s + len - 8);
		uint32_t d = fetch32(s + (len >> 1));
		uint32_t e = fetch32(s);
		uint32_t f = fetch32(s + len - 4);
		uint32_t h = static_cast<uint32_t>(len);

		return fmix(Mur(f, Mur(e, Mur(d, Mur(c, Mur(b, Mur(a, h)))))));
	}

	constexpr static inline uint32_t Hash32Len0to4(const char *s, size_t len) noexcept {
		uint32_t b = 0;
		uint32_t c = 9;
		for (size_t i = 0; i < len; i++) {
			signed char v = static_cast<signed char>(s[i]);
			b = b * c1 + static_cast<uint32_t>(v);
			c ^= b;
		}
		return fmix(Mur(b, Mur(static_cast<uint32_t>(len), c)));
	}

	constexpr static inline uint32_t Hash32Len5to12(const char *s, size_t len) noexcept {
		uint32_t a = static_cast<uint32_t>(len), b = a * 5, c = 9, d = b;
		a += fetch32(s);
		b += fetch32(s + len - 4);
		c += fetch32(s + ((len >> 1) & 4));
		return fmix(Mur(c, Mur(b, Mur(a, d))));
	}

	constexpr uint32_t CityHash32(const char *s, size_t len) noexcept {
		if (len <= 24) {
			return len <= 12 ? (len <= 4 ? Hash32Len0to4(s, len) : Hash32Len5to12(s, len)) : Hash32Len13to24(s, len);
		}

		// len > 24
		uint32_t h = static_cast<uint32_t>(len), g = c1 * h, f = g;
		uint32_t a0 = Rotate32(fetch32(s + len - 4) * c1, 17) * c2;
		uint32_t a1 = Rotate32(fetch32(s + len - 8) * c1, 17) * c2;
		uint32_t a2 = Rotate32(fetch32(s + len - 16) * c1, 17) * c2;
		uint32_t a3 = Rotate32(fetch32(s + len - 12) * c1, 17) * c2;
		uint32_t a4 = Rotate32(fetch32(s + len - 20) * c1, 17) * c2;
		h ^= a0;
		h = Rotate32(h, 19);
		h = h * 5 + 0xe6546b64;
		h ^= a2;
		h = Rotate32(h, 19);
		h = h * 5 + 0xe6546b64;
		g ^= a1;
		g = Rotate32(g, 19);
		g = g * 5 + 0xe6546b64;
		g ^= a3;
		g = Rotate32(g, 19);
		g = g * 5 + 0xe6546b64;
		f += a4;
		f = Rotate32(f, 19);
		f = f * 5 + 0xe6546b64;
		size_t iters = (len - 1) / 20;
		do {
			uint32_t a0 = Rotate32(fetch32(s) * c1, 17) * c2;
			uint32_t a1 = fetch32(s + 4);
			uint32_t a2 = Rotate32(fetch32(s + 8) * c1, 17) * c2;
			uint32_t a3 = Rotate32(fetch32(s + 12) * c1, 17) * c2;
			uint32_t a4 = fetch32(s + 16);
			h ^= a0;
			h = Rotate32(h, 18);
			h = h * 5 + 0xe6546b64;
			f += a1;
			f = Rotate32(f, 19);
			f = f * c1;
			g += a2;
			g = Rotate32(g, 18);
			g = g * 5 + 0xe6546b64;
			h ^= a3 + a1;
			h = Rotate32(h, 19);
			h = h * 5 + 0xe6546b64;
			g ^= a4;
			g = bswap_32(g) * 5;
			h += a4 * 5;
			h = bswap_32(h);
			f += a0;
			PERMUTE3(f, h, g);
			s += 20;
		} while (--iters != 0);
		g = Rotate32(g, 11) * c1;
		g = Rotate32(g, 17) * c1;
		f = Rotate32(f, 11) * c1;
		f = Rotate32(f, 17) * c1;
		h = Rotate32(h + g, 19);
		h = h * 5 + 0xe6546b64;
		h = Rotate32(h, 17) * c1;
		h = Rotate32(h + f, 19);
		h = h * 5 + 0xe6546b64;
		h = Rotate32(h, 17) * c1;
		return h;
	}

	// Bitwise right rotate.  Normally this will compile to a single
	// instruction, especially if the shift is a manifest constant.
	constexpr static inline uint64_t Rotate(const uint64_t val, const int shift) noexcept {
		// Avoid shifting by 64: doing so yields an undefined result.
		return shift == 0 ? val : ((val >> shift) | (val << (64 - shift)));
	}

	constexpr static inline uint64_t ShiftMix(const uint64_t val) noexcept {
		return val ^ (val >> 47);
	}

	constexpr static inline uint64_t HashLen16(uint64_t u, uint64_t v) noexcept {
		return Hash128to64(__uint128_t(u) ^ (__uint128_t(v) << 64u));
	}

	constexpr static inline uint64_t HashLen16(uint64_t u, uint64_t v, uint64_t mul) noexcept {
		// Murmur-inspired hashing.
		uint64_t a = (u ^ v) * mul;
		a ^= (a >> 47);
		uint64_t b = (v ^ a) * mul;
		b ^= (b >> 47);
		b *= mul;
		return b;
	}

	constexpr static inline uint64_t HashLen0to16(const char *s, size_t len) noexcept {
		if (len >= 8) {
			uint64_t mul = k2 + len * 2;
			uint64_t a = fetch64(s) + k2;
			uint64_t b = fetch64(s + len - 8);
			uint64_t c = Rotate(b, 37) * mul + a;
			uint64_t d = (Rotate(a, 25) + b) * mul;
			return HashLen16(c, d, mul);
		}
		if (len >= 4) {
			uint64_t mul = k2 + len * 2;
			uint64_t a = fetch32(s);
			return HashLen16(len + (a << 3), fetch32(s + len - 4), mul);
		}
		if (len > 0) {
			uint8_t a = static_cast<uint8_t>(s[0]);
			uint8_t b = static_cast<uint8_t>(s[len >> 1]);
			uint8_t c = static_cast<uint8_t>(s[len - 1]);
			uint32_t y = static_cast<uint32_t>(a) + (static_cast<uint32_t>(b) << 8);
			uint32_t z = static_cast<uint32_t>(len) + (static_cast<uint32_t>(c) << 2);
			return ShiftMix(y * k2 ^ z * k0) * k2;
		}
		return k2;
	}

	// This probably works well for 16-byte strings as well, but it may be overkill
	// in that case.
	constexpr static inline uint64_t HashLen17to32(const char *s, size_t len) noexcept {
		uint64_t mul = k2 + len * 2;
		uint64_t a = fetch64(s) * k1;
		uint64_t b = fetch64(s + 8);
		uint64_t c = fetch64(s + len - 8) * mul;
		uint64_t d = fetch64(s + len - 16) * k2;
		return HashLen16(Rotate(a + b, 43) + Rotate(c, 30) + d,
		                 a + Rotate(b + k2, 18) + c, mul);
	}

	// Return a 16-byte hash for 48 bytes.  Quick and dirty.
	// Callers do best to use "random-looking" values for a and b.
	constexpr static inline std::pair<uint64_t, uint64_t> WeakHashLen32WithSeeds(
	        const uint64_t w, const uint64_t x, const uint64_t y, const uint64_t z, uint64_t a, uint64_t b) noexcept {
		a += w;
		b = Rotate(b + a + z, 21);
		uint64_t c = a;
		a += x;
		a += y;
		b += Rotate(a, 44);
		return std::make_pair(a + z, b + c);
	}

	// Return a 16-byte hash for s[0] ... s[31], a, and b.  Quick and dirty.
	constexpr static inline std::pair<uint64_t, uint64_t> WeakHashLen32WithSeeds(
	        const char *s, const uint64_t a, const uint64_t b)  noexcept {
		return WeakHashLen32WithSeeds(fetch64(s),
		                              fetch64(s + 8),
		                              fetch64(s + 16),
		                              fetch64(s + 24),
		                              a,
		                              b);
	}

	// Return an 8-byte hash for 33 to 64 bytes.
	constexpr static uint64_t HashLen33to64(const char *s, size_t len) noexcept {
		uint64_t mul = k2 + len * 2;
		uint64_t a = fetch64(s) * k2;
		uint64_t b = fetch64(s + 8);
		uint64_t c = fetch64(s + len - 24);
		uint64_t d = fetch64(s + len - 32);
		uint64_t e = fetch64(s + 16) * k2;
		uint64_t f = fetch64(s + 24) * 9;
		uint64_t g = fetch64(s + len - 8);
		uint64_t h = fetch64(s + len - 16) * mul;
		uint64_t u = Rotate(a + g, 43) + (Rotate(b, 30) + c) * 9;
		uint64_t v = ((a + g) ^ d) + f + 1;
		uint64_t w = bswap64((u + v) * mul) + h;
		uint64_t x = Rotate(e + f, 42) + c;
		uint64_t y = (bswap64((v + w) * mul) + g) * mul;
		uint64_t z = e + f + c;
		a = bswap64((x + z) * mul + y) + b;
		b = ShiftMix((z + a) * mul + d + h) * mul;
		return b + x;
	}

	constexpr uint64_t CityHash64(const char *s, const size_t len) noexcept {
		if (len <= 32) {
			if (len <= 16) {
				return HashLen0to16(s, len);
			} else {
				return HashLen17to32(s, len);
			}
		} else if (len <= 64) {
			return HashLen33to64(s, len);
		}

		// For strings over 64 bytes we hash the end first, and then as we
		// loop we keep 56 bytes of state: v, w, x, y, and z.
		uint64_t x = fetch64(s + len - 40);
		uint64_t y = fetch64(s + len - 16) + fetch64(s + len - 56);
		uint64_t z = HashLen16(fetch64(s + len - 48) + len, fetch64(s + len - 24));
		std::pair<uint64_t, uint64_t> v = WeakHashLen32WithSeeds(s + len - 64, len, z);
		std::pair<uint64_t, uint64_t> w = WeakHashLen32WithSeeds(s + len - 32, y + k1, x);
		x = x * k1 + fetch64(s);

		// Decrease len to the nearest multiple of 64, and operate on 64-byte chunks.
		size_t len_ = (len - 1) & ~static_cast<size_t>(63);
		do {
			x = Rotate(x + y + v.first + fetch64(s + 8), 37) * k1;
			y = Rotate(y + v.second + fetch64(s + 48), 42) * k1;
			x ^= w.second;
			y += v.first + fetch64(s + 40);
			z = Rotate(z + w.first, 33) * k1;
			v = WeakHashLen32WithSeeds(s, v.second * k1, x + w.first);
			w = WeakHashLen32WithSeeds(s + 32, z + w.second, y + fetch64(s + 16));
			std::swap(z, x);
			s += 64;
			len_ -= 64;
		} while (len_ != 0);
		return HashLen16(HashLen16(v.first, w.first) + ShiftMix(y) * k1 + z,
		                 HashLen16(v.second, w.second) + x);
	}

	constexpr inline uint64_t CityHash64WithSeeds(const char *s,
	                                              const size_t len,
	                                              const uint64_t seed0,
	                                              const uint64_t seed1) noexcept {
		return HashLen16(CityHash64(s, len) - seed0, seed1);
	}

	constexpr inline uint64_t CityHash64WithSeed(const char *s,
	                                             const size_t len,
	                                             const uint64_t seed) noexcept {
		return CityHash64WithSeeds(s, len, k2, seed);
	}


	// A subroutine for CityHash128().  Returns a decent 128-bit hash for strings
	// of any length representable in signed long.  Based on City and Murmur.
	constexpr static __uint128_t CityMurmur(const char *s,
	                                        size_t len,
	                                        const __uint128_t seed) noexcept {
		uint64_t a = seed;
		uint64_t b = seed >> 64u;
		uint64_t c = 0;
		uint64_t d = 0;
		if (len <= 16) {
			a = ShiftMix(a * k1) * k1;
			c = b * k1 + HashLen0to16(s, len);
			d = ShiftMix(a + (len >= 8 ? fetch64(s) : c));
		} else {
			c = HashLen16(fetch64(s + len - 8) + k1, a);
			d = HashLen16(b + len, c + fetch64(s + len - 16));
			a += d;
			// len > 16 here, so do...while is safe
			do {
				a ^= ShiftMix(fetch64(s) * k1) * k1;
				a *= k1;
				b ^= a;
				c ^= ShiftMix(fetch64(s + 8) * k1) * k1;
				c *= k1;
				d ^= c;
				s += 16;
				len -= 16;
			} while (len > 16);
		}
		a = HashLen16(a, c);
		b = HashLen16(d, b);
		return (__uint128_t(a ^ b)) ^ (__uint128_t(HashLen16(b, a)) << 64u);
	}

	constexpr inline __uint128_t CityHash128WithSeed(const char *s, size_t len, __uint128_t seed) noexcept {
		if (len < 128) {
			return CityMurmur(s, len, seed);
		}

		// We expect len >= 128 to be the common case.  Keep 56 bytes of state:
		// v, w, x, y, and z.
		std::pair<uint64_t, uint64_t> v, w;
		uint64_t x = seed;
		uint64_t y = seed >> 64;
		uint64_t z = len * k1;
		v.first = Rotate(y ^ k1, 49) * k1 + fetch64(s);
		v.second = Rotate(v.first, 42) * k1 + fetch64(s + 8);
		w.first = Rotate(y + z, 35) * k1 + x;
		w.second = Rotate(x + fetch64(s + 88), 53) * k1;

		// This is the same inner loop as CityHash64(), manually unrolled.
		do {
			x = Rotate(x + y + v.first + fetch64(s + 8), 37) * k1;
			y = Rotate(y + v.second + fetch64(s + 48), 42) * k1;
			x ^= w.second;
			y += v.first + fetch64(s + 40);
			z = Rotate(z + w.first, 33) * k1;
			v = WeakHashLen32WithSeeds(s, v.second * k1, x + w.first);
			w = WeakHashLen32WithSeeds(s + 32, z + w.second, y + fetch64(s + 16));
			std::swap(z, x);
			s += 64;
			x = Rotate(x + y + v.first + fetch64(s + 8), 37) * k1;
			y = Rotate(y + v.second + fetch64(s + 48), 42) * k1;
			x ^= w.second;
			y += v.first + fetch64(s + 40);
			z = Rotate(z + w.first, 33) * k1;
			v = WeakHashLen32WithSeeds(s, v.second * k1, x + w.first);
			w = WeakHashLen32WithSeeds(s + 32, z + w.second, y + fetch64(s + 16));
			std::swap(z, x);
			s += 64;
			len -= 128;
		} while (likely(len >= 128));
		x += Rotate(v.first + z, 49) * k0;
		y = y * k0 + Rotate(w.second, 37);
		z = z * k0 + Rotate(w.first, 27);
		w.first *= 9;
		v.first *= k0;
		// If 0 < len < 128, hash up to 4 chunks of 32 bytes each from the end of s.
		for (size_t tail_done = 0; tail_done < len;) {
			tail_done += 32;
			y = Rotate(x + y, 42) * k0 + v.second;
			w.first += fetch64(s + len - tail_done + 16);
			x = x * k0 + w.first;
			z += w.second + fetch64(s + len - tail_done);
			w.second += v.first;
			v = WeakHashLen32WithSeeds(s + len - tail_done, v.first + z, v.second);
			v.first *= k0;
		}
		// At this point our 56 bytes of state should contain more than
		// enough information for a strong 128-bit hash.  We use two
		// different 56-byte-to-8-byte hashes to get a 16-byte final result.
		x = HashLen16(x, v.first);
		y = HashLen16(y + z, w.first);
		return (__uint128_t(HashLen16(x + v.second, w.second) + y)) ^
		       (__uint128_t(HashLen16(x + w.second, y + v.second)) << 64u);
	}

	constexpr __uint128_t CityHash128(const char *s,
	                                  const size_t len) noexcept {
		return len >= 16 ? CityHash128WithSeed(s + 16, len - 16,
		                     (__uint128_t(fetch64(s))) ^ (__uint128_t(fetch64(s + 8) + k0) << 64))
		                 : CityHash128WithSeed(s, len, (__uint128_t(k0)) ^ (__uint128_t(k1) << 64u));
	}

#ifdef __SSE4_2__

	// Requires len >= 240.
	constexpr static void CityHashCrc256Long(const char *s, size_t len,
	                               const uint32_t seed,
	                                         uint64_t *result) {
		uint64_t a = fetch64(s + 56) + k0;
		uint64_t b = fetch64(s + 96) + k0;
		uint64_t c = result[0] = HashLen16(b, len);
		uint64_t d = result[1] = fetch64(s + 120) * k0 + len;
		uint64_t e = fetch64(s + 184) + seed;
		uint64_t f = 0;
		uint64_t g = 0;
		uint64_t h = c + d;
		uint64_t x = seed;
		uint64_t y = 0;
		uint64_t z = 0;

		// 240 bytes of input per iter.
		size_t iters = len / 240;
		len -= iters * 240;
		do {
#undef CHUNK
#define CHUNK(r)                                \
    PERMUTE3(x, z, y);                          \
    b += fetch64(s);                            \
    c += fetch64(s + 8);                        \
    d += fetch64(s + 16);                       \
    e += fetch64(s + 24);                       \
    f += fetch64(s + 32);                       \
    a += b;                                     \
    h += f;                                     \
    b += c;                                     \
    f += d;                                     \
    g += e;                                     \
    e += z;                                     \
    g += x;                                     \
    z = __builtin_ia32_crc32di(z, b + g);       \
    y = __builtin_ia32_crc32di(y, e + h);       \
    x = __builtin_ia32_crc32di(x, f + a);       \
    e = Rotate(e, r);                           \
    c += e;                                     \
    s += 40

			CHUNK(0);
			PERMUTE3(a, h, c);
			CHUNK(33);
			PERMUTE3(a, h, f);
			CHUNK(0);
			PERMUTE3(b, h, f);
			CHUNK(42);
			PERMUTE3(b, h, d);
			CHUNK(0);
			PERMUTE3(b, h, e);
			CHUNK(33);
			PERMUTE3(a, h, e);
		} while (--iters > 0);

		while (len >= 40) {
			CHUNK(29);
			e ^= Rotate(a, 20);
			h += Rotate(b, 30);
			g ^= Rotate(c, 40);
			f += Rotate(d, 34);
			PERMUTE3(c, h, g);
			len -= 40;
		}
		if (len > 0) {
			s = s + len - 40;
			CHUNK(33);
			e ^= Rotate(a, 43);
			h += Rotate(b, 42);
			g ^= Rotate(c, 41);
			f += Rotate(d, 40);
		}
		result[0] ^= h;
		result[1] ^= g;
		g += h;
		a = HashLen16(a, g + z);
		x += y << 32;
		b += x;
		c = HashLen16(c, z) + h;
		d = HashLen16(d, e + result[0]);
		g += e;
		h += HashLen16(x, f);
		e = HashLen16(a, d) + g;
		z = HashLen16(b, c) + a;
		y = HashLen16(g, h) + c;
		result[0] = e + z + y + x;
		a = ShiftMix((a + y) * k0) * k0 + b;
		result[1] += a + result[0];
		a = ShiftMix(a * k0) * k0 + c;
		result[2] = a + result[1];
		a = ShiftMix((a + e) * k0) * k0;
		result[3] = a + result[2];
	}

	// Requires len < 240.
	constexpr static void CityHashCrc256Short(const char *s,
	                                          const size_t len,
	                                          uint64_t *result) noexcept{
		char buf[240];
		cryptanalysislib::template memcpy<uint8_t>((uint8_t *)buf, (uint8_t *)s, len);
		cryptanalysislib::template memset<uint8_t>((uint8_t *)buf + len, 0, 240 - len);
		CityHashCrc256Long(buf, 240, ~static_cast<uint32_t>(len), result);
	}

	constexpr void CityHashCrc256(const char *s,
	                    const size_t len,
	                    uint64_t *result) noexcept {
		if (likely(len >= 240)) {
			CityHashCrc256Long(s, len, 0, result);
		} else {
			CityHashCrc256Short(s, len, result);
		}
	}

	constexpr __uint128_t CityHashCrc128WithSeed(const char *s,
	                                             const size_t len,
	                                             __uint128_t seed) noexcept {
		if (len <= 900) {
			return CityHash128WithSeed(s, len, seed);
		} else {
			uint64_t result[4];
			CityHashCrc256(s, len, result);
			uint64_t u = (seed >> 64u) + result[0];
			uint64_t v = (seed) + result[1];
			return (__uint128_t(HashLen16(u, v + result[2]))) ^
			       (__uint128_t(HashLen16(Rotate(v, 32), u * k0 + result[3])) << 64u);
		}
	}

	constexpr __uint128_t CityHashCrc128(const char *s,
	                                     const size_t len) noexcept {
		if (len <= 900) {
			return CityHash128(s, len);
		} else {
			uint64_t result[4];
			CityHashCrc256(s, len, result);
			return (__uint128_t(result[2])) ^ (__uint128_t(result[3]) << 64u);
		}
	}
#endif// __SSE4_2__

#undef PERMUTE3
}
#endif//CRYPTANALYSISLIB_CITYHASH_H
