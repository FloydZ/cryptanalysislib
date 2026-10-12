#ifndef CRYPTANALYSISLIB_CRYPTO_MD5_H
#define CRYPTANALYSISLIB_CRYPTO_MD5_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

/// MD5 (RFC 1321), usable at compile time.
/// NOTE: MD5 is broken as a cryptographic hash (collisions), use it only for
/// 	checksums/compatibility or as a target in cryptanalysis.
namespace cryptanalysislib::crypto {
	namespace internal {
		/// per round shift amounts
		constexpr static uint32_t md5_s[64] = {
			7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
			5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
			4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
			6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
		};

		/// K[i] = floor(2**32 * |sin(i + 1)|)
		constexpr static uint32_t md5_k[64] = {
			0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
			0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
			0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
			0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
			0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
			0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
			0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
			0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
		};

		[[nodiscard]] constexpr inline uint32_t md5_rotl(const uint32_t x, const uint32_t c) noexcept {
			return (x << c) | (x >> (32u - c));
		}

		/// processes one 64 byte block (little endian words)
		constexpr inline void md5_block(uint32_t h[4], const uint8_t block[64]) noexcept {
			uint32_t m[16];
			for (uint32_t i = 0; i < 16; ++i) {
				m[i] = uint32_t(block[4*i]) | (uint32_t(block[4*i + 1]) << 8u) |
				       (uint32_t(block[4*i + 2]) << 16u) | (uint32_t(block[4*i + 3]) << 24u);
			}

			uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
			for (uint32_t i = 0; i < 64; ++i) {
				uint32_t f, g;
				if (i < 16)      { f = (b & c) | (~b & d); g = i; }
				else if (i < 32) { f = (d & b) | (~d & c); g = (5u*i + 1u) % 16u; }
				else if (i < 48) { f = b ^ c ^ d;          g = (3u*i + 5u) % 16u; }
				else             { f = c ^ (b | ~d);       g = (7u*i) % 16u; }

				f = f + a + md5_k[i] + m[g];
				a = d;
				d = c;
				c = b;
				b = b + md5_rotl(f, md5_s[i]);
			}

			h[0] += a; h[1] += b; h[2] += c; h[3] += d;
		}

		/// \tparam C `uint8_t` or `char` (for compile time strings)
		/// \param data message
		/// \param len number of bytes
		template<typename C>
		[[nodiscard]] constexpr inline std::array<uint8_t, 16> md5(const C *data,
		                                                           const size_t len) noexcept {
			uint32_t h[4] = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};

			uint8_t block[64] = {};
			size_t i = 0;
			for (; i + 64 <= len; i += 64) {
				for (size_t j = 0; j < 64; ++j) { block[j] = uint8_t(data[i + j]); }
				md5_block(h, block);
			}

			// padding: 0x80, zeros, 64 bit little endian bit length
			uint8_t last[128] = {};
			const size_t rest = len - i;
			for (size_t j = 0; j < rest; ++j) { last[j] = uint8_t(data[i + j]); }
			last[rest] = 0x80;
			const size_t total = (rest < 56) ? 64 : 128;
			const uint64_t bits = uint64_t(len) * 8u;
			for (uint32_t j = 0; j < 8; ++j) {
				last[total - 8 + j] = uint8_t(bits >> (8u * j));
			}

			md5_block(h, last);
			if (total == 128) { md5_block(h, last + 64); }

			std::array<uint8_t, 16> out{};
			for (uint32_t j = 0; j < 4; ++j) {
				for (uint32_t k = 0; k < 4; ++k) {
					out[4*j + k] = uint8_t(h[j] >> (8u * k));
				}
			}
			return out;
		}
	} // end namespace internal

	/// \param data[in]: message
	/// \param len[in]: number of bytes
	/// \return the 16 byte MD5 digest
	[[nodiscard]] constexpr inline std::array<uint8_t, 16> md5(const uint8_t *data,
	                                                           const size_t len) noexcept {
		return internal::md5(data, len);
	}

	/// \param s[in]: message
	/// \return the 16 byte MD5 digest
	/// NOTE: usable at compile time, e.g. `static_assert(md5("abc")[0] == 0x90)`
	[[nodiscard]] constexpr inline std::array<uint8_t, 16> md5(const std::string_view s) noexcept {
		return internal::md5(s.data(), s.size());
	}
} // end namespace cryptanalysislib::crypto

#endif // CRYPTANALYSISLIB_CRYPTO_MD5_H
