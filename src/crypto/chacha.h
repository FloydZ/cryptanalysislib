#ifndef CRYPTANALYSISLIB_CRYPTO_CHACHA_H
#define CRYPTANALYSISLIB_CRYPTO_CHACHA_H

#include <array>
#include <cstddef>
#include <cstdint>

/// ChaCha stream cipher (RFC 8439: 256 bit key, 96 bit nonce, 32 bit block
/// counter), usable at compile time.
/// Usage:
/// \code
/// 	cryptanalysislib::chacha20 c(key, nonce, 1);
/// 	c.crypt(out, in, n);   // encrypts and decrypts
/// \endcode
namespace cryptanalysislib {
	namespace internal {
		[[nodiscard]] constexpr inline uint32_t chacha_rotl(const uint32_t x, const uint32_t n) noexcept {
			return (x << n) | (x >> (32u - n));
		}

		constexpr inline void chacha_quarter_round(uint32_t *s,
		                                           const uint32_t a, const uint32_t b,
		                                           const uint32_t c, const uint32_t d) noexcept {
			s[a] += s[b]; s[d] ^= s[a]; s[d] = chacha_rotl(s[d], 16);
			s[c] += s[d]; s[b] ^= s[c]; s[b] = chacha_rotl(s[b], 12);
			s[a] += s[b]; s[d] ^= s[a]; s[d] = chacha_rotl(s[d], 8);
			s[c] += s[d]; s[b] ^= s[c]; s[b] = chacha_rotl(s[b], 7);
		}

		[[nodiscard]] constexpr inline uint32_t chacha_load32(const uint8_t *p) noexcept {
			return uint32_t(p[0]) | (uint32_t(p[1]) << 8u) | (uint32_t(p[2]) << 16u) | (uint32_t(p[3]) << 24u);
		}
	} // end namespace internal

	/// \tparam rounds number of rounds: 8, 12 or 20
	template<const uint32_t rounds = 20>
	class chacha {
		static_assert(rounds % 2 == 0);

		/// constants, key, counter, nonce
		std::array<uint32_t, 16> input{};
		/// current key stream block
		std::array<uint8_t, 64> stream{};
		/// used bytes of `stream`, 64: a new block is needed
		uint32_t used = 64;

	public:
		/// \param key[in]: 32 bytes
		/// \param nonce[in]: 12 bytes
		/// \param counter[in]: block counter of the first block
		constexpr chacha(const uint8_t *key,
		                 const uint8_t *nonce,
		                 const uint32_t counter = 0) noexcept {
			input[0] = 0x61707865; input[1] = 0x3320646e;
			input[2] = 0x79622d32; input[3] = 0x6b206574;
			for (uint32_t i = 0; i < 8; ++i) {
				input[4 + i] = internal::chacha_load32(key + 4 * i);
			}
			input[12] = counter;
			for (uint32_t i = 0; i < 3; ++i) {
				input[13 + i] = internal::chacha_load32(nonce + 4 * i);
			}
		}

		/// \return the key stream block of the current counter (the block
		/// 	function), and increments the counter
		[[nodiscard]] constexpr std::array<uint8_t, 64> block() noexcept {
			std::array<uint32_t, 16> x = input;
			for (uint32_t i = 0; i < rounds; i += 2) {
				internal::chacha_quarter_round(x.data(), 0, 4, 8, 12);
				internal::chacha_quarter_round(x.data(), 1, 5, 9, 13);
				internal::chacha_quarter_round(x.data(), 2, 6, 10, 14);
				internal::chacha_quarter_round(x.data(), 3, 7, 11, 15);
				internal::chacha_quarter_round(x.data(), 0, 5, 10, 15);
				internal::chacha_quarter_round(x.data(), 1, 6, 11, 12);
				internal::chacha_quarter_round(x.data(), 2, 7, 8, 13);
				internal::chacha_quarter_round(x.data(), 3, 4, 9, 14);
			}

			std::array<uint8_t, 64> out{};
			for (uint32_t i = 0; i < 16; ++i) {
				const uint32_t v = x[i] + input[i];
				out[4 * i + 0] = uint8_t(v);
				out[4 * i + 1] = uint8_t(v >> 8u);
				out[4 * i + 2] = uint8_t(v >> 16u);
				out[4 * i + 3] = uint8_t(v >> 24u);
			}

			input[12] += 1;
			return out;
		}

		/// xors the next `n` key stream bytes to `in` (encryption and
		/// decryption are the same); `out` may equal `in`
		constexpr void crypt(uint8_t *out,
		                     const uint8_t *in,
		                     const size_t n) noexcept {
			for (size_t i = 0; i < n; ++i) {
				if (used == 64) {
					stream = block();
					used = 0;
				}
				out[i] = in[i] ^ stream[used++];
			}
		}

		/// writes the next `n` key stream bytes to `out`
		constexpr void keystream(uint8_t *out, const size_t n) noexcept {
			for (size_t i = 0; i < n; ++i) {
				if (used == 64) {
					stream = block();
					used = 0;
				}
				out[i] = stream[used++];
			}
		}
	};

	using chacha8 = chacha<8>;
	using chacha12 = chacha<12>;
	using chacha20 = chacha<20>;
} // end namespace cryptanalysislib
#endif
