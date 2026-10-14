#ifndef CRYPTANALYSISLIB_SHA3_H
#define CRYPTANALYSISLIB_SHA3_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "crypto/sha3/keccak.h"

/// SHA-3 and SHAKE (FIPS 202): the Keccak[1600] sponge, usable at compile
/// time.
/// Usage:
/// \code
/// 	constexpr auto d = cryptanalysislib::sha3_256("abc");   // std::array<uint8_t, 32>
/// 	cryptanalysislib::shake128_t x; x.update(p, n);
/// 	x.squeeze(out, 1000); x.squeeze(out2, 50);               // any output length
/// \endcode
namespace cryptanalysislib {
	/// Keccak sponge with `rate` bytes per block and the domain separation
	/// `suffix` (incl. the first bit of the padding): 0x06 SHA-3, 0x1f SHAKE
	template<const size_t rate, const uint8_t suffix>
	class keccak_sponge {
		static_assert((rate > 0) && (rate < 200) && (rate % 8 == 0));

		hash::keccak::state_1600 state{};
		/// absorbing: bytes of the current block, squeezing: bytes read
		size_t pos = 0;
		bool squeezing = false;

		constexpr void xor_byte(const size_t i, const uint8_t b) noexcept {
			state[i / 8] ^= uint64_t(b) << (8u * (i % 8));
		}

		[[nodiscard]] constexpr uint8_t get_byte(const size_t i) const noexcept {
			return uint8_t(state[i / 8] >> (8u * (i % 8)));
		}

		/// pads the message and switches to squeezing
		constexpr void finish() noexcept {
			xor_byte(pos, suffix);
			xor_byte(rate - 1, 0x80);
			hash::keccak::keccak_f(state);
			pos = 0;
			squeezing = true;
		}

	public:
		constexpr keccak_sponge() noexcept = default;

		/// absorbs `n` bytes. Must not be called after `squeeze`.
		constexpr void update(const uint8_t *data, const size_t n) noexcept {
			for (size_t i = 0; i < n; ++i) {
				xor_byte(pos++, data[i]);
				if (pos == rate) {
					hash::keccak::keccak_f(state);
					pos = 0;
				}
			}
		}

		/// absorbs the chars of `s`
		constexpr void update(const std::string_view s) noexcept {
			for (const char c : s) {
				xor_byte(pos++, uint8_t(c));
				if (pos == rate) {
					hash::keccak::keccak_f(state);
					pos = 0;
				}
			}
		}

		/// writes the next `n` output bytes; can be called several times
		constexpr void squeeze(uint8_t *out, const size_t n) noexcept {
			if (!squeezing) { finish(); }
			for (size_t i = 0; i < n; ++i) {
				if (pos == rate) {
					hash::keccak::keccak_f(state);
					pos = 0;
				}
				out[i] = get_byte(pos++);
			}
		}
	};

	/// SHA-3 hash state
	/// \tparam bits digest length in bits: 224, 256, 384 or 512
	template<const uint32_t bits>
	class sha3 : public keccak_sponge<200 - 2 * (bits / 8), 0x06> {
		static_assert(bits == 224 || bits == 256 || bits == 384 || bits == 512);

	public:
		constexpr static size_t digest_len = bits / 8;

		/// \return the digest. The state must not be updated afterwards.
		[[nodiscard]] constexpr std::array<uint8_t, digest_len> digest() noexcept {
			std::array<uint8_t, digest_len> ret{};
			this->squeeze(ret.data(), digest_len);
			return ret;
		}
	};

	/// SHAKE128 (rate 168 bytes) / SHAKE256 (rate 136 bytes), extendable output
	using shake128_t = keccak_sponge<168, 0x1f>;
	using shake256_t = keccak_sponge<136, 0x1f>;
	using sha3_224_t = sha3<224>;
	using sha3_256_t = sha3<256>;
	using sha3_384_t = sha3<384>;
	using sha3_512_t = sha3<512>;

	/// \return SHA-3 digest of the `n` bytes at `data`
	template<const uint32_t bits>
	[[nodiscard]] constexpr std::array<uint8_t, bits / 8> sha3_hash(const uint8_t *data, const size_t n) noexcept {
		sha3<bits> s;
		s.update(data, n);
		return s.digest();
	}

	/// \return SHA-3 digest of the chars of `str`
	template<const uint32_t bits>
	[[nodiscard]] constexpr std::array<uint8_t, bits / 8> sha3_hash(const std::string_view str) noexcept {
		sha3<bits> s;
		s.update(str);
		return s.digest();
	}

	[[nodiscard]] constexpr std::array<uint8_t, 28> sha3_224(const uint8_t *data, const size_t n) noexcept { return sha3_hash<224>(data, n); }
	[[nodiscard]] constexpr std::array<uint8_t, 32> sha3_256(const uint8_t *data, const size_t n) noexcept { return sha3_hash<256>(data, n); }
	[[nodiscard]] constexpr std::array<uint8_t, 48> sha3_384(const uint8_t *data, const size_t n) noexcept { return sha3_hash<384>(data, n); }
	[[nodiscard]] constexpr std::array<uint8_t, 64> sha3_512(const uint8_t *data, const size_t n) noexcept { return sha3_hash<512>(data, n); }
	[[nodiscard]] constexpr std::array<uint8_t, 28> sha3_224(const std::string_view s) noexcept { return sha3_hash<224>(s); }
	[[nodiscard]] constexpr std::array<uint8_t, 32> sha3_256(const std::string_view s) noexcept { return sha3_hash<256>(s); }
	[[nodiscard]] constexpr std::array<uint8_t, 48> sha3_384(const std::string_view s) noexcept { return sha3_hash<384>(s); }
	[[nodiscard]] constexpr std::array<uint8_t, 64> sha3_512(const std::string_view s) noexcept { return sha3_hash<512>(s); }

	/// writes `outlen` bytes of SHAKE128(`data`) to `out`
	constexpr void shake128(uint8_t *out, const size_t outlen, const uint8_t *data, const size_t n) noexcept {
		shake128_t s;
		s.update(data, n);
		s.squeeze(out, outlen);
	}

	/// writes `outlen` bytes of SHAKE256(`data`) to `out`
	constexpr void shake256(uint8_t *out, const size_t outlen, const uint8_t *data, const size_t n) noexcept {
		shake256_t s;
		s.update(data, n);
		s.squeeze(out, outlen);
	}
} // end namespace cryptanalysislib
#endif//CRYPTANALYSISLIB_SHA3_H
