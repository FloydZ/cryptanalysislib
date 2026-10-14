#ifndef CRYPTANALYSISLIB_SHA2_H
#define CRYPTANALYSISLIB_SHA2_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

/// SHA-2 (FIPS 180-4): SHA-224, SHA-256, SHA-384, SHA-512, usable at
/// compile time.
/// Usage:
/// \code
/// 	constexpr auto d = cryptanalysislib::sha256("abc");      // std::array<uint8_t, 32>
/// 	cryptanalysislib::sha512_t h; h.update(p, n); h.update(q, m);
/// 	const auto e = h.digest();
/// \endcode
namespace cryptanalysislib {
	namespace internal {
		/// constexpr rotation (the intrinsics of `algorithm/rotate.h` are not)
		template<typename T>
		[[nodiscard]] constexpr inline T sha2_rotr(const T x, const uint32_t n) noexcept {
			return (x >> n) | (x << (8u * sizeof(T) - n));
		}

		constexpr static std::array<uint32_t, 64> sha256_k = {
			0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
			0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
			0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
			0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
			0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
			0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
			0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
			0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
		};

		constexpr static std::array<uint64_t, 80> sha512_k = {
			0x428a2f98d728ae22, 0x7137449123ef65cd, 0xb5c0fbcfec4d3b2f, 0xe9b5dba58189dbbc, 0x3956c25bf348b538,
			0x59f111f1b605d019, 0x923f82a4af194f9b, 0xab1c5ed5da6d8118, 0xd807aa98a3030242, 0x12835b0145706fbe,
			0x243185be4ee4b28c, 0x550c7dc3d5ffb4e2, 0x72be5d74f27b896f, 0x80deb1fe3b1696b1, 0x9bdc06a725c71235,
			0xc19bf174cf692694, 0xe49b69c19ef14ad2, 0xefbe4786384f25e3, 0x0fc19dc68b8cd5b5, 0x240ca1cc77ac9c65,
			0x2de92c6f592b0275, 0x4a7484aa6ea6e483, 0x5cb0a9dcbd41fbd4, 0x76f988da831153b5, 0x983e5152ee66dfab,
			0xa831c66d2db43210, 0xb00327c898fb213f, 0xbf597fc7beef0ee4, 0xc6e00bf33da88fc2, 0xd5a79147930aa725,
			0x06ca6351e003826f, 0x142929670a0e6e70, 0x27b70a8546d22ffc, 0x2e1b21385c26c926, 0x4d2c6dfc5ac42aed,
			0x53380d139d95b3df, 0x650a73548baf63de, 0x766a0abb3c77b2a8, 0x81c2c92e47edaee6, 0x92722c851482353b,
			0xa2bfe8a14cf10364, 0xa81a664bbc423001, 0xc24b8b70d0f89791, 0xc76c51a30654be30, 0xd192e819d6ef5218,
			0xd69906245565a910, 0xf40e35855771202a, 0x106aa07032bbd1b8, 0x19a4c116b8d2d0c8, 0x1e376c085141ab53,
			0x2748774cdf8eeb99, 0x34b0bcb5e19b48a8, 0x391c0cb3c5c95a63, 0x4ed8aa4ae3418acb, 0x5b9cca4f7763e373,
			0x682e6ff3d6b2b8a3, 0x748f82ee5defb2fc, 0x78a5636f43172f60, 0x84c87814a1f0ab72, 0x8cc702081a6439ec,
			0x90befffa23631e28, 0xa4506cebde82bde9, 0xbef9a3f7b2c67915, 0xc67178f2e372532b, 0xca273eceea26619c,
			0xd186b8c721c0c207, 0xeada7dd6cde0eb1e, 0xf57d4f7fee6ed178, 0x06f067aa72176fba, 0x0a637dc5a2c898a6,
			0x113f9804bef90dae, 0x1b710b35131c471b, 0x28db77f523047d84, 0x32caab7b40c72493, 0x3c9ebe0a15c9bebc,
			0x431d67c49c100d4c, 0x4cc5d4becb3e42b6, 0x597f299cfc657e2a, 0x5fcb6fab3ad6faec, 0x6c44198c4a475817,
		};

		/// parameters of the SHA-2 variant with a `bits` bit digest
		template<const uint32_t bits>
		struct sha2_traits;

		template<> struct sha2_traits<224> {
			using word = uint32_t;
			constexpr static std::array<word, 8> iv = {
				0xc1059ed8, 0x367cd507, 0x3070dd17, 0xf70e5939, 0xffc00b31, 0x68581511, 0x64f98fa7, 0xbefa4fa4};
		};
		template<> struct sha2_traits<256> {
			using word = uint32_t;
			constexpr static std::array<word, 8> iv = {
				0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
		};
		template<> struct sha2_traits<384> {
			using word = uint64_t;
			constexpr static std::array<word, 8> iv = {
				0xcbbb9d5dc1059ed8, 0x629a292a367cd507, 0x9159015a3070dd17, 0x152fecd8f70e5939,
				0x67332667ffc00b31, 0x8eb44a8768581511, 0xdb0c2e0d64f98fa7, 0x47b5481dbefa4fa4};
		};
		template<> struct sha2_traits<512> {
			using word = uint64_t;
			constexpr static std::array<word, 8> iv = {
				0x6a09e667f3bcc908, 0xbb67ae8584caa73b, 0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
				0x510e527fade682d1, 0x9b05688c2b3e6c1f, 0x1f83d9abfb41bd6b, 0x5be0cd19137e2179};
		};
	} // end namespace internal

	/// SHA-2 hash state, data can be added via several calls of `update`
	/// \tparam bits digest length in bits: 224, 256, 384 or 512
	template<const uint32_t bits>
	class sha2 {
		using traits = internal::sha2_traits<bits>;
		using word = typename traits::word;
		constexpr static bool is64 = sizeof(word) == 8;
		/// message block length in bytes
		constexpr static size_t block_len = 16 * sizeof(word);
		constexpr static uint32_t rounds = is64 ? 80 : 64;

		std::array<word, 8> h = traits::iv;
		std::array<uint8_t, block_len> buf{};
		/// bytes in `buf`
		size_t buf_len = 0;
		/// total number of bytes
		uint64_t total = 0;

		constexpr void compress(const uint8_t *block) noexcept {
			using internal::sha2_rotr;
			word w[80]{};
			for (uint32_t i = 0; i < 16; ++i) {
				word x = 0;
				for (uint32_t j = 0; j < sizeof(word); ++j) {
					x = word(x << 8u) | block[i * sizeof(word) + j];
				}
				w[i] = x;
			}

			for (uint32_t i = 16; i < rounds; ++i) {
				word s0, s1;
				if constexpr (is64) {
					s0 = sha2_rotr(w[i - 15], 1) ^ sha2_rotr(w[i - 15], 8) ^ (w[i - 15] >> 7u);
					s1 = sha2_rotr(w[i - 2], 19) ^ sha2_rotr(w[i - 2], 61) ^ (w[i - 2] >> 6u);
				} else {
					s0 = sha2_rotr(w[i - 15], 7) ^ sha2_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3u);
					s1 = sha2_rotr(w[i - 2], 17) ^ sha2_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10u);
				}
				w[i] = w[i - 16] + s0 + w[i - 7] + s1;
			}

			word a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
			for (uint32_t i = 0; i < rounds; ++i) {
				word S0, S1, k;
				if constexpr (is64) {
					S1 = sha2_rotr(e, 14) ^ sha2_rotr(e, 18) ^ sha2_rotr(e, 41);
					S0 = sha2_rotr(a, 28) ^ sha2_rotr(a, 34) ^ sha2_rotr(a, 39);
					k = internal::sha512_k[i];
				} else {
					S1 = sha2_rotr(e, 6) ^ sha2_rotr(e, 11) ^ sha2_rotr(e, 25);
					S0 = sha2_rotr(a, 2) ^ sha2_rotr(a, 13) ^ sha2_rotr(a, 22);
					k = internal::sha256_k[i];
				}
				const word ch = (e & f) ^ (~e & g);
				const word maj = (a & b) ^ (a & c) ^ (b & c);
				const word t1 = hh + S1 + ch + k + w[i];
				const word t2 = S0 + maj;
				hh = g; g = f; f = e; e = d + t1;
				d = c; c = b; b = a; a = t1 + t2;
			}

			h[0] += a; h[1] += b; h[2] += c; h[3] += d;
			h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
		}

	public:
		/// digest length in bytes
		constexpr static size_t digest_len = bits / 8;

		constexpr sha2() noexcept = default;

		/// adds `n` bytes to the message
		constexpr void update(const uint8_t *data, const size_t n) noexcept {
			total += n;
			for (size_t i = 0; i < n; ++i) {
				buf[buf_len++] = data[i];
				if (buf_len == block_len) {
					compress(buf.data());
					buf_len = 0;
				}
			}
		}

		/// adds the chars of `s` to the message
		constexpr void update(const std::string_view s) noexcept {
			total += s.size();
			for (const char c : s) {
				buf[buf_len++] = uint8_t(c);
				if (buf_len == block_len) {
					compress(buf.data());
					buf_len = 0;
				}
			}
		}

		/// pads the message and returns the digest. The state must not be
		/// updated afterwards.
		[[nodiscard]] constexpr std::array<uint8_t, digest_len> digest() noexcept {
			const uint64_t bit_len = total * 8u;
			// 0x80, zeros, and the message length (16 bytes for SHA-384/512)
			constexpr size_t len_bytes = 2 * sizeof(word);
			buf[buf_len++] = 0x80;
			if (buf_len > block_len - len_bytes) {
				while (buf_len < block_len) { buf[buf_len++] = 0; }
				compress(buf.data());
				buf_len = 0;
			}
			while (buf_len < block_len - 8) { buf[buf_len++] = 0; }
			for (uint32_t i = 0; i < 8; ++i) {
				buf[buf_len++] = uint8_t(bit_len >> (56u - 8u * i));
			}
			compress(buf.data());

			std::array<uint8_t, digest_len> ret{};
			for (size_t i = 0; i < digest_len; ++i) {
				ret[i] = uint8_t(h[i / sizeof(word)] >> (8u * (sizeof(word) - 1u - (i % sizeof(word)))));
			}
			return ret;
		}
	};

	using sha224_t = sha2<224>;
	using sha256_t = sha2<256>;
	using sha384_t = sha2<384>;
	using sha512_t = sha2<512>;

	/// \return the SHA-2 digest of the `n` bytes at `data`
	template<const uint32_t bits>
	[[nodiscard]] constexpr std::array<uint8_t, bits / 8> sha2_hash(const uint8_t *data, const size_t n) noexcept {
		sha2<bits> s;
		s.update(data, n);
		return s.digest();
	}

	/// \return the SHA-2 digest of the chars of `s`
	template<const uint32_t bits>
	[[nodiscard]] constexpr std::array<uint8_t, bits / 8> sha2_hash(const std::string_view s) noexcept {
		sha2<bits> h;
		h.update(s);
		return h.digest();
	}

	[[nodiscard]] constexpr std::array<uint8_t, 28> sha224(const uint8_t *data, const size_t n) noexcept { return sha2_hash<224>(data, n); }
	[[nodiscard]] constexpr std::array<uint8_t, 32> sha256(const uint8_t *data, const size_t n) noexcept { return sha2_hash<256>(data, n); }
	[[nodiscard]] constexpr std::array<uint8_t, 48> sha384(const uint8_t *data, const size_t n) noexcept { return sha2_hash<384>(data, n); }
	[[nodiscard]] constexpr std::array<uint8_t, 64> sha512(const uint8_t *data, const size_t n) noexcept { return sha2_hash<512>(data, n); }
	[[nodiscard]] constexpr std::array<uint8_t, 28> sha224(const std::string_view s) noexcept { return sha2_hash<224>(s); }
	[[nodiscard]] constexpr std::array<uint8_t, 32> sha256(const std::string_view s) noexcept { return sha2_hash<256>(s); }
	[[nodiscard]] constexpr std::array<uint8_t, 48> sha384(const std::string_view s) noexcept { return sha2_hash<384>(s); }
	[[nodiscard]] constexpr std::array<uint8_t, 64> sha512(const std::string_view s) noexcept { return sha2_hash<512>(s); }
} // end namespace cryptanalysislib
#endif //CRYPTANALYSISLIB_SHA2_H
