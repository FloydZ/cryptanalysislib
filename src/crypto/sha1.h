#ifndef CRYPTANALYSISLIB_SHA1_H
#define CRYPTANALYSISLIB_SHA1_H

#include "macros.h"
#include "mem/memcpy.h"
#include "mem/memset.h"

constexpr int get_string_size(const uint8_t *input, int i = 0) {
    return input[i] == '\0' ? i : get_string_size(input, i+1);
}

namespace cryptanalysislib {
	namespace internal {
		// output size in bytes
		constexpr size_t SHA1_DIGEST_LEN = 20;

		// same as `SHA1_DIGEST_LEN` only as number of limbs of type `uint32_t`
		constexpr size_t SHA1_DIGEST_UINT32_LEN = SHA1_DIGEST_LEN/4;

		// length of the internal state in bytes
		constexpr size_t SHA1_STATE_LEN = 32;

		// length of the CBLOCK in bytes
		constexpr size_t SHA1_CBLOCK_LEN = 64;

		constexpr size_t SHA1_ROUNDS = 80;

		// internal helper functions
		constexpr void sha1_transform(
				uint32_t ctx_s[SHA1_CBLOCK_LEN/4],
				const uint32_t ctx_buf[SHA1_CBLOCK_LEN/4]) {
			uint32_t w[SHA1_ROUNDS], s[SHA1_DIGEST_LEN/4], t;

			// copy input into local buffer
			for (uint32_t i = 0; i < 16; ++i) {
				w[i] = SWAP32(ctx_buf[i]);
			}

			// expand it
			for (int i = 16; i < SHA1_ROUNDS; ++i) {
				w[i] = ROTL32(w[i-1] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
			}

			// memcpy(static_cast<uint8_t *>(s), (uint8_t *)ctx_s, SHA1_DIGEST_LEN);
			cryptanalysislib::memcpy<uint8_t>((uint8_t *)s, ctx_s, SHA1_DIGEST_LEN);

			// for 80 rounds
			for (uint32_t i = 0; i < 80; i++) {
				if (i < 20) {
					t = (s[3] ^ (s[1] & (s[2] ^ s[3]))) + 0x5A827999L;
				} else if (i < 40) {
					t = (s[1] ^ s[2] ^ s[3]) + 0x6ED9EBA1L;
				} else if (i < 60) {
					t = ((s[1] & s[2]) | (s[3] & (s[1] | s[2]))) + 0x8F1BBCDCL;
				} else {
					t = (s[1] ^ s[2] ^ s[3]) + 0xCA62C1D6L;
				}
				t += ROTL32(s[0], 5) + s[4] + w[i];
				s[4] = s[3];
				s[3] = s[2];
				s[2] = ROTL32(s[1], 30);
				s[1] = s[0];
				s[0] = t;
			}
			// update state
			for (uint32_t i = 0; i < SHA1_DIGEST_LEN/4; i++) {
				ctx_s[i] += s[i];
			}
		}
	};

	constexpr void sha1(uint8_t *out, const uint8_t *in, const size_t len) noexcept {
		// init
		uint32_t s[internal::SHA1_DIGEST_UINT32_LEN] = {0}, buf[internal::SHA1_CBLOCK_LEN/4];
		s[0] = 0x67452301;
		s[1] = 0xefcdab89;
		s[2] = 0x98badcfe;
		s[3] = 0x10325476;
		s[4] = 0xc3d2e1f0;

		uint8_t *p = (uint8_t *)in;
		size_t internal_len = len;
		uint32_t r;

		if (len == 0)
			return;

		// update
		while (internal_len > 0) {
			r = std::min(internal_len, internal::SHA1_CBLOCK_LEN);
			cryptanalysislib::memcpy<uint8_t>(buf, p, r);

			if (r < internal::SHA1_CBLOCK_LEN)
				break;

			internal::sha1_transform(s, buf);
			internal_len -= r;
			p += r;
		}

		// finalize
		uint32_t idx = len & (internal::SHA1_CBLOCK_LEN - 1);

		// fill remaining space with zeros
		cryptanalysislib::memset<uint8_t>(&((uint8_t *)buf)[len], 0, internal::SHA1_CBLOCK_LEN - len);

		// add the end bit
		((uint8_t *)buf)[len] = 0x80;

		// if exceeding 56 bytes, transform it
		if (internal_len >= 56) {
			internal::sha1_transform(s, buf);
			// clear buffer
			memset((uint8_t *)buf, 0, internal::SHA1_CBLOCK_LEN);
		}
		// add total bits
		((uint64_t *)buf)[7] = SWAP64(internal_len * 8);

		// compress
		internal::sha1_transform(s, buf);

		// swap byte order and save
		for (uint32_t i = 0; i < internal::SHA1_DIGEST_UINT32_LEN; i++) {
			s[i] = SWAP32(s[i]);
		}
		// copy digest to buffer
		memcpy((uint8_t *)out, (uint8_t *)s, internal::SHA1_DIGEST_LEN);
	}
	
	class SHA1 {
		uint8_t data[internal::SHA1_DIGEST_LEN] = {0};
		public:
		constexpr SHA1(const uint8_t *input) {
			sha1(data, input, get_string_size(input));

		}
	};
};


#endif //CRYPTANALYSISLIB_SHA1_H
