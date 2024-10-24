#ifndef CRYPTANALYSISLIB_HASH_CRC_H
#define CRYPTANALYSISLIB_HASH_CRC_H

#include <cstdint>
#include <cstdlib>

const uint32_t crc32_tab[] = {
	0x00000000, 0x77073096, 0xee0e612c, 0x990951ba, 0x076dc419, 0x706af48f,
	0xe963a535, 0x9e6495a3,	0x0edb8832, 0x79dcb8a4, 0xe0d5e91e, 0x97d2d988,
	0x09b64c2b, 0x7eb17cbd, 0xe7b82d07, 0x90bf1d91, 0x1db71064, 0x6ab020f2,
	0xf3b97148, 0x84be41de,	0x1adad47d, 0x6ddde4eb, 0xf4d4b551, 0x83d385c7,
	0x136c9856, 0x646ba8c0, 0xfd62f97a, 0x8a65c9ec,	0x14015c4f, 0x63066cd9,
	0xfa0f3d63, 0x8d080df5,	0x3b6e20c8, 0x4c69105e, 0xd56041e4, 0xa2677172,
	0x3c03e4d1, 0x4b04d447, 0xd20d85fd, 0xa50ab56b,	0x35b5a8fa, 0x42b2986c,
	0xdbbbc9d6, 0xacbcf940,	0x32d86ce3, 0x45df5c75, 0xdcd60dcf, 0xabd13d59,
	0x26d930ac, 0x51de003a, 0xc8d75180, 0xbfd06116, 0x21b4f4b5, 0x56b3c423,
	0xcfba9599, 0xb8bda50f, 0x2802b89e, 0x5f058808, 0xc60cd9b2, 0xb10be924,
	0x2f6f7c87, 0x58684c11, 0xc1611dab, 0xb6662d3d,	0x76dc4190, 0x01db7106,
	0x98d220bc, 0xefd5102a, 0x71b18589, 0x06b6b51f, 0x9fbfe4a5, 0xe8b8d433,
	0x7807c9a2, 0x0f00f934, 0x9609a88e, 0xe10e9818, 0x7f6a0dbb, 0x086d3d2d,
	0x91646c97, 0xe6635c01, 0x6b6b51f4, 0x1c6c6162, 0x856530d8, 0xf262004e,
	0x6c0695ed, 0x1b01a57b, 0x8208f4c1, 0xf50fc457, 0x65b0d9c6, 0x12b7e950,
	0x8bbeb8ea, 0xfcb9887c, 0x62dd1ddf, 0x15da2d49, 0x8cd37cf3, 0xfbd44c65,
	0x4db26158, 0x3ab551ce, 0xa3bc0074, 0xd4bb30e2, 0x4adfa541, 0x3dd895d7,
	0xa4d1c46d, 0xd3d6f4fb, 0x4369e96a, 0x346ed9fc, 0xad678846, 0xda60b8d0,
	0x44042d73, 0x33031de5, 0xaa0a4c5f, 0xdd0d7cc9, 0x5005713c, 0x270241aa,
	0xbe0b1010, 0xc90c2086, 0x5768b525, 0x206f85b3, 0xb966d409, 0xce61e49f,
	0x5edef90e, 0x29d9c998, 0xb0d09822, 0xc7d7a8b4, 0x59b33d17, 0x2eb40d81,
	0xb7bd5c3b, 0xc0ba6cad, 0xedb88320, 0x9abfb3b6, 0x03b6e20c, 0x74b1d29a,
	0xead54739, 0x9dd277af, 0x04db2615, 0x73dc1683, 0xe3630b12, 0x94643b84,
	0x0d6d6a3e, 0x7a6a5aa8, 0xe40ecf0b, 0x9309ff9d, 0x0a00ae27, 0x7d079eb1,
	0xf00f9344, 0x8708a3d2, 0x1e01f268, 0x6906c2fe, 0xf762575d, 0x806567cb,
	0x196c3671, 0x6e6b06e7, 0xfed41b76, 0x89d32be0, 0x10da7a5a, 0x67dd4acc,
	0xf9b9df6f, 0x8ebeeff9, 0x17b7be43, 0x60b08ed5, 0xd6d6a3e8, 0xa1d1937e,
	0x38d8c2c4, 0x4fdff252, 0xd1bb67f1, 0xa6bc5767, 0x3fb506dd, 0x48b2364b,
	0xd80d2bda, 0xaf0a1b4c, 0x36034af6, 0x41047a60, 0xdf60efc3, 0xa867df55,
	0x316e8eef, 0x4669be79, 0xcb61b38c, 0xbc66831a, 0x256fd2a0, 0x5268e236,
	0xcc0c7795, 0xbb0b4703, 0x220216b9, 0x5505262f, 0xc5ba3bbe, 0xb2bd0b28,
	0x2bb45a92, 0x5cb36a04, 0xc2d7ffa7, 0xb5d0cf31, 0x2cd99e8b, 0x5bdeae1d,
	0x9b64c2b0, 0xec63f226, 0x756aa39c, 0x026d930a, 0x9c0906a9, 0xeb0e363f,
	0x72076785, 0x05005713, 0x95bf4a82, 0xe2b87a14, 0x7bb12bae, 0x0cb61b38,
	0x92d28e9b, 0xe5d5be0d, 0x7cdcefb7, 0x0bdbdf21, 0x86d3d2d4, 0xf1d4e242,
	0x68ddb3f8, 0x1fda836e, 0x81be16cd, 0xf6b9265b, 0x6fb077e1, 0x18b74777,
	0x88085ae6, 0xff0f6a70, 0x66063bca, 0x11010b5c, 0x8f659eff, 0xf862ae69,
	0x616bffd3, 0x166ccf45, 0xa00ae278, 0xd70dd2ee, 0x4e048354, 0x3903b3c2,
	0xa7672661, 0xd06016f7, 0x4969474d, 0x3e6e77db, 0xaed16a4a, 0xd9d65adc,
	0x40df0b66, 0x37d83bf0, 0xa9bcae53, 0xdebb9ec5, 0x47b2cf7f, 0x30b5ffe9,
	0xbdbdf21c, 0xcabac28a, 0x53b39330, 0x24b4a3a6, 0xbad03605, 0xcdd70693,
	0x54de5729, 0x23d967bf, 0xb3667a2e, 0xc4614ab8, 0x5d681b02, 0x2a6f2b94,
	0xb40bbe37, 0xc30c8ea1, 0x5a05df1b, 0x2d02ef8d
};

///
constexpr static uint32_t crc32(const uint8_t *buf, 
							 	const size_t size,
                                const uint32_t crc_) {
	uint32_t crc = ~0U ^ crc_;
	size_t size_ = size;
	while (size_--) {
		crc = crc32_tab[(crc ^ *buf++) & 0xFF] ^ (crc >> 8);
	}

	return crc ^ ~0U;
}

#ifdef USE_PCLMULDQD
#include <immintrin.h>

// SSE4.2+PCLMUL
uint32_t static sse42_crc32(const unsigned char *buf,
    						const size_t len,
    						const uint32_t crc) noexcept {
    // Definitions of the bit-reflected domain constants k1,k2,k3, etc and
    // the CRC32+Barrett polynomials given at the end of the paper.
    static const uint64_t __attribute__((aligned(16))) k1k2[] = { 0x0154442bd4, 0x01c6e41596 };
    static const uint64_t __attribute__((aligned(16))) k3k4[] = { 0x01751997d0, 0x00ccaa009e };
    static const uint64_t __attribute__((aligned(16))) k5k0[] = { 0x0163cd6124, 0x0000000000 };
    static const uint64_t __attribute__((aligned(16))) poly[] = { 0x01db710641, 0x01f7011641 };

    __m128i x0, x1, x2, x3, x4, x5, x6, x7, x8, y5, y6, y7, y8;

    /*
     * There's at least one block of 64.
     */
    x1 = _mm_loadu_si128((__m128i *)(buf + 0x00));
    x2 = _mm_loadu_si128((__m128i *)(buf + 0x10));
    x3 = _mm_loadu_si128((__m128i *)(buf + 0x20));
    x4 = _mm_loadu_si128((__m128i *)(buf + 0x30));
    x1 = _mm_xor_si128(x1, _mm_cvtsi32_si128(crc));
    x0 = _mm_load_si128((__m128i *)k1k2);

    buf += 64;
	size_t len_ = len;
    len_ -= 64;

    /*
     * Parallel fold blocks of 64, if any.
     */
    while (len_ >= 64) {
        x5 = _mm_clmulepi64_si128(x1, x0, 0x00);
        x6 = _mm_clmulepi64_si128(x2, x0, 0x00);
        x7 = _mm_clmulepi64_si128(x3, x0, 0x00);
        x8 = _mm_clmulepi64_si128(x4, x0, 0x00);

        x1 = _mm_clmulepi64_si128(x1, x0, 0x11);
        x2 = _mm_clmulepi64_si128(x2, x0, 0x11);
        x3 = _mm_clmulepi64_si128(x3, x0, 0x11);
        x4 = _mm_clmulepi64_si128(x4, x0, 0x11);

        y5 = _mm_loadu_si128((__m128i *)(buf + 0x00));
        y6 = _mm_loadu_si128((__m128i *)(buf + 0x10));
        y7 = _mm_loadu_si128((__m128i *)(buf + 0x20));
        y8 = _mm_loadu_si128((__m128i *)(buf + 0x30));

        x1 = _mm_xor_si128(x1, x5);
        x2 = _mm_xor_si128(x2, x6);
        x3 = _mm_xor_si128(x3, x7);
        x4 = _mm_xor_si128(x4, x8);

        x1 = _mm_xor_si128(x1, y5);
        x2 = _mm_xor_si128(x2, y6);
        x3 = _mm_xor_si128(x3, y7);
        x4 = _mm_xor_si128(x4, y8);

        buf += 64;
        len_ -= 64;
    }

    /*
     * Fold into 128-bits.
     */
    x0 = _mm_load_si128((__m128i *)k3k4);

    x5 = _mm_clmulepi64_si128(x1, x0, 0x00);
    x1 = _mm_clmulepi64_si128(x1, x0, 0x11);
    x1 = _mm_xor_si128(x1, x2);
    x1 = _mm_xor_si128(x1, x5);

    x5 = _mm_clmulepi64_si128(x1, x0, 0x00);
    x1 = _mm_clmulepi64_si128(x1, x0, 0x11);
    x1 = _mm_xor_si128(x1, x3);
    x1 = _mm_xor_si128(x1, x5);

    x5 = _mm_clmulepi64_si128(x1, x0, 0x00);
    x1 = _mm_clmulepi64_si128(x1, x0, 0x11);
    x1 = _mm_xor_si128(x1, x4);
    x1 = _mm_xor_si128(x1, x5);

    /*
     * Single fold blocks of 16, if any.
     */
    while (len_ >= 16) {
        x2 = _mm_loadu_si128((__m128i *)buf);

        x5 = _mm_clmulepi64_si128(x1, x0, 0x00);
        x1 = _mm_clmulepi64_si128(x1, x0, 0x11);
        x1 = _mm_xor_si128(x1, x2);
        x1 = _mm_xor_si128(x1, x5);

        buf += 16;
        len_ -= 16;
    }

    /*
     * Fold 128-bits to 64-bits.
     */
    x2 = _mm_clmulepi64_si128(x1, x0, 0x10);
    x3 = _mm_setr_epi32(~0, 0, ~0, 0);
    x1 = _mm_srli_si128(x1, 8);
    x1 = _mm_xor_si128(x1, x2);

    x0 = _mm_loadl_epi64((__m128i*)k5k0);

    x2 = _mm_srli_si128(x1, 4);
    x1 = _mm_and_si128(x1, x3);
    x1 = _mm_clmulepi64_si128(x1, x0, 0x00);
    x1 = _mm_xor_si128(x1, x2);

    /*
     * Barret reduce to 32-bits.
     */
    x0 = _mm_load_si128((__m128i*)poly);

    x2 = _mm_and_si128(x1, x3);
    x2 = _mm_clmulepi64_si128(x2, x0, 0x10);
    x2 = _mm_and_si128(x2, x3);
    x2 = _mm_clmulepi64_si128(x2, x0, 0x00);
    x1 = _mm_xor_si128(x1, x2);

    /*
     * Return the crc32.
     */
    return _mm_extract_epi32(x1, 1);
}
#endif


#ifdef USE_NEON
#include <arm_neon.h>
#include <arm_acle.h>
#include <stdint.h>
#include <stddef.h>

uint32_t crc32(uint32_t crc, uint8_t *buf, size_t len) {
    crc = ~crc;

    while (len >= 8) {
        crc = __crc32d(crc, *(uint64_t*)buf);
        len -= 8;
        buf += 8;
    }

    if (len & 4) {
        crc = __crc32w(crc, *(uint32_t*)buf);
        buf += 4;
    }
    if (len & 2) {
        crc = __crc32h(crc, *(uint16_t*)buf);
        buf += 2;
    }
    if (len & 1) {
        crc = __crc32b(crc, *buf);
    }

    return ~crc;
}
#endif

#ifdef USE_SSE41

#define CRC_ITER(i) case i:								\
crcA = _mm_crc32_u64(crcA, *(uint64_t*)(pA - 8*(i)));	\
crcB = _mm_crc32_u64(crcB, *(uint64_t*)(pB - 8*(i)));

#define X0(n) CRC_ITER(n);
#define X1(n) X0(n+1) X0(n)
#define X2(n) X1(n+2) X1(n)
#define X3(n) X2(n+4) X2(n)
#define X4(n) X3(n+8) X3(n)
#define X5(n) X4(n+16) X4(n)
#define X6(n) X5(n+32) X5(n)
#define CRC_ITERS_128_TO_2() do {X0(128) X1(126) X2(122) X3(114) X4(98) X5(66) X6(2)} while(0)

/// Source: https://github.com/komrad36/CRC/tree/master/CRC
/// OPTION 14
uint32_t option_14_golden_amd(const void* M,
							  uint32_t bytes,
							  uint32_t prev/* = 0*/) noexcept {
	// must be >= 16
	constexpr uint32_t LEAF_SIZE_AMD = 7 * 16;

	// for this approach, the poly CANNOT be changed, because this approach
	// uses x86 hardware instructions which hardcode this poly internally.
	constexpr uint32_t P = 0x82f63b78U;

	constexpr uint32_t g_lut_amd[] = {
	    0x00000001, 0x493c7d27, 0xf20c0dfe, 0xba4fc28e, 0x3da6d0cb, 0xddc0152b, 0x1c291d04, 0x9e4addf8,
	    0x740eef02, 0x39d3b296, 0x083a6eec, 0x0715ce53, 0xc49f4f67, 0x47db8317, 0x2ad91c30, 0x0d3b6092,
	    0x6992cea2, 0xc96cfdc0, 0x7e908048, 0x878a92a7, 0x1b3d8f29, 0xdaece73e, 0xf1d0f55e, 0xab7aff2a,
	    0xa87ab8a8, 0x2162d385, 0x8462d800, 0x83348832, 0x71d111a8, 0x299847d5, 0xffd852c6, 0xb9e02b86,
	    0xdcb17aa4, 0x18b33a4e, 0xf37c5aee, 0xb6dd949b, 0x6051d5a2, 0x78d9ccb7, 0x18b0d4ff, 0xbac2fd7b,
	    0x21f3d99c, 0xa60ce07b, 0x8f158014, 0xce7f39f4, 0xa00457f7, 0x61d82e56, 0x8d6d2c43, 0xd270f1a2,
	    0x00ac29cf, 0xc619809d, 0xe9adf796, 0x2b3cac5d, 0x96638b34, 0x65863b64, 0xe0e9f351, 0x1b03397f,
	    0x9af01f2d, 0xebb883bd, 0x2cff42cf, 0xb3e32c28, 0x88f25a3a, 0x064f7f26, 0x4e36f0b0, 0xdd7e3b0c,
	    0xbd6f81f8, 0xf285651c, 0x91c9bd4b, 0x10746f3c, 0x885f087b, 0xc7a68855, 0x4c144932, 0x271d9844,
	    0x52148f02, 0x8e766a0c, 0xa3c6f37a, 0x93a5f730, 0xd7c0557f, 0x6cb08e5c, 0x63ded06a, 0x6b749fb2,
	    0x4d56973c, 0x1393e203, 0x9669c9df, 0xcec3662e, 0xe417f38a, 0x96c515bb, 0x4b9e0f71, 0xe6fc4e6a,
	    0xd104b8fc, 0x8227bb8a, 0x5b397730, 0xb0cd4768, 0xe78eb416, 0x39c7ff35, 0x61ff0e01, 0xd7a4825c,
	    0x8d96551c, 0x0ab3844b, 0x0bf80dd2, 0x0167d312, 0x8821abed, 0xf6076544, 0x6a45d2b2, 0x26f6a60a,
	    0xd8d26619, 0xa741c1bf, 0xde87806c, 0x98d8d9cb, 0x14338754, 0x49c3cc9c, 0x5bd2011f, 0x68bce87a,
	    0xdd07448e, 0x57a3d037, 0xdde8f5b9, 0x6956fc3b, 0xa3e3e02c, 0x42d98888, 0xd73c7bea, 0x3771e98f,
	    0x80ff0093, 0xb42ae3d9, 0x8fe4c34d, 0x2178513a, 0xdf99fc11, 0xe0ac139e, 0x6c23e841, 0x170076fa,
	};

	// using hardware crc instructions to generate lut
	auto compute_golden_lut_amd = [](uint32_t* pTbl,
									 uint32_t n) noexcept {
	    uint64_t R = 1;
	    for (uint32_t i = 0; i < n << 1; ++i) {
	        pTbl[i] = (uint32_t)R;
	        R = _mm_crc32_u64(R, 0);
	    }
	};

    uint64_t pA = (uint64_t)M;
    uint64_t crcA = prev;
    uint32_t toAlign = ((uint64_t)-(int64_t)pA) & 7;

    for (; toAlign && bytes; ++pA, --bytes, --toAlign)
        crcA = _mm_crc32_u8((uint32_t)crcA, *(uint8_t*)pA);

    while (bytes >= LEAF_SIZE_AMD) {
        const uint32_t n = bytes < 128 * 16 ? bytes >> 4 : 128;
        pA += 8 * n;
        uint64_t pB = pA + 8 * n;
        uint64_t crcB = 0;
        switch (n)
            CRC_ITERS_128_TO_2();

        crcA = _mm_crc32_u64(crcA, *(uint64_t*)(pA - 8));
        const __m128i vK = _mm_cvtsi32_si128(g_lut_amd[n - 1]);
        const __m128i vA = _mm_clmulepi64_si128(_mm_cvtsi64_si128(crcA), vK, 0);
        crcA = _mm_crc32_u64(crcB, _mm_cvtsi128_si64(vA) ^ *(uint64_t*)(pB - 8));

        bytes -= n << 4;
        pA = pB;
    }

    for (; bytes >= 8; bytes -= 8, pA += 8) {
	    crcA = _mm_crc32_u64(crcA, *(uint64_t*)(pA));
    }

    for (; bytes; --bytes, ++pA) {
	    crcA = _mm_crc32_u8((uint32_t)crcA, *(uint8_t*)(pA));
    }

    return (uint32_t)crcA;
}

#undef CRC_ITER
#undef X0
#undef X1
#undef X2
#undef X3
#undef X4
#undef X5
#undef X6
#undef CRC_ITERS_128_TO_2

#endif
#endif
