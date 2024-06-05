#ifndef CRYPTANALYSISLIB_SORTING_NETWORK_NEON_H
#define CRYPTANALYSISLIB_SORTING_NETWORK_NEON_H

#ifndef USE_NEON
#error "no neon"
#endif

#ifndef CRYPTANALYSISLIB_SORTING_NETWORK_H
#error "dont inlude 'sort/sorting_network/avx2.h' directly. Use `sort/sorting_network/common.h`."
#endif


#include <arm_neon.h>
#include <stdint.h>

// signed
#define COEX8X16(a, b, tmp)               \
	{                                     \
		tmp = a;                 		  \
		a = vminq_s8(a, b);        	  	  \
		b = vmaxq_s8(tmp, b);  	          \
	}
#define COEX16X8(a, b, tmp)               \
	{                                     \
		tmp = a;                 		  \
		a = vminq_s16(a, b);       	  	  \
		b = vmaxq_s16(tmp, b); 	  	      \
	}
#define COEX32X4(a, b, tmp)               \
	{                                     \
		tmp = a;                 		  \
		a = vminq_s32(a, b);       	  	  \
		b = vmaxq_s32(tmp, b); 	          \
	}

// unsigned
#define UCOEX8X16(a, b, tmp)              \
	{                                     \
		tmp = a;                 		  \
		a = vminq_u8(a, b);        	  	  \
		b = vmaxq_u8(tmp, b);  	          \
	}
#define UCOEX16X8(a, b, tmp)              \
	{                                     \
		tmp = a;                 		  \
		a = vminq_u16(a, b);       	  	  \
		b = vmaxq_u16(tmp, b); 	  	      \
	}
#define UCOEX32X4(a, b, tmp)              \
	{                                     \
		tmp = a;                 		  \
		a = vminq_u32(a, b);       	  	  \
		b = vmaxq_u32(tmp, b); 	          \
	}




#define shuffle_epi8(a, b) vqtbl1q_s8(a, b)


/// simply sorting u8x16 elements
uint8x16_t sort_u8x16(uint8x16_t v) {
    uint8x16_t t, tmp;
    t = v;

    // Step 1
    t = vrev16q_u8(t);
    UCOEX8X16(t, v, tmp);
    t = vtrn2q_u8(v, t);

    // Step 2
    v = vrev32q_u8(t);
    UCOEX8X16(t, v, tmp);
    v = vtrn2q_u16(v, t);

    // Step 3
    t = vrev16q_u8(v);
    UCOEX8X16(v, t, tmp);
    t = vtrn2q_u8(t, v);

    // Step 4
    v = vrev64q_u8(t);
    UCOEX8X16(t, v, tmp);
    v = vtrn2q_u32(v, t);

    // Step 5
    t = shuffle_epi8(v, vld1q_u8(layers[0]));
    UCOEX8X16(v, t, tmp);
    t = vtrn2q_u16(t, v);

    // Step 6
    v = vrev16q_u8(t);
    UCOEX8X16(t, v, tmp);
    v = vtrn2q_u8(v, t);

    // Step 7
    t = shuffle_epi8(v, vld1q_u8(layers[1]));
    UCOEX8X16(v, t, tmp);
    t = vtrn2q_u64(v, t);

    // Step 8
    v = shuffle_epi8(t, vld1q_u8(layers[2]));
    UCOEX8X16(t, v, tmp);
    v = vtrn2q_u32(t, v);

    // Step 9
    t = shuffle_epi8(v, vld1q_u8(layers[0]));
    UCOEX8X16(v, t, tmp);
    t = vtrn2q_u16(v, t);

    // Step 10
    v = vrev16q_u8(t);
    UCOEX8X16(t, v, tmp);
    v = vtrn2q_u8(t, v);
    return v;
}

///
void sortingnetwork_mergesort_u8x32(uint8x16_t *a, uint8x16_t *b) {
    uint8x16_t H1 = shuffle_epi8((*b), vld1q_u8(layers[1]));
    uint8x16_t L1 = *a;
    uint8x16_t tmp;

    UCOEX8X16(L1, H1, tmp);
    uint8x16_t L1p = vzip1q_u64(L1, H1);
    uint8x16_t H1p = vzip2q_u64(L1, H1);

    UCOEX8X16(L1p, H1p, tmp);
    uint8x16_t L2p = vtrn1q_u32(L1p, H1p);
    uint8x16_t H2p = vtrn2q_u32(L1p, H1p);

    UCOEX8X16(L2p, H2p, tmp);
    uint8x16_t L3p = vtrn1q_u16(L2p, H2p);
    uint8x16_t H3p = vtrn2q_u16(L2p, H2p);

    UCOEX8X16(L3p, H3p, tmp);
    uint8x16_t L4p = vtrn1q_u8(L3p, H3p);
    uint8x16_t H4p = vtrn2q_u8(L3p, H3p);

    UCOEX8X16(L4p, H4p, tmp);
    *a = vzip1q_u8(L4p, H4p);
    *b = vzip2q_u8(L4p, H4p);
}

void sortingnetwork_sort_u8x322(uint8x16_t *a, uint8x16_t *b) {
    *a = sort_u8x16(*a);
    *b = sort_u8x16(*b);
    simd_mergesorted_2(a, b);
}
#endif
