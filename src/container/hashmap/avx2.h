//
// Created by duda on 9/20/23.
//

#ifndef DECODING_AVX2_H
#define DECODING_AVX2_H

#include <immintrin.h>
#include <cstdint>
#include <array>


#include "container/hashmap/common.h"

class AVXHashMap {
public:
	using T         = __m256i;
	using lLimbType = uint32_t;
	using LoadType = uint32_t;

	struct __attribute__ ((packed)) DataContainer {
		// how man 32 bit values fit into this
		constexpr static size_t elements = 8;

		// data container
		union {
			__m256i data;
			uint32_t data32[8];
		};

		DataContainer() : data32() {}
		DataContainer(__m256i d) {
			data = d;
		}
	};

	// number of indices to be able to save in the hashmap
	constexpr static size_t nri 	= 1;
	constexpr static size_t nri_mult= nri+1;

	// number of buckets
	constexpr static size_t nrb     = 1u << 10u;

	// number of elements in each bucket
	constexpr static size_t sizeb   = 4;

	constexpr static size_t nr_elements_container = DataContainer::elements;

	// needed for the hashmap
	constexpr static uint32_t low   = 0, high = 10;

	// data
	std::array<DataContainer[2], nrb*sizeb> __buckets;
	std::array<LoadType, nrb>               __load;

	// Factor this out, so we can get messy with pdep_u64 vs. variable-shift
	static inline __m256i unpack_24b_shufmask(unsigned int packed_mask) noexcept {
#ifdef __amd64__
		uint64_t want_bytes = _pdep_u64(packed_mask, 0x0707070707070707);  // deposit every 3bit index in a separate byte
		__m128i bytevec = _mm_cvtsi64_si128(want_bytes);
		__m256i shufmask = _mm256_cvtepu8_epi32(bytevec);
#else
		// alternative: broadcast / variable-shift.  Requires a vector constant, though.
		// This strategy is probably better if the packed mask is coming directly from memory, esp. on Skylake where variable-shift is cheap
		__m256i indices = _mm256_set1_epi32(packed_mask);
		__m256i m = _mm256_sllv_epi32(indices, _mm256_setr_epi32(29, 26, 23, 20, 17, 14, 11, 8));
		__m256i shufmask = _mm256_srli_epi32(m, 29);
#endif
		return shufmask;
	}

	// old version, using packed 3bit groups, instead of each in a separate byte
	static inline __m256 compress256_32bit(__m256 src, unsigned int mask /* from movmskps */) noexcept {
		unsigned expanded_mask = _pdep_u32(mask, 0b001'001'001'001'001'001'001'001);
		expanded_mask += (expanded_mask<<1) + (expanded_mask<<2);  // ABC -> AAABBBCCC: triplicate each bit
		// + instead of | lets the compiler implement it with a multiply by 7

		const unsigned identity_indices = 0b111'110'101'100'011'010'001'000;    // the identity shuffle for vpermps, packed
		unsigned wanted_indices = _pext_u32(identity_indices, expanded_mask);   // just the indices we want, contiguous in the low 24 bits or less.

		// unpack the same as we would for the LUT version
		__m256i shufmask = unpack_24b_shufmask(wanted_indices);
		return _mm256_permutevar8x32_ps(src, shufmask);
	}

	static inline __m256 compress256(__m256 src, unsigned int mask /* from movmskps */) noexcept {
		//return compress256_32bit(src, mask);
		uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);  // unpack each bit to a byte
		expanded_mask *= 0xFFU;  // mask |= mask<<1 | mask<<2 | ... | mask<<7;
		// ABC... -> AAAAAAAABBBBBBBBCCCCCCCC...: replicate each bit to fill its byte

		const uint64_t identity_indices = 0x0706050403020100;    // the identity shuffle for vpermps, packed to one index per byte
		uint64_t wanted_indices = _pext_u64(identity_indices, expanded_mask);

		__m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
		__m256i shufmask = _mm256_cvtepu8_epi32(bytevec);

		return _mm256_permutevar8x32_ps(src, shufmask);
	}

	// we need different hash functions
	// This is the simplest version of a hash function. It extracts the bits between l and h
	// and returns them with zero alignment
	template<const uint32_t l, const uint32_t h>
	constexpr inline static lLimbType HashSimple(const uint64_t a) noexcept {
		static_assert(l < h);
		static_assert(h < 64);
		constexpr uint64_t mask = (~((uint64_t(1u) << l) - 1u)) &
								  ((uint64_t(1u) << h) - 1u);
		return (uint64_t(a) & mask) >> l;
	}

	// Broadcast on every 32 bit limb within the avx register
	template<const uint32_t l, const uint32_t h>
	constexpr inline static __m256i HashBroadCast(const uint64_t a) noexcept {
		const lLimbType b = HashSimple<l, h>(a);
		return _mm256_set1_epi32(b);
	}

	//
	template<const uint32_t l, const uint32_t h>
	constexpr inline static __m256i HashAVX(const DataContainer &a) noexcept {
		constexpr lLimbType mask1 = (~((lLimbType(1u) << l) - 1u)) & ((lLimbType(1u) << h) - 1u);
		const __m256i   mask2 = _mm256_setr_epi32(mask1, mask1, mask1, mask1, mask1, mask1, mask1, mask1);
		return _mm256_srli_epi32(_mm256_xor_si256(a.data, mask2), l);
	}

	void clear() {
		for (size_t i = 0; i < __load.size(); ++i) {
			__load[0] = 0;
		}
	}

	// return 'load' factor
	constexpr inline LoadType get_load(const lLimbType bid) noexcept{
		return __load[bid];
	}

	constexpr inline void bucket_offset(size_t &bucket_index,
										size_t &inner_bucket_index,
										const lLimbType bid,
										const lLimbType load) {
		bucket_index = nri_mult*bid*sizeb + load;
		inner_bucket_index = bucket_index%nr_elements_container;
		bucket_index = bucket_index/nr_elements_container;
	}


	inline void bucket_offset_avx(DataContainer &bucket_index,
								  DataContainer &inner_bucket_index,
								  const __m256i bid,
								  const __m256i load) {

		// TODO mul 2 missing
		const static __m256i avxsizeb = _mm256_setr_epi32(sizeb, sizeb, sizeb, sizeb, sizeb, sizeb, sizeb, sizeb);

		// mod 8 is the same as &7
		const static __m256i avxmod   = _mm256_setr_epi32(7,7,7,7,7,7,7,7);

		// for the multiplication times 2/3 to get the exact array entry positions
		const static __m256i avxmul   = _mm256_setr_epi32(nri_mult, nri_mult, nri_mult, nri_mult, nri_mult, nri_mult, nri_mult, nri_mult);

		bucket_index.data = _mm256_add_epi32(_mm256_mullo_epi32(_mm256_mullo_epi32(bid, avxsizeb), avxmul), load);
		inner_bucket_index.data = _mm256_srli_epi32(bucket_index.data, 3);
		bucket_index.data = _mm256_and_epi32(bucket_index.data, avxmod);
	}

	// insert a single element
	void insert_simple(const lLimbType data, const lLimbType index) noexcept {
		const lLimbType bid  = HashSimple<low, high>(data);
		const lLimbType load = get_load(bid);

		// TODO check if faster to simply overwrite an other element in the bucket
		if (load >= sizeb)
			return;

		size_t bucket_index, inner_bucket_index;
		bucket_offset(bucket_index, inner_bucket_index, bid, load);

		__buckets[bucket_index + 0]->data32[inner_bucket_index] = data;
		__buckets[bucket_index + 1]->data32[inner_bucket_index] = index;
	}

	// nearly fullly avx implementation
	void insert_avx(const DataContainer &data,
					const DataContainer &index,
					const __m256i load) noexcept {
		const __m256i bid = HashAVX<low, high>(data);
		DataContainer bucket_index, inner_bucket_index;
		bucket_offset_avx(bucket_index, inner_bucket_index, bid, load);

		for (uint32_t i = 0; i < nr_elements_container; i++) {
			__buckets[bucket_index.data32[i] + 0]->data32[inner_bucket_index.data32[i]] = data.data32[i];
			__buckets[bucket_index.data32[i] + 1]->data32[inner_bucket_index.data32[i]] = index.data32[i];

		}
	}

	void insert(const DataContainer &data, const DataContainer &index) noexcept {
		const __m256i bid = HashAVX<low, high>(data);
		for (uint32_t i = 0; i < nr_elements_container; i++) {
			const lLimbType load = get_load(bid[i]);
			size_t bucket_index, inner_bucket_index;
			bucket_offset(bucket_index, inner_bucket_index, bid[i], load);

			__buckets[bucket_index + 0]->data32[inner_bucket_index] = data.data32[i];
			__buckets[bucket_index + 1]->data32[inner_bucket_index] = index.data32[i];
		}
	}
};
#endif//DECODING_AVX2_H
