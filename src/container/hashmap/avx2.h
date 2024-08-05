#ifndef CRYTPANALYSISLIB_HASHMAP_AVX2_H
#define CRYTPANALYSISLIB_HASHMAP_AVX2_H

#if !defined(CRYPTANALYSISLIB_HASHMAP_H)
#error "Do not include this file directly. Use: `#include <container/hashmap.h>`"
#endif

#include <array>
#include <cstdint>
#include <immintrin.h>

#include "container/hashmap/common.h"
#include "simd/simd.h"

class SIMDHashMapConfig {
public:
	const uint32_t bucket_size = 10;
	const uint32_t nr_buckets = 1 << 4;
	const uint32_t low = 0;
	const uint32_t high = 0;
	const uint32_t threads = 1;

	constexpr SIMDHashMapConfig(const uint64_t bucket_size,
	                            const uint64_t nr_buckets,
	                            const uint32_t low,
	                            const uint32_t high,
	                            const uint32_t threads = 1u) : bucket_size(bucket_size), nr_buckets(nr_buckets),
	                                                           low(low), high(high), threads(threads){};
};


/// stores the data like: each internal element:
/// lower addr high addr
/// [uint32x8_t, uint32x8_t]
/// [[l part, ..., lpart], [index to base list, ..., index, to base list]]
///
/// \tparam config
template<const SIMDHashMapConfig &config>
class SIMDHashMap {
public:
	using index_type = size_t;
	using data_type = uint32x8_t;
	using T = uint32x8_t;

	// must be chosen in such a way that `l` bits fit into it.
	using TLimbType = uint32_t;
	using LoadType = uint32_t;

	// this stuff must be hardcoded
	constexpr static size_t nr_elements_container = 8;

	// number of indices to be able to save in the hashmap
	constexpr static size_t nri = 2;
	constexpr static size_t nri_mult = nri;

	// number of buckets
	constexpr static size_t nrb = config.nr_buckets;

	// number of elements in each bucket
	constexpr static size_t sizeb = config.bucket_size;
	constexpr static size_t total_size = sizeb * nrb;
	constexpr static size_t total_load = nrb * nr_elements_container;

	// needed for the hashmap
	constexpr static uint32_t low = config.low,
	                          high = config.high;

	using internal_type = data_type[2];
	// first is for the data, second index is for the index within the base list.
	std::array<internal_type, nrb * sizeb> __buckets;
	// load for each bucket
	std::array<LoadType, nrb> __load;

	// we need different hash functions
	// This is the simplest version of a hash function. It extracts the bits between l and h
	// and returns them with zero alignment
	template<const uint32_t l, const uint32_t h>
	constexpr inline static TLimbType HashSimple(const uint64_t a) noexcept {
		static_assert(l < h);
		static_assert(h < 64);
		constexpr uint64_t mask = (~((uint64_t(1u) << l) - 1u)) &
		                          ((uint64_t(1u) << h) - 1u);
		return (uint64_t(a) & mask) >> l;
	}

	// Broadcast on every 32 bit limb within the avx register
	template<const uint32_t l, const uint32_t h>
	constexpr inline static T HashBroadCast(const uint64_t a) noexcept {
		const TLimbType b = HashSimple<l, h>(a);
		return T::set1(b);
	}

	///
	template<const uint32_t l, const uint32_t h>
	constexpr inline static T HashAVX(const T &a) noexcept {
		constexpr TLimbType mask1 = (~((TLimbType(1u) << l) - 1u)) & ((TLimbType(1u) << h) - 1u);
		constexpr T mask2 = T::set1(mask1);
		return (a & mask2) >> l;
	}

	/// doesnt clear anything, but only sets the load factors to zero.
	constexpr void clear(const uint32_t tid) noexcept {
		const size_t start = tid * __load.size() / config.threads;
		const size_t end = (tid + 1) * __load.size() / config.threads;
		for (size_t i = start; i < end; ++i) {
			__load[i] = 0;
		}
	}

	/// doesnt clear anything, but only sets the load factors to zero.
	constexpr void clear() noexcept {
		for (size_t i = 0; i < __load.size(); ++i) {
			__load[i] = 0;
		}
	}

	///
	/// \return
	constexpr inline internal_type *ptr() noexcept {
		return __buckets.data();
	}

	///
	/// \param i
	/// \return
	constexpr inline data_type ptr(const index_type i) noexcept {
		ASSERT(i < total_size);
		return (data_type) __buckets[i][0];
	}

	// return 'load' factor
	constexpr inline LoadType load(const TLimbType bid) noexcept {
		ASSERT(bid < nrb);
		return __load[bid];
	}

	///
	/// \param bucket_index
	/// \param inner_bucket_index
	/// \param bid
	/// \param load
	/// \return
	constexpr inline void bucket_offset(size_t &bucket_index,
	                                    size_t &inner_bucket_index,
	                                    const TLimbType bid,
	                                    const TLimbType load) noexcept {
		ASSERT(bid < nrb);
		ASSERT(load < sizeb);

		bucket_index = bid * sizeb + load;
		inner_bucket_index = bucket_index % nr_elements_container;
		bucket_index = bucket_index / nr_elements_container;
	}


	///
	/// \param bucket_index
	/// \param inner_bucket_index
	/// \param bid
	/// \param load
	constexpr inline void bucket_offset_avx(T &bucket_index,
	                              T &inner_bucket_index,
	                              const T bid,
	                              const T load) noexcept {
		constexpr T avxsizeb = T::set1(sizeb);

		// mod 8 is the same as &7
		constexpr T avxmod = T::set1(7u);

		bucket_index = (bid * avxsizeb) + load;
		inner_bucket_index = bucket_index & avxmod;
		bucket_index = bucket_index >> 3u;
	}

	// insert a single element
	constexpr void insert_simple(const TLimbType data,
	                             const TLimbType index) noexcept {
		const TLimbType bid = HashSimple<low, high>(data);
		const TLimbType _load = load(bid);

		// early exit
		if (_load >= total_load) {
			return;
		}

		size_t bucket_index, inner_bucket_index;
		bucket_offset(bucket_index, inner_bucket_index, bid, _load);

		__buckets[bucket_index][0].v32[inner_bucket_index] = data;
		__buckets[bucket_index][1].v32[inner_bucket_index] = index;

		__load[bid] += 1;
	}

	// nearly fullly avx implementation
	constexpr void insert_simd(const T &data,
	                 const T &index) noexcept {
		const T bid = HashAVX<low, high>(data);
		T bucket_index{}, inner_bucket_index{}, load{};
		bucket_offset_avx(bucket_index, inner_bucket_index, bid, load);

		for (uint32_t i = 0; i < nr_elements_container; i++) {
			uint32x8_t::scatter((void *) __buckets.data(), bucket_index, data);
			uint32x8_t::scatter((void *) ((uint8_t *) __buckets.data() + 32), bucket_index, index);
		}
	}
};
#endif
