#ifndef CRYTPANALYSISLIB_HASHMAP_AVX2_H
#define CRYTPANALYSISLIB_HASHMAP_AVX2_H

#include <immintrin.h>
#include <cstdint>
#include <array>

#include "container/hashmap/common.h"
#include "simd/simd.h"

/// TODO all this values as constructio
class SIMDHashMapConfig {
public:
	const uint32_t bucket_size = 10;
	const uint32_t nr_buckets = 1<<4;
	const uint32_t low = 0;
	const uint32_t high = 0;
	const uint32_t threads = 1;

	constexpr SIMDHashMapConfig(const uint64_t bucket_size,
								const uint64_t nr_buckets,
	                            const uint32_t low,
	                            const uint32_t high,
								const uint32_t threads = 1u) :
			bucket_size(bucket_size), nr_buckets(nr_buckets),
	        low(low), high(high),threads(threads) {};
};



template<const SIMDHashMapConfig &config>
class SIMDHashMap {
public:
	using T         = uint32x8_t;

	// must be chosen in such a way that `l` bits fit into it.
	using TLimbType = uint32_t;
	using LoadType 	= uint32_t;

	constexpr static size_t nr_elements_container = 8; // uin

	// number of indices to be able to save in the hashmap
	constexpr static size_t nri 	= 2;
	constexpr static size_t nri_mult= nri;

	// number of buckets
	constexpr static size_t nrb     = config.nr_buckets;

	// number of elements in each bucket
	constexpr static size_t sizeb   = config.bucket_size;

	// needed for the hashmap
	constexpr static uint32_t low = config.low,
	                          high = config.high;

	// first is for the data, second index is for the index within the base list.
	std::array<T [nri], nrb*sizeb> __buckets;
	// load for each bucket
	std::array<LoadType, nrb>     __load;

	// we need different hash functions
	// This is the simplest version of a hash function. It extracts the bits between l and h
	// and returns them with zero alignment
	template<const uint32_t l, const uint32_t h>
	constexpr inline static TLimbType HashSimple(const uint64_t a) noexcept {
		static_assert(l < h);
		static_assert(h < 64);
		constexpr uint64_t mask = (~((uint64_t(1u) << l) - 1u)) &
								  ( (uint64_t(1u) << h) - 1u);
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
	void clear() {
		for (size_t i = 0; i < __load.size(); ++i) {
			__load[0] = 0;
		}
	}

	// return 'load' factor
	constexpr inline LoadType get_load(const TLimbType bid) noexcept{
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
										const TLimbType load) {
		ASSERT(bid < nrb);
		ASSERT(load < sizeb);

		// you need to multiply it with 2, because each bucket element
		// consists of 1 data element and one index element.
		bucket_index = nri_mult*bid*sizeb + load;
		inner_bucket_index = bucket_index%nr_elements_container;
		bucket_index = bucket_index/nr_elements_container;
	}


	///
	/// \param bucket_index
	/// \param inner_bucket_index
	/// \param bid
	/// \param load
	inline void bucket_offset_avx(T &bucket_index,
								  T &inner_bucket_index,
								  const T bid,
								  const T load) {
		constexpr static T avxsizeb = T::set1(sizeb);

		// mod 8 is the same as &7
		constexpr static T avxmod   = T::set1(7u);

		// for the multiplication times 2/3 to get the exact array entry positions
		constexpr static T avxmul   = T::set1(nri_mult);

		bucket_index = (bid*avxsizeb*avxmul) + load;
		inner_bucket_index = bucket_index & avxmod;
		bucket_index = bucket_index >> 3u;
	}

	// insert a single element
	void insert_simple(const TLimbType data, const TLimbType index) noexcept {
		const TLimbType bid  = HashSimple<low, high>(data);
		const TLimbType load = get_load(bid);

		// early exit
		if (load >= sizeb) {
			return;
		}

		size_t bucket_index, inner_bucket_index;
		bucket_offset(bucket_index, inner_bucket_index, bid, load);

		__buckets[bucket_index][0].v32[inner_bucket_index] = data;
		__buckets[bucket_index][1].v32[inner_bucket_index] = index;
	}

	// nearly fullly avx implementation
	void insert_avx(const T &data,
					const T &index,
					const T &load) noexcept {
		const T bid = HashAVX<low, high>(data);
		T bucket_index, inner_bucket_index;
		bucket_offset_avx(bucket_index, inner_bucket_index, bid, load);

		// TODO replace with scatter operations from simd wrapper
		for (uint32_t i = 0; i < nr_elements_container; i++) {
			__buckets[bucket_index.v32[i]][0].v32[inner_bucket_index.v32[i]] = data.v32[i];
			__buckets[bucket_index.v32[i]][1].v32[inner_bucket_index.v32[i]] = index.v32[i];
		}
	}

	void insert(const T &data, const T &index) noexcept {
		const T bid = HashAVX<low, high>(data);
		for (uint32_t i = 0; i < nr_elements_container; i++) {
			const TLimbType load = get_load(bid.v32[i]);
			size_t bucket_index, inner_bucket_index;
			bucket_offset(bucket_index, inner_bucket_index, bid.v32[i], load);

			__buckets[bucket_index][0].v32[inner_bucket_index] = data.v32[i];
			__buckets[bucket_index][1].v32[inner_bucket_index] = index.v32[i];
		}
	}
};
#endif
