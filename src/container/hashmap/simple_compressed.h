#ifndef SMALLSECRETLWE_CONTAINER_HASHMAP_SIMPLE_CONTAINER_H
#define SMALLSECRETLWE_CONTAINER_HASHMAP_SIMPLE_CONTAINER_H

#if !defined(CRYPTANALYSISLIB_HASHMAP_H)
#error "Do not include this file directly. Use: `#include <container/hashmap.h>`"
#endif

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "hash/hash.h"
#include "compression/compression.h"
#include "helper.h"

///
struct SimpleCompressedHashMapConfig {
public:
	/// NOTE no constructor as not neded
	const uint64_t bytes_per_bucket;
	const uint64_t nrbuckets;
};

/// NOTE: Only the indices of the list entries can be saved in here.
/// NOTE: cannot be multithreaded 
/// NOTE: key difference between this class and the `SimpleHashMap` is that the 
/// 	deltas of the input indicies are compressed via led128.
/// \tparam keyType		base type of the input keys
/// \tparam valueType	base type of the values to save in the list
/// \tparam config		`SimpleHashmapConfig` object
/// \tparam Hash		internal hash function to use.
template<
        typename keyType,
        typename valueType,
        const SimpleCompressedHashMapConfig &config,
        class Hash>
//#if __cplusplus > 201709L // TODO
//	requires HashFunction<Hash, valueType>
//#endif
class SimpleCompressedHashMap {
public:
	using data_type  = valueType;
	using key_type   = keyType;
	using index_type = size_t;
	using load_type  = size_t;

	Hash hashclass = Hash{};

	// size per bucket
	constexpr static size_t bytes_per_bucket = config.bytes_per_bucket;

	// number of buckets
	constexpr static size_t nrbuckets = config.nrbuckets;

	// total number of bytes in the HM
	constexpr static size_t total_size = bytes_per_bucket * nrbuckets;

	// NOTE: cannot be multithreaded
	constexpr static bool multithreaded = false;

private:

	/// the max number of elements which can be stored within `bytes_per_bucket` bytes
	using internal_data_load_type = TypeTemplate<bytes_per_bucket>;

	struct internal_data_helper {
		uint8_t bytes[bytes_per_bucket - sizeof(internal_data_load_type)];
		// load in bytes
		internal_data_load_type load;
	};

	struct internal_data {
		union {
		    uint8_t bytes[bytes_per_bucket];
			internal_data_helper load;
		};
	};
public:
	/// constructor. Zero initializing everything
	constexpr SimpleCompressedHashMap() noexcept : __internal_hashmap_array() {}

	/// hashes down `e` (Element) to an index where to store
	/// the element.
	/// NOTE: Boundary checks are performed in debug mode.
	/// \param e element to hash
	/// \param value element to insert
	/// \return nothing
	constexpr inline void insert(const keyType &e,
	                             const data_type &value) noexcept {
		const size_t index = hash(e);
		ASSERT(index < nrbuckets);

		const size_t l = load(index);
		ASSERT(l <= bytes_per_bucket);

		// early exit, if it's already full
		if ((l >= sizeof(data_type)) &&
		    ((l-sizeof(data_type)) >= bytes_per_bucket)) {
			return;
		}

		// get current position within the byte array
		uint8_t *ptr_ = ((uint8_t *)(__internal_hashmap_array + index)) + l;
		data_type v = value;
		if (l > 0) {
			// TODO what happens if the old value is bigger?
			const auto *ptr_2 = (const data_type *)ptr_;
			v -= *ptr_2;
		}
	
		const size_t nl = leb128_encode<data_type>(ptr_, v);
		ASSERT(l+nl <= bytes_per_bucket);

		auto *ptr_3 = (data_type *)(ptr_ + nl);
		*ptr_3 = value;
		set_load(index, l+nl);
	}

	inline void decompress(data_type **out,
	                       uint32_t &nr,
	                       const keyType &e) noexcept {
		// NOTE: we need to choose some upper bound.
		// So this is more or less arbitrary
		static data_type tmp[bytes_per_bucket];

		const size_t index = hash(e);
		const size_t l = load(index);
		auto *pbuf = (uint8_t *)(__internal_hashmap_array + index);
		uint8_t *buf = pbuf;
		nr= 0;
		while (buf < (pbuf + l - sizeof(internal_data_load_type ))) {
			ASSERT(nr < bytes_per_bucket);
			tmp[nr] = leb128_decode<data_type>(&buf);
			nr += 1;
		}
		
		// resolve the diffs
		for (uint32_t j = 1; j < nr; ++j) {
			tmp[j] += tmp[j-1];
		}

		*out = tmp;
	}

	/// \return
	[[nodiscard]] constexpr inline valueType *ptr() noexcept {
		return __internal_hashmap_array;
	}

	/// \param i
	/// \return
	using inner_data_type = typename std::remove_all_extents<data_type>::type;
	using ret_type = typename std::conditional<std::is_bounded_array_v<data_type>,
	                                           inner_data_type *,
	                                           valueType>::type;
	constexpr inline ret_type ptr(const index_type i) noexcept {
		ASSERT(i < total_size);
		if constexpr (std::is_bounded_array_v<data_type>) {
			return (inner_data_type *) __internal_hashmap_array[i];
		} else {
			return (valueType) __internal_hashmap_array[i];
		}
	}

	/// calls ptr
	[[nodiscard]] constexpr inline ret_type operator[](const index_type i) noexcept {
		return ptr(i);
	}

	/// Quite same to `probe` but instead it will directly return
	/// the position of the element.
	/// \param e Element to hash down.
	/// \return the position within the internal const_array of `e`
	[[nodiscard]] constexpr inline index_type find(const keyType &e) const noexcept {
		const index_type index = hash(e);
		ASSERT(index < nrbuckets);
		return index;
	}

	/// \param e
	/// \param __load
	/// \return
	[[nodiscard]] constexpr inline index_type find(const keyType &e,
	                                               load_type &__load) const noexcept {
		const index_type index = hash(e);
		ASSERT(index < nrbuckets);
		__load = load(index);
		// return the index instead of the actual element, to
		// reduce the size of the returned element.
		return index;
	}

	/// \param e
	/// \param __load
	/// \return
	[[nodiscard]] constexpr inline index_type find_without_hash(const keyType &e,
																load_type &__load) const noexcept {
		ASSERT(e < nrbuckets);
		__load = load(e);
		return e;
	}

	/// match the api
	[[nodiscard]] constexpr inline size_t hash(const keyType &e) const noexcept {
		return hashclass(e);
	}

private:
	/// magic function doing all the work
	[[nodiscard]] constexpr inline internal_data_load_type load(const size_t index) const noexcept {
		ASSERT(index < nrbuckets);
		return __internal_hashmap_array[index].load.load;
	}

	/// magic function doing the other half of the work
	constexpr inline void set_load(const size_t index,
			const internal_data_load_type l) noexcept {
		ASSERT(index < nrbuckets);
		__internal_hashmap_array[index].load.load = l;
	}
public:
	/// NOTE: can be called with only a single thread
	/// overwrites the internal data const_array
	/// with zero initialized elements.
	constexpr inline void clear() noexcept {
		memset(__internal_hashmap_array, 0, nrbuckets * bytes_per_bucket);
	}


	/// prints some basic information about the hashmap
	constexpr void info() const noexcept {
		std::cout << "total_size:" << total_size
		          << ", total_size_byts:" << sizeof(__internal_hashmap_array)
		          << ", nrbuckets:" << nrbuckets
		          << ", bytes_per_bucket:" << bytes_per_bucket
		          << ", sizeof(keyType): " << sizeof(keyType)
		          << ", sizeof(valueType): " << sizeof(valueType)
		          << ", sizeof(index_type): " << sizeof(index_type)
				  << ", sizeof(internal_data_load_type): " << sizeof(internal_data_load_type)
		          << ", multithreaded: " << multithreaded;

		std::cout << std::endl;
	}

	// internal const_array
	alignas(1024) internal_data __internal_hashmap_array[nrbuckets];
};

#endif//SMALLSECRETLWE_SIMPLE_H
