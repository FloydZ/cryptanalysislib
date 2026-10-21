#ifndef CRYPTANALYSISLIB_HASH_FNV1A_H
#define CRYPTANALYSISLIB_HASH_FNV1A_H

#ifndef CRYPTANALYSISLIB_HASH_H
#error "do not include this file directly. Use `#inluce <cryptanalysislib/hash/hash.h>`"
#endif

#include <cstdint>
#include <cstdlib>
#include <cassert>
#include <type_traits>

#include "simd/simd.h"

namespace cryptanalysislib::hash {

/// Computes a FNV-1 hash for the provided data
/// Implementation based on: https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function#FNV-1_hash
/// Supports 32-bit and 64-bit hash values
/// 
/// \tparam T Type of the hash value (default: uint64_t)
/// \param data[in] Pointer to the input data
/// \param size[in] Size of the input data in bytes
/// \return Computed FNV-1 hash value
template<typename T=uint64_t>
[[nodiscard]] constexpr T fnv1(const uint8_t *data, const size_t size) noexcept {
	if constexpr (sizeof(T) == 4) {
		T hash = 0x811c9dc5;
		for (size_t i = 0; i < size; ++i) {
			hash *= 0x01000193;
			hash ^= data[i];
		}
		return hash;
	} else if constexpr (sizeof(T) == 8) {
		T hash = 0xcbf29ce484222325;
		for (size_t i = 0; i < size; ++i) {
			hash *= 0x100000001b3;
			hash ^= data[i];
		}
		return hash;
	} else {
        assert(0);
        return 0;
	}
}

/// Computes a FNV-1a hash for the provided data
/// Implementation based on: https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function#FNV-1a_hash
/// This is the alternative version of FNV-1 with better avalanche characteristics
/// Supports 32-bit and 64-bit hash values
/// 
/// \tparam T Type of the hash value (default: uint64_t)
/// \param data[in] Pointer to the input data
/// \param size[in] Size of the input data in bytes
/// \return Computed FNV-1a hash value
template<typename T=uint64_t>
[[nodiscard]] constexpr T fnv1a(const uint8_t *data, const size_t size) noexcept {
	if constexpr (sizeof(T) == 4) {
		T hash = 0x811c9dc5;
		for (size_t i = 0; i < size; ++i) {
			hash ^= data[i];
			hash *= 0x01000193;
		}
		return hash;
	} else if constexpr (sizeof(T) == 8) {
		T hash = 0xcbf29ce484222325;
		for (size_t i = 0; i < size; ++i) {
			hash ^= data[i];
			hash *= 0x100000001b3;
		}
		return hash;
	} else {
        assert(0);
        return 0;
	}
}

/// SIMD version of `fnv1a`: hashes `nr` messages of `size` bytes each, the
/// `i`-th starts at `data + i*size`: `out[i] = fnv1a<T>(data + i*size, size)`.
/// The hash is sequential per message, so 8 (uint32_t) or 4 (uint64_t)
/// messages are hashed at once, one in each SIMD lane. Every gather loads
/// `sizeof(T)` bytes of each message.
/// NOTE: for uint32_t `nr*size` must be < 2**31 (signed 32 bit gather offsets)
/// \tparam T Type of the hash values (uint32_t or uint64_t)
/// \param out[out] `nr` hash values
/// \param data[in] the messages
/// \param size[in] size of each message in bytes
/// \param nr[in] number of messages
template<typename T=uint64_t>
	requires (sizeof(T) == 4) || (sizeof(T) == 8)
static inline void fnv1a_multi(T *out,
                               const uint8_t *data,
                               const size_t size,
                               const size_t nr) noexcept {
	using S = std::conditional_t<sizeof(T) == 4, uint32x8_t, uint64x4_t>;
	using L = typename S::limb_type;
	constexpr size_t lanes = S::LIMBS;
	constexpr uint32_t word = sizeof(T);
	constexpr T basis = (sizeof(T) == 4) ? T(0x811c9dc5u) : T(0xcbf29ce484222325ull);
	constexpr T prime = (sizeof(T) == 4) ? T(0x01000193u) : T(0x100000001b3ull);
	assert((sizeof(T) == 8) || ((nr * size) < (1ull << 31u)));

	const S p = S::set1(prime);
	const S m = S::set1(0xffu);
	// byte offsets of the messages of the current block
	S off;
	for (size_t k = 0; k < lanes; ++k) { off[k] = L(k * size); }
	const S step = S::set1(L(lanes * size));

	size_t i = 0;
	for (; i + lanes <= nr; i += lanes) {
		S h = S::set1(basis);
		size_t j = 0;
		for (; j + word <= size; j += word) {
			const S w = S::template gather<1>(data + j, off);
			for (uint32_t b = 0; b < word; ++b) {
				h = S::mullo(S::xor_(h, S::and_(S::srli(w, 8u * b), m)), p);
			}
		}

		// the last `size % word` bytes: a gather could read past the buffer
		for (; j < size; ++j) {
			S x = S::set1(0);
			for (size_t k = 0; k < lanes; ++k) {
				x[k] = data[(i + k) * size + j];
			}
			h = S::mullo(S::xor_(h, x), p);
		}

		S::unaligned_store((L *)(out + i), h);
		off = S::add(off, step);
	}

	for (; i < nr; ++i) {
		out[i] = fnv1a<T>(data + i * size, size);
	}
}

} // end namespace cryptanalysislib::hash
#endif
