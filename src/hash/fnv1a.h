#ifndef CRYPTANALYSISLIB_HASH_FNV1A_H
#define CRYPTANALYSISLIB_HASH_FNV1A_H

#ifndef CRYPTANALYSISLIB_HASH_H
#error "do not include this file directly. Use `#inluce <cryptanalysislib/hash/hash.h>`"
#endif

#include <cstdint>
#include <cstdlib>
#include <cassert>

// TODO SIMD version 
// TODO add namespace

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

#endif
