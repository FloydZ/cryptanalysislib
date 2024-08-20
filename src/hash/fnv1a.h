#ifndef CRYPTANALYSISLIB_HASH_FNV1A_H
#define CRYPTANALYSISLIB_HASH_FNV1A_H

#ifndef CRYPTANALYSISLIB_HASH_H
#error "do not include this file directly. Use `#inluce <cryptanalysislib/hash/hash.h>`"
#endif

#include <cstdint>
#include <cstdlib>

/// Source: https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function#FNV-1_hash
/// \param data
/// \param size
/// \return
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
	} else if constexpr (sizeof(T) == 16) {
		return 0;
	} else {
		return 0;
	}

}
/// Source: https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function#FNV-1a_hash
/// \param data
/// \param size
/// \return
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
	} else if constexpr (sizeof(T) == 16) {
		return 0;
	} else {
		return 0;
	}
}
#endif
