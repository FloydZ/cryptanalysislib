#ifndef CRYPTANALYSISLIB_ALGORITH_ROTATE_H
#define CRYPTANALYSISLIB_ALGORITH_ROTATE_H

#include <cstdint>
#include <type_traits>

/// left rotate
/// \param x value to rotate
/// \param k how much to rotate
/// \return x <<< k
template<typename T=uint64_t>
#if __cplusplus > 201709L
	requires std::is_arithmetic_v<T>
#endif
[[nodiscard]] constexpr static inline uint64_t rotl(const T x,
		                                            const uint32_t k) noexcept {
	return (x << k) | (x >> ((sizeof(T)*8) - k));
}

/// right rotate
/// \param x value to rotate
/// \param k how much to rotate
/// \return x <<< k
template<typename T=uint64_t>
#if __cplusplus > 201709L
	requires std::is_arithmetic_v<T>
#endif
[[nodiscard]] constexpr static inline uint64_t rotr(const T x, 
                                                    const uint32_t k) {
    return (x >> k) | (x << ((-k) & ((sizeof(T)*8)-1u)));
}
#endif
