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


/// \tparam num_bits The number of bits to rotate.
/// \tparam word_t   The type of number to rotate.
/// \param x The number to be rotated right.
/// \returns The result of right-rotating the bits of x by num_bits.
template <std::size_t num_bits, typename T> 
#if __cplusplus > 201709L
	requires std::is_arithmetic_v<T>
#endif
consteval T rotr(const T x) noexcept {
    return (x >> num_bits) | (x << ((sizeof(T) * 8u) - num_bits));
}

/// \tparam num_bits The number of bits to rotate.
/// \tparam word_t   The type of number to rotate.
/// \param x The number to be rotated left.
/// \returns The result of left-rotating the bits of x by num_bits.
template <std::size_t num_bits, typename T> 
#if __cplusplus > 201709L
	requires std::is_arithmetic_v<T>
#endif
consteval T rotl(const T x) noexcept {
    return (x << num_bits) | (x >> ((sizeof(T) * 8u) - num_bits));
}
#endif
