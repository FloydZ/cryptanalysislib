#ifndef CRYPTANALYSISLIB_SIMPLE_H
#define CRYPTANALYSISLIB_SIMPLE_H

#include <cstdint>
#include <type_traits>
#include <stdlib.h>


template<typename T=uint64_t, const uint32_t l=0, const uint32_t h=8*sizeof(T)>
requires std::is_integral<T>::value
class Hash {
public:
	constexpr inline size_t operator()(const T &k) const noexcept {
		constexpr T mask1 = ~((1u << l) - 1u);
		constexpr T mask2 = (1u << h) - 1u;
		constexpr T mask = mask1 & mask2;
		return (k & mask) >> l;
	}
};

// TODO rename to Hash und concept fuer `ptr`
template<typename L, const uint32_t l, const uint32_t h>
class HashD {
public:
	constexpr inline size_t operator()(const L &k) const noexcept {
		constexpr __uint128_t mask1 = ~((1u << l) - 1u);
		constexpr __uint128_t mask2 = (1u << h) - 1u;
		constexpr __uint128_t mask = mask1 & mask2;
		return ((*(__uint128_t *)k.ptr()) & mask) >> l;
	}
};
#endif//CRYPTANALYSISLIB_SIMPLE_H
