#ifndef CRYPTANALYSISLIB_POPCOUNT_H
#define CRYPTANALYSISLIB_POPCOUNT_H

#include <cstdint>
#include <type_traits>

#include "helper.h"

#ifdef USE_AVX2
#include "./popcount/avx2.h"
#endif

/// namespace containing popcount algorithms
namespace cryptanalysislib::popcount {
	/// Count the number of set bits in an integer (Population Count)
	/// \tparam T Base integer data type
	/// \param data [in]: Input value to count the set bits in
	/// \return [out]: Number of set bits (Hamming weight)
	template<typename T>
#if __cplusplus > 201709L
		requires std::is_integral<T>::value
#endif
	constexpr inline uint32_t popcount(const T data) noexcept {
		if constexpr(sizeof(T) < 8) {
			return __builtin_popcountl(data);
		} else if constexpr(sizeof(T) == 8) {
			return  __builtin_popcountll(data);
		} else if constexpr(sizeof(T) == 16) {
			return  __builtin_popcountll((uint64_t )data) +
					__builtin_popcountll(data >> 64u);
		} else {
			assert(false);
            return 0;
		}
	}

	/// Count the number of set bits in an array of integers
	/// \tparam T Base integer data type
	/// \param data [in]: Pointer to the array of integers
	/// \param size [in]: Number of elements in the array
	/// \return [out]: Total number of set bits (Hamming weight) across all elements
	template<class T>
#if __cplusplus > 201709L
		requires std::is_integral<T>::value
#endif
	constexpr uint64_t popcount(const T *__restrict__ data, 
						  		const size_t size) noexcept {
		uint32_t sum = 0;
		for (size_t i = 0; i < size; ++i) {
			sum += popcount<T>(data[i]);
		}

		return sum;
	}
}

#endif//CRYPTANALYSISLIB_POPCOUNT_H
