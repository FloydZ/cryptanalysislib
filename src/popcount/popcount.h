#ifndef CRYPTANALYSISLIB_POPCOUNT_H
#define CRYPTANALYSISLIB_POPCOUNT_H

#include <cstdint>
#include "helper.h"


/// namespace containing popcount algorithms
namespace cryptanalysislib::popcount {

	/// specialized avx2 popcount instructions
	namespace internal {
#ifdef USE_AVX2
    #include "popcount/avx2.h"
#endif
	};

	///
	/// \tparam T base data type
	/// \param data input data type
	/// \return hamming weight (popcount) of the input vector
	template<class T>
#if __cplusplus > 201709L
		requires std::integral<T>
#endif
	constexpr inline uint32_t popcount(const T data) noexcept {
		if constexpr(sizeof(T) < 8) {
			return __builtin_popcount(data);
		} else if constexpr(sizeof(T) == 8) {
			return  __builtin_popcountll(data);
		} else {
			ASSERT(false);
		}
	}

	/// \tparam T base type
	/// \param data pointer to the array
	/// \param size number of elements in the array
	/// \return hamming weight (popcount) of the input vector
	template<class T>
#if __cplusplus > 201709L
		requires std::integral<T>
#endif
	constexpr uint64_t popcount(const T *__restrict__ data, const size_t size) noexcept {
		uint32_t sum = 0;
		for (size_t i = 0; i < size; ++i) {
			sum += popcount<T>(data[i]);
		}

		return sum;
	}
};
#endif//CRYPTANALYSISLIB_POPCOUNT_H
