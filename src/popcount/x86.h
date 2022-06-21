#ifndef CRYPTANALYSISLIB_POPCOUNT_X86_H
#define CRYPTANALYSISLIB_POPCOUNT_X86_H

#include "helper.h"

struct popcount_config {
public:

	// alignment in bytes
	const uint32_t alignment;

	constexpr popcount_config(const uint32_t alignment) noexcept :
		alignment(alignment) {}
};

class popcount {
private:
	constexpr static popcount_config config{16};

public:
	/// delete the default construct/deconstructor
	popcount() = delete;
	~popcount() = delete;


	///
	/// \tparam T
	/// \param data
	/// \return
	template<class T>
		requires std::integral<T>
	constexpr static uint32_t count(T data) noexcept {
		if constexpr(sizeof(T) < 8) {
			return __builtin_popcount(data);
		} else if constexpr(sizeof(T) == 8) {
			return  __builtin_popcountll(data);
		} else {
			throw ;
		}
	}

	///
	/// \tparam T
	/// \param data
	/// \param size
	/// \return
	template<class T, popcount_config config = config>
		requires std::integral<T>
	constexpr static uint32_t count(T *__restrict__ data, const size_t size) noexcept {
		// ASSERT(uint64_t(data) % config.alignment);
		uint32_t sum = 0;
		for (size_t i = 0; i < size; ++i) {
			sum += count(data[i]);
		}

		return sum;
	}
};
#endif
