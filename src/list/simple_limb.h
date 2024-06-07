#ifndef CRYPTANALYSISLIB_LIST_SIMPLE_LIMB_H
#define CRYPTANALYSISLIB_LIST_SIMPLE_LIMB_H

#include <cstdint>
#include <cstddef>

#include "helper.h"

/// most simple list
/// \tparam T: must be a
template<class T, const size_t __size>
#if __cplusplus > 201709L
	requires std::is_integral_v<T>
#endif
class ParallelListSimpleLimb {
private:
	uint32_t threads;
	uint32_t thread_block;
	size_t load;

public:
	using ElementType = T;

	/// \param thread_block number of elements per thread
	constexpr  ParallelListSimpleLimb(const uint32_t threads=1) noexcept
	    : threads(threads), thread_block(__size/threads), load(0) {
	}

	/// get the load parameter
	[[nodiscard]] constexpr size_t get_load() const noexcept { return load; }

	/// set the internal load factor
	/// \param l new load
	constexpr void set_load(const size_t l) noexcept {
		ASSERT(l <= __size);
		load = l;
	}

	/// \return size the size of the list
	[[nodiscard]] constexpr inline size_t size() const noexcept { return __size; }
	/// \return the number of elements each thread enumerates
	[[nodiscard]] constexpr inline size_t size(const uint32_t tid) const noexcept {
		ASSERT(tid < threads);
		if (tid == (threads - 1)) {
			return thread_block + (__size - (threads*thread_block));
		}

		return thread_block; 
	}

	/// NOTE: boundary checks are done
	///
	[[nodiscard]] constexpr inline ElementType &at(const size_t i) noexcept {
		ASSERT(i < size());
		return this->__data[i];
	}

	/// NOTE: boundary checks are done
	///
	[[nodiscard]] constexpr inline const ElementType &at(const size_t i) const noexcept {
		ASSERT(i <size());
		return this->__data[i];
	}

	/// NOTE: boundary checks are done
	/// \param i
	/// \return the i-th element in the list
	[[nodiscard]] ElementType &operator[](const size_t i) noexcept {
		ASSERT(i < __size);
		return __data[i];
	}

	/// NOTE: boundary checks are done
	/// \param i
	/// \return the i-th elementin the list
	[[nodiscard]] const ElementType &operator[](const size_t i) const noexcept {
		ASSERT(i < __size);
		return this->__data[i];
	}

	/// NOTE: boundary checks are done
	/// returning the starting position in which one thread is allowed to operate
	/// \param tid thread id
	/// \return
	[[nodiscard]] constexpr inline size_t start_pos(const uint32_t tid) const noexcept {
		ASSERT(tid < threads);
		return tid*thread_block;
	};

	/// NOTE: boundary checks are done
	/// returning the starting position in which one thread is allowed to operate
	/// \param tid thread id
	/// \return
	[[nodiscard]] constexpr inline size_t end_pos(const uint32_t tid) const noexcept {
		ASSERT(tid < threads);
		return (tid+1)*thread_block;
	};

	/// zeros all elements
	/// \param tid
	constexpr void zero(const uint32_t tid=0) noexcept {
		ASSERT(tid < threads);
		for (size_t i = start_pos(tid); i < end_pos(tid); ++i) {
			__data[i].zero();
		}
	}

	/// zeros a single element
	/// \param i
	constexpr void zero_element(const size_t i) noexcept {
		ASSERT(i < load);
		__data[i].zero();
	}

	/// not implemented
	constexpr void sort() noexcept {
		ASSERT(0);
	}

	/// some useful stuff
	auto begin() noexcept { return __data.begin(); }
	auto end() noexcept { return __data.end(); }

	/// returns the size in bytes
	[[nodiscard]] __FORCEINLINE__ constexpr uint64_t bytes() noexcept { 
		return __size*sizeof(T); 
	}

	alignas(CUSTOM_PAGE_SIZE) std::array<T, __size> __data;
};

#endif//CRYPTANALYSISLIB_LIST_SIMPLE_LIMB_H
