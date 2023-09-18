#ifndef CRYPTANALYSISLIB_LIST_SIMPLE_H
#define CRYPTANALYSISLIB_LIST_SIMPLE_H

#include <cstdint>
#include <cstddef>

#include "helper.h"

#if __cplusplus > 201709L
///
/// \tparam Container
template<class Container>
concept ParallelListSimpleAble = requires(Container c) {
	requires requires(const uint32_t i) {
		c[i];
		c.zero();
	};
};
#endif

/// most simple list
/// \tparam Element, can be anything.
///		Does not need to be a `Element`
/// 	that's the reason, why we have a single concept only for this class
template<class Element, const size_t __size>
#if __cplusplus > 201709L
requires ParallelListSimpleAble<Element>
#endif
class ParallelListSimple {
private:
	uint32_t threads;
	uint32_t thread_block;
	size_t load;

public:
	using ElementType = Element;

	/// \param thread_block number of elements per thread
	constexpr  ParallelListSimple(const uint32_t threads=1) noexcept
	    : threads(threads), thread_block(__size/threads), load(0) {
	}

	/// get the load parameter
	[[nodiscard]] constexpr size_t get_load() const noexcept { return load; }

	/// set the internal load factor
	/// \param l new load
	void set_load(const size_t l) noexcept {
		ASSERT(l <= __size);
		load = l;
	}

	/// \return size the size of the list
	[[nodiscard]] constexpr inline size_t size() const noexcept { return size; }
	/// \return the number of elements each thread enumerates
	[[nodiscard]] constexpr inline size_t size(const uint32_t tid) const noexcept { return thread_block; }

	constexpr inline Element &at(const size_t i) noexcept {
		ASSERT(i < size());
		return this->__data[i];
	}
	constexpr inline const Element &at(const size_t i) const noexcept {
		ASSERT(i <size());
		return this->__data[i];
	}

	/// NOTE: boundary checks are done
	/// \param i
	/// \return the i-th element in the list
	Element &operator[](const size_t i) noexcept {
		ASSERT(i < __size);
		return __data[i];
	}

	/// NOTE: boundary checks are done
	/// \param i
	/// \return the i-th elementin the list
	const Element &operator[](const size_t i) const noexcept {
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
		return( tid+1)*thread_block;
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
	[[nodiscard]] __FORCEINLINE__ constexpr uint64_t bytes() noexcept { return __size*sizeof(Element); }

	alignas(PAGE_SIZE) std::array<Element, __size> __data;
};
#endif//CRYPTANALYSISLIB_LIST_SIMPLE_H