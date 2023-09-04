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
	requires requires(const unsigned int i) {
		c[i];
		c.zero();
	};
};
#endif

/// most simple list
/// \tparam Element
template<class Element, const size_t size>
#if __cplusplus > 201709L
requires ParallelListSimpleAble<Element>
#endif
class ParallelListSimple {
private:
	uint32_t thread_block;
	size_t load;

public:
	typedef Element ElementType;

	///
	/// \param thread_block
	ParallelListSimple(const uint32_t thread_block=size) noexcept {
		this->thread_block = thread_block;
		this->load = 0;
	}

	///
	/// \param tid
	void clear(const uint32_t tid=0) noexcept {
		for (size_t i = start_pos(tid); i < end_pos(tid); ++i) {
			__data[i].clear();
		}
	}

	///
	/// \return
	size_t get_load() const noexcept { return load; }

	///
	/// \param l
	void set_load(const size_t l) noexcept { load = l; }

	///
	/// \param i
	/// \return
	Element &operator[](const size_t i) noexcept {
		ASSERT(i < load);
		return this->__data[i];
	}

	///
	/// \param i
	/// \return
	const Element &operator[](const size_t i) const noexcept {
		ASSERT(i < load);
		return this->__data[i];
	}

	/// returning the range in which one thread is allowed to operate
	/// \param tid
	/// \return
	inline uint64_t start_pos(const uint32_t tid) const noexcept { return tid*thread_block; };

	///
	/// \param tid
	/// \return
	inline uint64_t end_pos(const uint32_t tid) const noexcept { return( tid+1)*thread_block; };

	alignas(1024) std::array<Element, size> __data;
};
#endif//CRYPTANALYSISLIB_LIST_SIMPLE_H
