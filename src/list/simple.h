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
/// load factor is not multithreaded
/// \tparam Element, can be anything.
///		Does not need to be a `Element`
/// 	that's the reason, why we have a single concept only for this class
template<class Element, const size_t __size>
#if __cplusplus > 201709L
requires ParallelListSimpleAble<Element>
#endif
class ParallelListSimple {
private:
	uint32_t __threads;
	size_t __thread_block_size;
	size_t __load;

public:
	using ElementType = Element;

	/// well technically that's not correct, but we fake it, as some other
	/// data structures need it, like `ListEnumerationMeta`
	using ValueType = Element;
	using LabelType = Element;
	using MatrixType = Element;

	/// \param thread_block number of elements per thread
	constexpr  ParallelListSimple(const uint32_t threads=1) noexcept
	    : __threads(threads), __thread_block_size(__size/threads), __load(0) {
	}


	/// \return size the size of the list
	[[nodiscard]] constexpr size_t size() const noexcept { return __size; }
	/// \return the number of elements each thread enumerates
	[[nodiscard]] constexpr size_t size(const uint32_t tid) const noexcept {
		if (tid == __threads-1) {
			return std::max(__thread_block_size*__threads, __size);
		}

		return __thread_block_size;
	}

	/// set the size
	/// NOTE: not implementable as this list has a static size
	constexpr void set_size(const size_t new_size) noexcept {
		(void)new_size;
	}

	/// get the load parameter
	[[nodiscard]] constexpr size_t load() const noexcept { return __load; }

	/// set the internal load factor
	/// \param l new load
	void set_load(const size_t l) noexcept {
		ASSERT(l <= __size);
		__load = l;
	}

	/// some setter/getter
	[[nodiscard]] uint32_t threads() const noexcept { return __threads; }
	[[nodiscard]] size_t thread_block_size() const noexcept { return __thread_block_size; }
	void set_threads(const uint32_t new_threads) noexcept {
		__threads = new_threads;
		__thread_block_size = size()/__threads;
	}
	void set_thread_block_size(const size_t a) noexcept {  __thread_block_size = a; }

	/// Get a const pointer. Sometimes useful if one ones to tell the kernel how to access memory.
	constexpr inline auto* data() noexcept{ return __data.data(); }
	const auto* data() const noexcept { return __data.data(); }

	/// wrapper
	constexpr inline std::nullptr_t* data_value() noexcept { return nullptr; }
	constexpr inline const std::nullptr_t* data_value() const noexcept { return nullptr;}
	constexpr inline Element* data_label() noexcept { return __data.data(); }
	constexpr inline const Element* data_label() const noexcept { return __data.data(); }

	constexpr inline Element& data_value(const size_t i) noexcept {  ASSERT(i < __size); return __data[i]; }
	constexpr inline const Element& data_value(const size_t i) const noexcept { ASSERT(i < __size); return __data[i]; }
	constexpr inline Element& data_label(const size_t i) noexcept { ASSERT(i < __size); return __data[i]; }
	constexpr inline const Element& data_label(const size_t i) const noexcept { ASSERT(i < __size); return __data[i]; }


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
		ASSERT(tid < __threads);
		return tid*__thread_block_size;
	};

	/// NOTE: boundary checks are done
	/// returning the starting position in which one thread is allowed to operate
	/// \param tid thread id
	/// \return
	[[nodiscard]] constexpr inline size_t end_pos(const uint32_t tid) const noexcept {
		ASSERT(tid < __threads);
		return( tid+1)*__thread_block_size;
	};

	/// print the `pos` element
	/// 	label between [label_k_lower, label_k_upper)
	/// 	value between [value_k_lower, value_k_upper)
	/// \param pos position of the element in the list to print
	/// \param value_k_lower inclusive
	/// \param value_k_higher exclusive
	/// \param label_k_lower inclusive
	/// \param label_k_higher exclusive
	void print_binary(const uint64_t pos,
					  const uint32_t value_k_lower,
					  const uint32_t value_k_higher,
					  const uint32_t label_k_lower,
					  const uint32_t label_k_higher) const noexcept {
		ASSERT(value_k_lower < value_k_higher);
		ASSERT(label_k_lower < label_k_higher);

		data_label(pos).print_binary(label_k_lower, label_k_higher);
	}

	/// print the `pos` element
	/// 	label between [label_k_lower, label_k_upper)
	/// 	value between [value_k_lower, value_k_upper)
	/// \param pos position of the element in the list to print
	/// \param value_k_lower inclusive
	/// \param value_k_higher exclusive
	/// \param label_k_lower inclusive
	/// \param label_k_higher exclusive
	void print(const uint64_t pos,
			   const uint32_t value_k_lower,
			   const uint32_t value_k_higher,
			   const uint32_t label_k_lower,
			   const uint32_t label_k_higher) const noexcept {
		ASSERT(value_k_lower < value_k_higher);
		ASSERT(label_k_lower < label_k_higher);

		data_label(pos).print(label_k_lower, label_k_higher);
	}

	/// print the element between [start, end) s.t.:
	/// 	label between [label_k_lower, label_k_upper)
	/// 	value between [value_k_lower, value_k_upper)
	/// \param pos position of the element in the list to print
	/// \param value_k_lower inclusive
	/// \param value_k_higher exclusive
	/// \param label_k_lower inclusive
	/// \param label_k_higher exclusive
	void print(const uint32_t value_k_lower,
			   const uint32_t value_k_higher,
			   const uint32_t label_k_lower,
			   const uint32_t label_k_higher,
			   const size_t start,
			   const size_t end) const noexcept {
		ASSERT(start < end);
		ASSERT(end <= __data.size());
		ASSERT(value_k_lower < value_k_higher);
		ASSERT(label_k_lower < label_k_higher);

		for (size_t i = start; i < end; ++i) {
			print(i, value_k_lower, value_k_higher,
				  label_k_lower, label_k_higher);
		}
	}

	/// print the element binary between [start, end) s.t.:
	/// 	label between [label_k_lower, label_k_upper)
	/// 	value between [value_k_lower, value_k_upper)
	/// \param pos position of the element in the list to print
	/// \param value_k_lower inclusive
	/// \param value_k_higher exclusive
	/// \param label_k_lower inclusive
	/// \param label_k_higher exclusive
	void print_binary(const uint32_t value_k_lower,
					  const uint32_t value_k_higher,
					  const uint32_t label_k_lower,
					  const uint32_t label_k_higher,
					  const size_t start,
					  const size_t end) const noexcept {
		ASSERT(start < end);
		ASSERT(end <= __data.size());
		ASSERT(value_k_lower < value_k_higher);
		ASSERT(label_k_lower < label_k_higher);

		for (size_t i = start; i < end; ++i) {
			print_binary(i, value_k_lower, value_k_higher,
						 label_k_lower, label_k_higher);
		}
	}

	/// zeros all elements
	/// \param tid
	constexpr void zero(const uint32_t tid=0) noexcept {
		ASSERT(tid < __threads);
		for (size_t i = start_pos(tid); i < end_pos(tid); ++i) {
			__data[i].zero();
		}
	}

	/// zeros a single element
	/// \param i
	constexpr void random(const size_t i) noexcept {
		ASSERT(i < __load);
		__data[i].random();
	}

	/// zeros a single element
	/// \param i
	constexpr void random() noexcept {
		for (size_t i = 0; i < __size; i++) {
			__data[i].random();
		}
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

	/// insert an element into the list past the load factor
	/// \param e element to insert
	/// \param pos is a relative position to the thread id
	/// \param tid thread id
	constexpr void insert(const Element &e, const size_t pos, const uint32_t tid=0) noexcept {
		const size_t spos = start_pos(tid);
		__data[spos + pos] = e;
	}

private:
	alignas(CUSTOM_PAGE_SIZE) std::array<Element, __size> __data;
};
#endif//CRYPTANALYSISLIB_LIST_SIMPLE_H
