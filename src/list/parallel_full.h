#ifndef DECODING_LIST_PARALLEL_FULL_H
#define DECODING_LIST_PARALLEL_FULL_H

#include <iterator>
#include <vector>           // main data container
#include <algorithm>        // search/find routines
#include <cassert>

#include "list/common.h"


/// same the class
///		`Parallel_List_T`
/// with the big difference that Elements = <Label, Value> are stored within the same array.
/// Additionally the thread parallel access is further implemented
/// In comparison to different approach this dataset does not track its load factors. It leaves this work completely to
///		the calling function.
template<class Element>
#if __cplusplus > 201709L
requires ListElementAble<Element>
#endif
class Parallel_List_FullElement_T {
private:
	// disable the empty constructor. So you have to specify a rough size of the list. This is for optimisations reasons.
	Parallel_List_FullElement_T() : nr_elements(0), thread_block(0), threads(1) {};

public:
	typedef Element ElementType;
	typedef typename ElementType::ValueType ValueType;
	typedef typename ElementType::LabelType LabelType;

	typedef typename ElementType::ValueContainerType ValueContainerType;
	typedef typename ElementType::LabelContainerType LabelContainerType;

	typedef typename ElementType::ValueContainerLimbType ValueContainerLimbType;
	typedef typename ElementType::LabelContainerLimbType LabelContainerLimbType;

	typedef typename ElementType::ValueDataType ValueDataType;
	typedef typename ElementType::LabelDataType LabelDataType;

	typedef typename ElementType::MatrixType MatrixType;

	using LoadType = uint64_t;

	/// internal data types lengths
	constexpr static uint32_t ValueLENGTH = ValueType::LENGTH;
	constexpr static uint32_t LabelLENGTH = LabelType::LENGTH;

	/// size in bytes
	constexpr static uint64_t element_size = Element::ssize();
	constexpr static uint64_t value_size = ValueType::ssize();
	constexpr static uint64_t label_size = LabelType::ssize();

	// somehow make such flags configurable
	constexpr static bool USE_STD_SORT = false;

	// default deconstructor
	~Parallel_List_FullElement_T() = default;

	/// simple single threaded constructor
	constexpr explicit Parallel_List_FullElement_T(const size_t size) noexcept :
			nr_elements(size), thread_block(size), threads(1)
	{
		__data.resize(size);
	}

	/// multithreaded constructor
	constexpr explicit Parallel_List_FullElement_T(const size_t size,
										 const uint32_t threads,
										 const size_t thread_block) noexcept :
			nr_elements(size),
			thread_block(thread_block),
	        threads(threads)
	{
		__data.resize(size);
	}


	/// \return size the size of the list
	[[nodiscard]] constexpr inline size_t size() const noexcept { return nr_elements; }
	/// \return the number of elements each thread enumerates
	[[nodiscard]] constexpr inline size_t size(const uint32_t tid) const noexcept { return thread_block; }

	/// Useless function, as this list class does not track its load
	[[nodiscard]] constexpr size_t get_load() const noexcept { return 0; }
	constexpr void set_load(const size_t l) noexcept { (void)l; }

	/// returning the range in which one thread is allowed to operate
	[[nodiscard]] constexpr inline size_t start_pos(const uint32_t tid) const noexcept { return tid*thread_block; };
	[[nodiscard]] constexpr inline size_t end_pos(const uint32_t tid) const noexcept { return( tid+1)*thread_block; };

	/// \return different pointers to value and label
	inline ValueType* data_value() noexcept { return (ValueType *)__data.data(); }
	inline const ValueType* data_value() const noexcept { return (ValueType *)__data.data(); }
	inline LabelType* data_label() noexcept { return (LabelType *)__data.data(); }
	inline const LabelType* data_label() const noexcept { return (LabelType *)__data.data(); }
	inline Element * data() noexcept{ return __data.data(); }
	inline const Element* data() const noexcept { return __data.data(); }

	/// \return difference references to value and label
	inline ValueType& data_value(const size_t i) noexcept {  ASSERT(i < nr_elements); return __data[i].get_value(); }
	inline const ValueType& data_value(const size_t i) const noexcept { ASSERT(i < nr_elements); return __data[i].get_value(); }
	inline LabelType& data_label(const size_t i) noexcept { ASSERT(i < nr_elements); return __data[i].get_label(); }
	inline const LabelType& data_label(const size_t i) const noexcept { ASSERT(i < nr_elements); return __data[i].get_label(); }
	inline Element &data(const size_t i) noexcept { ASSERT(i < size()); return this->__data[i]; }
	inline const Element& data(const size_t i) const noexcept { ASSERT(i < size()); return this->__data[i];  }

	/// operator overloading
	constexpr inline Element &at(const size_t i) noexcept {
		ASSERT(i < size());
		return this->__data[i];
	}
	constexpr inline const Element &at(const size_t i) const noexcept {
		ASSERT(i <size());
		return this->__data[i];
	}
	inline Element &operator[](const size_t i) noexcept { ASSERT(i < size()); return this->__data[i]; }
	const inline Element &operator[](const size_t i) const noexcept {ASSERT(i < size()); return this->__data[i]; }

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
		ASSERT(value_k_higher <= ValueLENGTH);
		ASSERT(label_k_lower < label_k_higher);
		ASSERT(label_k_higher <= LabelLENGTH);

		data_value(pos).print_binary(value_k_lower, value_k_higher);
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
		ASSERT(value_k_higher <= ValueLENGTH);
		ASSERT(label_k_lower < label_k_higher);
		ASSERT(label_k_higher <= LabelLENGTH);

		data_value(pos).print(value_k_lower, value_k_higher);
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
		ASSERT(end <= nr_elements);
		ASSERT(value_k_lower < value_k_higher);
		ASSERT(value_k_higher <= ValueLENGTH);
		ASSERT(label_k_lower < label_k_higher);
		ASSERT(label_k_higher <= LabelLENGTH);

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
		ASSERT(end <= nr_elements);
		ASSERT(value_k_lower < value_k_higher);
		ASSERT(value_k_higher <= ValueLENGTH);
		ASSERT(label_k_lower < label_k_higher);
		ASSERT(label_k_higher <= LabelLENGTH);

		for (size_t i = start; i < end; ++i) {
			print_binary(i, value_k_lower, value_k_higher,
						 label_k_lower, label_k_higher);
		}
	}

#ifdef  _OPENMP
	/// src: https://www.cs.rutgers.edu/~venugopa/parallel_summer2012/bitonic_openmp.html
	/// needs to be benchmarked against std::sort()
	/// \param start 	start position in `seq` to start from
	/// \param length 	number of elements to sort
	/// \param seq 		array to sort
	/// \param flag 	sorting direction.
	void bitonic_sort_par(int start, int length, int *seq, int flag) {
		int i;
		int split_length;

		if (length == 1)
			return;

		if (length % 2 !=0 ) {
			printf("The length of a (sub)sequence is not divided by 2.\n");
			exit(0);
		}

		split_length = length / 2;

		// bitonic split
#pragma omp parallel for default(none) shared(seq, flag, start, split_length) private(i)
		for (i = start; i < start + split_length; i++) {
			if (flag == 1) {
				if (seq[i] > seq[i + split_length])
					std::swap(seq[i], seq[i + split_length]);
			}
			else {
				if (seq[i] < seq[i + split_length])
					std::swap(seq[i], seq[i + split_length]);
			}
		}


		// if (split_length > m) {
		// m is the size of sub part-> n/numThreads
		if (threads > 2) {
			bitonic_sort_par(start, split_length, seq, flag);
			bitonic_sort_par(start + split_length, split_length, seq, flag);
		}
	}
#endif

	constexpr void sort() noexcept {
		ASSERT(0);
	}

	/// generic hash function
	/// \tparam Hash a function/lambda which hashes an element of the list down to a comparable number
	/// 			[](const Label &l) -> uint32_t { return l.data().data()[0]; }
	/// \param s	start position to start sorting
	/// \param e	upper bound of the sorting algorithm
	/// \param hash
	template<typename Hash>
	void sort(LoadType s, LoadType e, Hash hash) noexcept {
		ska_sort(__data.begin() + s,
				 __data.begin() + e,
				 hash
		);
	}

	/// the same as the function above but it will sort on all coordinates.`
	/// \tparam Hash
	/// \param tid thread id
	/// \param hash hash function
	template<typename Hash>
	void sort(uint32_t tid, Hash hash, const size_t size=-1) noexcept {
		ASSERT(tid < threads);

		const auto e = size == size_t(-1) ? end_pos(tid) : size;
		sort(start_pos(tid), e, hash);
	}

	/// \param i lower coordinate in the label used as the sorting index
	/// \param j upper   .....
	void sort_level(const uint32_t i, const uint32_t j) noexcept {
		ASSERT(i < j);
		using T = LabelContainerType;

		//constexpr uint64_t upper = T::round_down_to_limb(j);
		constexpr uint64_t lower = T::round_down_to_limb(i);
		const uint64_t mask = T::higher_mask(i) & T::lower_mask(j);
		if constexpr (USE_STD_SORT) {
			std::sort(__data.begin(),
					  __data.end(),
					  [lower, mask](const auto &e1, const auto &e2) {
						return e1.get_label().data().is_lower_simple2(e2.get_label().data(), lower, mask);
					  }
			);
		} else {
			ska_sort(__data.begin(),
					 __data.end(),
					 [lower, mask](const Element &e) {
					   return e.get_label_container_ptr()[lower] & mask;
					 }
			);
		}

	}

	/// \param i lower coordinate in the label used as the sorting index
	/// \param j upper   .....
	/// \param tid thread id
	void sort_level(const uint32_t i, const uint32_t j, const uint32_t tid) noexcept {
		ASSERT(i < j);
		using T = LabelContainerType;

		const uint64_t lower = T::round_down_to_limb(i);
		//const uint64_t upper = T::round_down_to_limb(j);
		//ASSERT(lower == upper); // TODO

		const uint64_t mask = T::higher_mask(i) & T::lower_mask(j);

		if constexpr (USE_STD_SORT) {
			std::sort(__data.begin() + start_pos(tid),
					  __data.begin() + end_pos(tid),
					  [lower, mask](const auto &e1, const auto &e2) {
						return e1.get_label().data().is_lower_simple2(e2.get_label().data(), lower, mask);
					  }
			);
		} else {
			ska_sort(__data.begin() + start_pos(tid),
					 __data.begin() + end_pos(tid),
					 [lower, mask](const Element &e) {
					   return e.get_label_container_ptr()[lower] & mask;
					 }
			);
		}

	}

	/// does what the name suggest.
	/// \param e element we want to search
	/// \param k_lower lower coordinate on which the element must be equal
	/// \param k_higher higher coordinate the elements must be equal
	/// \return the position of the first (lower) element which is equal to e. -1 if nothing found
	size_t search_level(const Element &e,
						const uint32_t k_lower,
						const uint32_t k_higher,
						bool sort=false) noexcept {
		auto r = std::find_if(__data.begin(),
							  __data.end(),
							  [&e, k_lower, k_higher](const Element &c) {
								return e.is_equal(c, k_lower, k_higher);
							  }
		);

		const auto dist = distance(__data.begin(), r);

		if (r == __data.end())
			return -1; // nothing found

		if (!__data[dist].is_equal(e, k_lower, k_higher))
			return -1;

		return dist;
	}

	/// \param e	element to search for
	/// \return	a tuple indicating the start and end indices within the list. start == end == load indicating nothing found,
	std::pair<uint64_t, uint64_t> search_boundaries(const Element &e,
													const uint32_t k_lower,
													const uint32_t k_higher,
													const uint32_t tid) noexcept {
		uint64_t end_index;
		uint64_t start_index = search_level(e, k_lower, k_higher, tid);
		if (start_index == uint64_t(-1))
			return std::pair<uint64_t, uint64_t>(-1, -1);

		// get the upper index
		end_index = start_index + 1;
		while (end_index < nr_elements && (__data[start_index].is_equal(__data[end_index], k_lower, k_higher)))
			end_index += 1;

		return std::pair<uint64_t, uint64_t>(start_index, end_index);
	}

	/// overwrites every element with the byte sym
	/// \param tid thread number
	/// \param sym byte to overwrite the memory with.
	void zero(const uint32_t tid, const uint8_t sym=0) noexcept {
		ASSERT(tid < threads);

		uint64_t s = start_pos(tid);
		uint64_t l = end_pos(tid) - s;
		memset((void *) (uint64_t(__data.data()) + s), sym, l);
	}

	/// zero out the i-th element.
	/// \param i
	void zero_element(size_t i) noexcept {
		ASSERT(i < nr_elements);
		__data[i].zero();
	}

	/// just drop it
	void resize(size_t i) noexcept {}

	/// remove the i-th element completely from the list
	/// \param i
	void erase(size_t i) noexcept {
		ASSERT(i < nr_elements);
		__data.erase(__data.begin()+i);
	}

	/// set L[load] = e1 + e2 and updated the load factor. Note this is usefull, because every thread can so maintain
	/// its own list size.
	/// \param e1	first element
	/// \param e2	second element to add
	/// \param load load factor = number of elements currently in the list.
	/// \param tid thread number
	void add_and_append(const Element &e1,
						const Element &e2,
						LoadType &load,
						const uint32_t tid) noexcept {
		ASSERT(tid < threads);

		if (load >= thread_block)
			return;

		Element::add(__data[start_pos(tid) + load], e1, e2);
		load += 1;
	}

	/// same function as above, but the label and value split into two separate function parameters.
	/// This is needed for lists which do not save an element in one variable.
	/// \param l1 label of the first element
	/// \param v1 value of the first element
	/// \param l2 label of the second element
	/// \param v2 value of the second element
	/// \param load current load of the list
	/// \param tid thread id inserting this element.
	void add_and_append(const LabelType &l1, const ValueType &v1,
						const LabelType &l2, const ValueType &v2,
						LoadType &load, const uint32_t tid) noexcept {
		ASSERT(tid < threads);

		if (load >= thread_block)
			return;

		ValueType::add(data_value(start_pos(tid) + load), v1, v2);
		LabelType::add(data_label(start_pos(tid) + load), l1, l2);
		load += 1;
	}


	// returning the start pointer for each thread
	inline LabelType* start_label_ptr(const uint32_t tid) noexcept { return (LabelType *)(((uint64_t)&__data[start_pos(tid)]) + value_size); };

	// Single threaded copy
	inline Parallel_List_FullElement_T<Element>& operator=(const Parallel_List_FullElement_T<Element>& other) noexcept {
		// Guard self assignment
		if (this == &other)
			return *this;

		nr_elements = other.nr_elements;
		thread_block = other.thread_block;
		threads = other.threads;

		__data = other.__data;
		return *this;
	}

	// parallel copy
	inline void static copy(Parallel_List_FullElement_T &out, const Parallel_List_FullElement_T &in , const uint32_t tid) noexcept {
		out.nr_elements = in.size();
		out.thread_block = in.thread_block;

		const std::size_t s = tid*in.threads;
		const std::size_t c = ((tid == in.threads - 1) ? in.thread_block : in.nr_elements- (in.threads -1)*in.thread_block);

		memcpy(out.__data_value+s, in.__data_value+s, c*sizeof(ValueType));
		memcpy(out.__data_label+s, in.__data_value+s, c*sizeof(LabelType));
	}

	/// some useful stuff
	auto begin() noexcept { return __data.begin(); }
	auto end() noexcept { return __data.end(); }

	/// returns the size in bytes
	[[nodiscard]] __FORCEINLINE__ constexpr uint64_t bytes() noexcept {
		return nr_elements*sizeof(Element);
	}

private:
	// I want the data ptr (the hot part of this class as aligned as possible.)
	alignas(PAGE_SIZE) std::vector<Element>  __data;

	/// total number of elements in the list
	size_t nr_elements;

	/// number of elements each threads needs to process
	size_t thread_block;

	/// number of threads which can access this list
	uint32_t threads;
};


///
/// \tparam Element
/// \param out
/// \param obj
/// \return
template<class Element>
std::ostream &operator<<(std::ostream &out, const Parallel_List_FullElement_T<Element> &obj) {
	for (uint64_t i = 0; i < obj.size(); ++i) {
		out << i << " " << obj.data(i) << std::flush;
	}

	return out;
}

#endif//DECODING_PARALLEL_FULL_H
