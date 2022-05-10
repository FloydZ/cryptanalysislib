#ifndef SMALLSECRETLWE_LIST_H
#define SMALLSECRETLWE_LIST_H

// stl include
#include <iterator>
#include <vector>           // main data container
#include <algorithm>        // search/find routines
#include <cassert>

#ifdef DUSE_FPLLL
// external includes
#include "fplll/nr/matrix.h"
#endif

// internal includes
#include "element.h"
#include "combinations.h"
#include "sort.h"

template<class Element>
concept ListAble = requires(Element a) {
	typename Element::ValueType;
	typename Element::LabelType;
	typename Element::MatrixType;

	typename Element::ValueContainerType;
	typename Element::LabelContainerType;

	typename Element::ValueDataType;
	typename Element::LabelDataType;

	requires ElementAble<typename Element::ValueType,
						 typename Element::LabelType,
						 typename Element::MatrixType>;

	// function requirements
	// TODO be more precise:
	requires requires(const uint32_t a1, const uint32_t a2) { a.print(a1, a2); };
	requires requires() { a.zero(); a.get_label(); a.get_value(); };
	requires requires(const uint32_t b1, const uint32_t b2) { Element::add(a, a, a, b1, b1, b2); };
};

/// This implements a data struct which can hold arbitrary amount of labels and values in two different lists.
///  	To separate the labels and values from each others, means that we are able to faster enumerate over only the labels
/// \tparam Element
template<class Element>
	requires ListAble<Element>
class Parallel_List_T {
private:
	// disable the empty constructor. So you have to specify a rough size of the list. This is for optimisations reasons.
	Parallel_List_T() : nr_elements(0) {};

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

	typedef uint64_t LoadType;

	~Parallel_List_T() noexcept {
		/*if (__data_value != nullptr) {
			//cryptanalysislib_aligned_free(__data_value);
		}
		if (__data_label != nullptr) {
			//cryptanalysislib_aligned_free(__data_label);
		}
		*/
	}

	/// TODO replace ptr with std::array
	/// \param size of the whole list
	/// \param threads number of threads access this list
	/// \param thread_block size of each block for each thread.
	/// \param no_value do not allocate the value array
	explicit Parallel_List_T(const uint64_t size,
	                         const uint32_t threads,
	                         const uint64_t thread_block,
	                         bool no_values=false) noexcept :
	    nr_elements(size),
		thread_block(thread_block),
		threads1(threads)
	{
		if (no_values == false) {
			__data_value = (ValueType *) cryptanalysislib_aligned_malloc(size * sizeof(ValueType), PAGE_SIZE);
			if (__data_value == NULL) {
				assert("could not alloc __data_value");
				exit(1);
			}

			memset(__data_value, 0, size * sizeof(ValueType));
		}

		__data_label = (LabelType *)cryptanalysislib_aligned_malloc(size*sizeof(LabelType), PAGE_SIZE);
		if (__data_label == NULL) {
			assert("could not alloc __data_label");
			exit(1);
		}

		memset(__data_label, 0, size*sizeof(LabelType));
	}

	constexpr size_t size() const noexcept { return nr_elements; }
	constexpr size_t size(const uint32_t tid) const noexcept { return thread_block; }

	inline ValueType* data_value() noexcept { return __data_value; }
	inline const ValueType* data_value() const noexcept { return __data_value; }
	inline LabelType* data_label() noexcept { return __data_label; }
	inline const LabelType* data_label() const noexcept { return __data_label; }

	inline ValueType& data_value(const size_t i) noexcept {  ASSERT(i < nr_elements); return __data_value[i]; }
	inline const ValueType& data_value(const size_t i) const noexcept { ASSERT(i < nr_elements); return __data_value[i]; }
	inline LabelType& data_label(const size_t i) noexcept { ASSERT(i < nr_elements); return __data_label[i]; }
	inline const LabelType& data_label(const size_t i) const noexcept { ASSERT(i < nr_elements); return __data_label[i]; }

	// print the `pos ` elements label between k_lower and k_higer in binary
	void print_binary(const uint64_t pos,
	                  const uint16_t k_lower,
	                  const uint16_t k_higher) const noexcept {
		data_label(pos).print_binary(k_lower, k_higher);
	}

	// single threaded memcopy
	inline Parallel_List_T<Element>& operator=(const Parallel_List_T<Element>& other) noexcept {
	    // Guard self assignment
	    if (this == &other)
	        return *this;

	    nr_elements = other.size();
		thread_block = other.thread_block;
		threads1 = other.threads1;

	    memcpy(__data_value, other.__data_value, nr_elements*sizeof(ValueType));
		memcpy(__data_label, other.__data_label, nr_elements*sizeof(LabelType));
	    return *this;
	}

	// parallel memcopy
	inline void static copy(Parallel_List_T &out, const Parallel_List_T &in , const uint32_t tid) noexcept {
		out.nr_elements = in.size();
		out.thread_block = in.thread_block;

		const std::size_t s = tid*in.threads1;
		const std::size_t c = ((tid == in.threads1 - 1) ? in.thread_block : in.nr_elements- (in.threads1-1)*in.thread_block);

		memcpy(out.__data_value+s, in.__data_value+s, c*sizeof(ValueType));
		memcpy(out.__data_label+s, in.__data_value+s, c*sizeof(LabelType));
	}

	template<typename Hash>
	void sort(uint32_t tid, Hash hash) noexcept {
		// TODO sort, good question how this should be implemented?
		ASSERT(0 && "not implemented");
	}

	/// print some information. Note: this will print the whole list. So be careful if its big.
	/// \param k_lower
	/// \param k_upper
	void print(const uint32_t k_lower=0,
	           const uint32_t k_upper=LabelType::LENGTH) const noexcept {
		for (size_t i = 0; i < nr_elements; ++i) {
			std::cout << __data_label[i] << " " << __data_value[i] << ", i: " << i << "\n" << std::flush;
		}
	}

	// returning the range in which one thread is allowed to operate
	inline uint64_t start_pos(const uint32_t tid) const noexcept { return tid*thread_block; };
	inline uint64_t end_pos(const uint32_t tid) const noexcept { return( tid+1)*thread_block; };

	// TODO implement ranges/iterators
	auto begin() noexcept {
		// return std::vector<LabelType>::iterator (__data_label);
	}

	uint64_t bytes() const noexcept {
		if (__data_value == nullptr)
			return nr_elements * sizeof(LabelType);

		return nr_elements * (sizeof(ValueType) + sizeof(LabelType));
	}

private:
	ValueType *__data_value = nullptr;
	LabelType *__data_label = nullptr;

	size_t nr_elements;
	uint64_t thread_block;
	uint64_t threads1;
};

/// same the class
///		`Parallel_List_T`
/// with the big difference that Elements = <Label, Value> are stored within the same array.
/// Additionally the thread parallel access is further implemented
/// In comparison to different approach this dataset does not track its load factors. It leaves this work completely to
///		the calling function.
template<class Element>
	requires ListAble<Element>
class Parallel_List_FullElement_T {
private:
	// disable the empty constructor. So you have to specify a rough size of the list. This is for optimisations reasons.
	Parallel_List_FullElement_T() : nr_elements(0), thread_block(0), threads1(1) {};

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

	// internal data types lengths
	constexpr static uint32_t ValueLENGTH = ValueType::LENGTH;
	constexpr static uint32_t LabelLENGTH = LabelType::LENGTH;

	constexpr static uint64_t element_size = Element::ssize();
	constexpr static uint64_t value_size = ValueType::ssize();
	constexpr static uint64_t label_size = LabelType::ssize();

	// somehow make such flags configurable
	constexpr static bool USE_STD_SORT = false;

	// default deconstructor
	~Parallel_List_FullElement_T() = default;

	// simple single threaded constructor
	explicit Parallel_List_FullElement_T(const size_t size) noexcept :
			nr_elements(size), thread_block(size), threads1(1)
	{
		__data.resize(size);
	}

	explicit Parallel_List_FullElement_T(const size_t size,
	                                     const uint32_t threads,
	                                     const size_t thread_block) noexcept :
		nr_elements(size),
	    thread_block(thread_block),
	    threads1(threads)
	{
		__data.resize(size);
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
		if (threads1 > 2) {
			bitonic_sort_par(start, split_length, seq, flag);
			bitonic_sort_par(start + split_length, split_length, seq, flag);
		}
	}
#endif

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
		ASSERT(tid < threads1);

		const auto e = size == size_t(-1) ? end_pos(tid) : size;
		sort(start_pos(tid), e, hash);
	}

	/// TODO generalize for kAryType
	/// \param i lower coordinate in the label used as the sorting index
	/// \param j upper   .....
	void sort_level(const uint32_t i, const uint32_t j) noexcept {
		ASSERT(i < j);
		// ASSERT(lower == upper); // TODO
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
	void reset(const uint32_t tid, const uint8_t sym=0) noexcept {
		ASSERT(tid < threads1);

		uint64_t s = start_pos(tid);
		uint64_t l = end_pos(tid) - s;
		memset((void *) (uint64_t(__data.data()) + s), sym, l);
	}

	/// zero out the i-th element.
	/// \param i
	void zero(size_t i) noexcept {
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
		ASSERT(tid < threads1);

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
		ASSERT(tid < threads1);

		if (load >= thread_block)
			return;

		ValueType::add(data_value(start_pos(tid) + load), v1, v2);
		LabelType::add(data_label(start_pos(tid) + load), l1, l2);
		load += 1;
	}

		// size information
	constexpr __FORCEINLINE__ size_t size() const noexcept { return nr_elements; }
	constexpr __FORCEINLINE__ size_t size(const uint32_t tid) const noexcept { return thread_block; }

	inline Element * data() noexcept{ return __data.data(); }
	inline const Element* data() const noexcept { return __data.data(); }

	inline Element &data(const size_t i) noexcept { ASSERT(i < size()); return this->__data[i]; }
	inline const Element& data(const size_t i) const noexcept { ASSERT(i < size()); return this->__data[i];  }

	inline Element &operator[](const size_t i) noexcept { ASSERT(i < size()); return this->__data[i]; }
	const inline Element &operator[](const size_t i) const noexcept {ASSERT(i < size()); return this->__data[i]; }

	inline ValueType* data_value() noexcept { return (ValueType *)__data.data(); }
	inline const ValueType* data_value() const noexcept { return (ValueType *)__data.data(); }
	inline LabelType* data_label() noexcept { return (LabelType *)__data.data(); }
	inline const LabelType* data_label() const noexcept { return (LabelType *)__data.data(); }

	inline ValueType& data_value(const size_t i) noexcept {  ASSERT(i < nr_elements); return __data[i].get_value(); }
	inline const ValueType& data_value(const size_t i) const noexcept { ASSERT(i < nr_elements); return __data[i].get_value(); }
	inline LabelType& data_label(const size_t i) noexcept { ASSERT(i < nr_elements); return __data[i].get_label(); }
	inline const LabelType& data_label(const size_t i) const noexcept { ASSERT(i < nr_elements); return __data[i].get_label(); }

	// returning the range in which one thread is allowed to operate
	inline size_t start_pos(const uint32_t tid) const noexcept { return tid*thread_block; };
	inline size_t end_pos(const uint32_t tid) const noexcept { return( tid+1)*thread_block; };

	// returning the start pointer for each thread
	inline LabelType* start_label_ptr(const uint32_t tid) noexcept { return (LabelType *)(((uint64_t)&__data[start_pos(tid)]) + value_size); };

	// Single threaded copy
	inline Parallel_List_FullElement_T<Element>& operator=(const Parallel_List_FullElement_T<Element>& other) noexcept {
		// Guard self assignment
		if (this == &other)
			return *this;

		nr_elements = other.nr_elements;
		thread_block = other.thread_block;
		threads1 = other.threads1;

		__data = other.__data;
		return *this;
	}

	// parallel copy
	inline void static copy(Parallel_List_FullElement_T &out, const Parallel_List_FullElement_T &in , const uint32_t tid) noexcept {
		out.nr_elements = in.size();
		out.thread_block = in.thread_block;

		const std::size_t s = tid*in.threads1;
		const std::size_t c = ((tid == in.threads1 - 1) ? in.thread_block : in.nr_elements- (in.threads1-1)*in.thread_block);

		memcpy(out.__data_value+s, in.__data_value+s, c*sizeof(ValueType));
		memcpy(out.__data_label+s, in.__data_value+s, c*sizeof(LabelType));
	}


	void print(uint64_t s=0, uint64_t e=30) const noexcept {
		for (uint64_t i = s; i < MIN(e,nr_elements); ++i) {
			std::cout << __data[i];
		}

		std::cout << "\n";
	}
private:
	// i want the data ptr (the hot part of this class as aligned as possible.)
	alignas(PAGE_SIZE) std::vector<Element>  __data;

	size_t nr_elements;
	size_t thread_block;
	uint32_t threads1;
};


/// nearly the same as
///		 `Parallel_List_T`
/// \tparam Element
template<class Element, uint32_t nri>
    requires ListAble<Element>
class Parallel_List_IndexElement_T {
private:
	// disable the empty constructor. So you have to specify a rough size of the list. This is for optimisations reasons.
	Parallel_List_IndexElement_T() : nr_elements(0) {};

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

	// internal data types lengths
	constexpr static uint32_t ValueLENGTH = ValueType::LENGTH;
	constexpr static uint32_t LabelLENGTH = LabelType::LENGTH;

	using IndexType = std::array<uint32_t, nri>;
	using InternalElementType = std::pair<LabelType, IndexType>;

	~Parallel_List_IndexElement_T() noexcept {}

	/// TODO explain
	/// \param size
	/// \param threads
	/// \param thread_block
	explicit Parallel_List_IndexElement_T(const size_t size,
	                                      const uint32_t threads,
	                                      const size_t thread_block) noexcept :
			nr_elements(size), thread_block(size), threads1(1)
	{
		__data.resize(size);
	}

	///
	constexpr size_t size() const noexcept { return nr_elements; }

	// Not implementable, because we do not save the value sin this list.
	// inline ValueType& data_value(const size_t i) noexcept { return __data[i]; }
	// inline const ValueType& data_value(const sizez_t i) const noexcept { return __data[]; }
	inline LabelType& data_label(const size_t i) noexcept { ASSERT(i < nr_elements); return __data[i].first; }
	inline const LabelType& data_label(const size_t i) const noexcept { ASSERT(i < nr_elements); return __data[i].first; }

	// returning the range in which one thread is allowed to operate
	inline size_t start_pos(const uint32_t tid) const noexcept { ASSERT(tid < threads1); return tid * thread_block; };
	inline size_t end_pos(const uint32_t tid) const noexcept  { ASSERT(tid < threads1); return (tid+1) * thread_block; };

	inline Parallel_List_IndexElement_T& operator=(const Parallel_List_IndexElement_T& other) noexcept {
		// Guard self assignment
		if (this == &other)
			return *this;

		nr_elements = other.size();
		thread_block = other.thread_block;
		threads1 = other.threads;

		memcpy(__data, other.data(), nr_elements*sizeof(InternalElementType));
		return *this;
	}

	inline void static copy(Parallel_List_IndexElement_T &out, const Parallel_List_IndexElement_T &in , const uint32_t tid) noexcept {
		out.nr_elements = in.size();
		out.thread_block = in.thread_block;

		const std::size_t s = tid*in.threads1;
		const std::size_t c = ((tid == in.threads1 - 1) ? in.thread_block : in.nr_elements- (in.threads1-1)*in.thread_block);

		memcpy(out.__data+s, in.__data+s, c*sizeof(InternalElementType));
	}

	/// \param l
	/// \param j
	/// \param tid
	void sort_level(const uint64_t l, const uint64_t j, const uint32_t tid) noexcept {
		ASSERT(0 && "not implemented\n");
	}

	/// TODO describe
	/// \param l1
	/// \param l2
	/// \param i2
	/// \param i2
	/// \param load
	/// \param tid
	inline void add_and_append(const LabelType &l1, const LabelType &l2,
	                    const uint32_t i1, const uint32_t i2,
	                    uint64_t &load, const uint32_t tid) noexcept {
		ASSERT(tid < threads1);

		if (load >= thread_block)
			return;

		LabelType::add(__data[start_pos(tid) + load].first, l1, l2);
		__data[start_pos(tid) + load].second[0] = i1;
        __data[start_pos(tid) + load].second[1] = i2;
        load += 1;
	}

	/// return the number of butes needed for this list
	uint64_t bytes() const noexcept {
		return __data.size() * sizeof(InternalElementType);
	}

public:
	size_t nr_elements;
	size_t thread_block;
	uint32_t threads1;

	alignas(PAGE_SIZE) std::vector<InternalElementType> __data;
};

template<class Element>
	requires ListAble<Element>
class List_T {
private:
	// disable the empty constructor. So you have to specify a rough size of the list.
	// This is for optimisations reasons.
	List_T() : load(0) {};

public:
	typedef Element ElementType;
	typedef typename Element::ValueType ValueType;
	typedef typename Element::LabelType LabelType;

	typedef typename Element::ValueType::ContainerLimbType ValueContainerLimbType;
	typedef typename Element::LabelType::ContainerLimbType LabelContainerLimbType;

	typedef typename Element::ValueContainerType ValueContainerType;
	typedef typename Element::LabelContainerType LabelContainerType;

	typedef typename Element::ValueDataType ValueDataType;
	typedef typename Element::LabelDataType LabelDataType;

	typedef typename Element::MatrixType MatrixType;

	// internal data types lengths
	constexpr static uint32_t ValueLENGTH = ValueType::LENGTH;
	constexpr static uint32_t LabelLENGTH = LabelType::LENGTH;

	/// Base constructor. The empty one is disabled.
	/// \param number
	List_T(uint64_t number) {
        __data.resize(number);
        load = 0;
    }

    /// checks if all elements in the list fullfil the equation: 	label == value*m
    /// \param m 		the matrix.
    /// \param rewrite 	if set to true, all labels within each element will we overwritten by the recalculated.
    /// \return 		true if ech element is correct.
	bool is_correct(const Matrix_T<MatrixType> &m, const bool rewrite=false) {
		bool ret = false;
		for (int i = 0; i < get_load(); ++i) {
			ret |= __data[i].is_correct(m, rewrite);
			if ((ret) && (!rewrite))
				return ret;
		}

		return ret;
	}

	/// Andres Code
    void static odl_merge(std::vector<std::pair<uint64_t,uint64_t>>&target, const List_T &L1, const List_T &L2, int klower = 0,
                int kupper = -1) {
        if (kupper == -1 && L1.get_load() > 0)
            kupper = L1[0].label_size();
        uint64_t i = 0, j = 0;
        target.resize(0);
        while (i < L1.get_load() && j < L2.get_load()) {
            if (L2[j].is_greater(L1[i], klower,kupper))
                i++;

            else if (L1[i].is_greater(L2[j], klower,kupper))
                j++;

            else {
                uint64_t i_max, j_max;
                // if elements are equal find max index in each list, such that they remain equal
                for (i_max = i + 1; i_max < L1.get_load() && L1[i].is_equal(L1[i_max], klower,kupper); i_max++) {}
                for (j_max = j + 1; j_max < L2.get_load() && L2[j].is_equal(L2[j_max], klower,kupper); j_max++) {}

                // store each matching tuple
                int jprev = j;
                for (; i < i_max; ++i) {
                    for (j = jprev; j < j_max; ++j) {
                        std::pair<uint64_t,uint64_t > a;
                        a.first=L1[i].get_value();
                        a.second=L2[j].get_value();
                        target.push_back(a);
                    }

                }
            }
        }
	}

	/// generate/initialize this list as a random base list.
	/// for each element a new value is randomly choosen and the label is calculated by a matrix-vector-multiplication.
	/// \param k amount of 'Elements' to add to the list.
	/// \param m base matrix to calculate the 'Labels' corresponding to a 'Value'
    void generate_base_random(const uint64_t k, const Matrix_T<MatrixType> &m) {
	    for (int i = 0; i < k; ++i) {
		    // by default this creates a complete random 'Element' with 'Value' coordinates \in \[0,1\]
		    Element e{};
		    e.random(m);    // this 'random' function takes care of the vector-matrix multiplication.
		    append(e);
	    }
    }

    /// create a list of lexicographical ordered elements (where the values are lexicographical ordered).
    /// IMPORTANT; No special trick is applied, so every Element needs a fill Matrix-vector multiplication.
    /// \param number_of_elements
    /// \param ones 				hamming weight of the 'Value' of all elements
    /// \param m
	void generate_base_lex(const uint64_t number_of_elements, const uint64_t ones, const Matrix_T<MatrixType> &m) {
		ASSERT(internal_counter == 0 && "already initialised");
		Element e{}; e.zero();

	    __data.resize(number_of_elements);
	    load = number_of_elements;

	    const uint64_t n = e.value_size();
	    Combinations_Lexicographic<decltype(e.get_value().data().get_type())> c{n, ones};
	    c.left_init(e.get_value().data().data().data());
	    while(c.left_step(e.get_value().data().data().data()) != 0) {
		    if (internal_counter >= size())
			    return;

		    new_vector_matrix_product<LabelType, ValueType, MatrixType>(e.get_label(), e.get_value(), m);

		    __data[internal_counter] = e;
		    internal_counter += 1;
	    }
	}
	
	/// \param level				current lvl within the tree.
	/// \param level_translation
	void sort_level(const uint32_t level, const std::vector<uint64_t> &level_translation) {
		uint64_t k_lower, k_higher;
		translate_level(&k_lower, &k_higher, level, level_translation);
		return sort_level(k_lower, k_higher);
	}

	/// sort the list
	void sort_level(const uint32_t k_lower, const uint32_t k_higher) {
		std::sort(__data.begin(), __data.begin()+load,
		          [k_lower, k_higher](const auto &e1, const auto &e2) {

#if !defined(SORT_INCREASING_ORDER)
			          return e1.is_lower(e2, k_lower, k_higher);
#else
			          return e1.is_greater(e2, k_lower, k_higher);
#endif
		          }
		);

		ASSERT(is_sorted(k_lower, k_higher));
	}


	void sort_level(const uint32_t k_lower, const uint32_t k_higher, const uint32_t tid) {
		std::sort(__data.begin()+start_pos(tid), __data.begin()+end_pos(tid),
		          [k_lower, k_higher](const auto &e1, const auto &e2) {

#if !defined(SORT_INCREASING_ORDER)
			          return e1.is_lower(e2, k_lower, k_higher);
#else
			          return e1.is_greater(e2, k_lower, k_higher);
#endif
		          }
		);

		ASSERT(is_sorted(k_lower, k_higher));
	}

	/// sort the list. only valid in the binary case
	inline void sort_level_binary(const uint32_t k_lower, const uint32_t k_higher) {
		ASSERT(get_load() > 0);

		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		const uint64_t upper = T::round_down_to_limb(k_higher);

		// choose the optimal implementation.
		if (lower == upper)
			sort_level_sim_binary(k_lower, k_higher);
		else
			sort_level_ext_binary(k_lower, k_higher);

		ASSERT(is_sorted(k_lower, k_higher));
	}

private:
	/// IMPORTANT: DO NOT CALL THIS FUNCTION directly. Use `sort_level` instead.
	/// special implementation of the sorting function using specialised compare functions of the `BinaryContainer` class
	/// which uses precomputed masks.
	inline void sort_level_ext_binary(const uint64_t k_lower, const uint64_t k_higher) {
		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		const uint64_t upper = T::round_down_to_limb(k_higher);
		const uint64_t lmask = T::higher_mask(k_lower);
		const uint64_t umask = T::lower_mask(k_higher);

		std::sort(__data.begin(), __data.begin() + load,
		          [lower, upper, lmask, umask](const auto &e1, const auto &e2) {
#if !defined(SORT_INCREASING_ORDER)
			          return e1.get_label().data().is_lower_ext2(e2.get_label().data(), lower, upper, lmask, umask);
#else
			          return e1.get_label().data().is_greater_ext2(e2.get_label().data(), lower, upper, lmask, umask);
#endif
		          }
		);

	}

	/// IMPORTANT: DO NOT CALL THIS FUNCTION directly. Use `sort_level` instead.
	/// special implementation of the sort function using highly optimized special compare routines from the data class
	/// `BinaryContainer` if one knows that k_lower, k_higher are in the same limb.
	inline void sort_level_sim_binary(const uint32_t k_lower, const uint32_t k_higher) {
		using T = LabelContainerType;

		const uint64_t lower = T::round_down_to_limb(k_lower);
		//const uint64_t upper = T::round_down_to_limb(k_higher);
		// TODO ASSERT(lower == upper);

		const uint64_t mask = T::higher_mask(k_lower) & T::lower_mask(k_higher);
		std::sort(__data.begin(), __data.begin() + load,
		          [lower, mask](const auto &e1, const auto &e2) {
#if !defined(SORT_INCREASING_ORDER)
			          return e1.get_label().data().is_lower_simple2(e2.get_label().data(), lower, mask);
#else
			          return e1.get_label().data().is_greater_simple2(e2.get_label().data(), lower, mask);
#endif
		          }
		);
	}

public:
	// implements a binary search on the given data.
	// if the boolean flag `sort` is set to true, the underlying list is sorted.
	// USAGE:
	// can be found: "tests/binary/list.h TEST(SerchBinary, Simple) "
	inline size_t search_level_binary(const Element &e, const uint32_t k_lower, const uint32_t k_higher,
	                                    const bool sort = false) {
		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		const uint64_t upper = T::round_down_to_limb(k_higher);

		if (sort)
			sort_level(k_lower, k_higher);

		if (lower == upper)
			return search_level_binary_simple(e, k_lower, k_higher);
		else
			return search_level_binary_extended(e, k_lower, k_higher);
	}

private:
	///
	/// \param e element to search for
	/// \param k_lower lower limit
	/// \param k_higher higher limit
	/// \return the position within the
	inline uint64_t search_level_binary_simple(const Element &e, const uint64_t k_lower, const uint64_t k_higher) {
		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		//const uint64_t upper = T::round_down_to_limb(k_higher);
		const uint64_t mask = T::higher_mask(k_lower) & T::lower_mask(k_higher);
		// TODO ASSERT(lower == upper);

		auto r = std::lower_bound(__data.begin(), __data.begin() + load, e,
		                          [lower, mask](const Element &c1, const Element &c2) {
			                          return c1.get_label().data().is_lower_simple2(c2.get_label().data(), lower, mask);
		                          }
		);

		const auto dist = distance(__data.begin(), r);
		if (r == __data.begin() + load) { return -1; }
		if (!__data[dist].is_equal(e, k_lower, k_higher)) return -1;
		return dist;
	}

	///
	/// \param e
	/// \param k_lower
	/// \param k_higher
	/// \return
	inline uint64_t search_level_binary_extended(const Element &e, const uint64_t k_lower, const uint64_t k_higher) {
		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		const uint64_t upper = T::round_down_to_limb(k_higher);
		const uint64_t lmask = T::higher_mask(k_lower);
		const uint64_t umask = T::lower_mask(k_higher);
		ASSERT(lower != upper);

		auto r = std::lower_bound(__data.begin(), __data.begin() + load, e,
		                          [lower, upper, lmask, umask](const Element &c1, const Element &c2) {
			                          return c1.get_label().data().is_lower_ext2(c2.get_label().data(), lower, upper, lmask, umask);
		                          }
		);

		const auto dist = distance(__data.begin(), r);
		if (r == __data.begin() + load) { return -1; }
		if (!__data[dist].is_equal(e, k_lower, k_higher)) return -1;
		return dist;
	}

public:
	///
	/// \param e
	/// \param k_lower
	/// \param k_higher
	/// \param sort
	/// \return
	inline uint64_t search_level_binary_custom(const Element &e, const uint64_t k_lower, const uint64_t k_higher,
	                                           const bool sort = false) {
		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		const uint64_t upper = T::round_down_to_limb(k_higher);

		if (sort)
			sort_level(k_lower, k_higher);

		if (lower == upper)
			return search_level_binary_custom_simple(e, k_lower, k_higher);
		else
			return search_level_binary_custom_extended(e, k_lower, k_higher);
	}

private:
	/// custom written binary search. Idea Taken from `https://academy.realm.io/posts/how-we-beat-cpp-stl-binary-search/`
	inline uint64_t search_level_binary_custom_simple(const Element &e, const uint64_t k_lower, const uint64_t k_higher) {
		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		//const uint64_t upper = T::round_down_to_limb(k_higher);
		const uint64_t mask = T::higher_mask(k_lower) & T::lower_mask(k_higher);
		// TODO  ASSERT(lower == upper);

		size_t size = load;
		size_t low = 0;

		LabelContainerType v;
		const LabelContainerType s = e.get_label().data();

		while (size > 0) {
			size_t half = size / 2;
			size_t other_half = size - half;
			size_t probe = low + half;
			size_t other_low = low + other_half;
			v = __data[probe].get_label().data();
			size = half;
			low = v.is_lower_simple2(s, lower, mask) ? other_low : low;
		}
		return low!=load ? low : -1;
	}

	///
	/// \param e
	/// \param k_lower
	/// \param k_higher
	/// \return
	inline uint64_t search_level_binary_custom_extended(const Element &e, const uint64_t k_lower, const uint64_t k_higher) {
		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		const uint64_t upper = T::round_down_to_limb(k_higher);
		const uint64_t lmask = T::higher_mask(k_lower);
		const uint64_t umask = T::lower_mask(k_higher);
		ASSERT(lower == upper);

		size_t size = load;
		size_t low = 0;

		LabelContainerType v;
		const LabelContainerType s = e.get_label().data();

		while (size > 0) {
			size_t half = size / 2;
			size_t other_half = size - half;
			size_t probe = low + half;
			size_t other_low = low + other_half;
			v = __data[probe].get_label().data();
			size = half;
			low = v.is_lower_ext2(s, lower, upper, lmask, umask) ? other_low : low;
		}
		return low!=load ? low : -1;
	}

public:
	/// does what the name suggest.
	/// \param e element we want to search
	/// \param k_lower lower coordinate on which the element must be equal
	/// \param k_higher higher coordinate the elements must be equal
	/// \return the position of the first (lower) element which is equal to e. -1 if nothing found
	size_t search_level(const Element &e, const uint64_t k_lower, const uint64_t k_higher, bool sort=false) {
		if constexpr (Element::binary()) {
			return search_level_binary(e, k_lower, k_higher, sort);
		} else {
			auto r = std::find_if(__data.begin(), __data.begin() + load, [&e, k_lower, k_higher](const Element &c) {
				                      return e.is_equal(c, k_lower, k_higher);
			                      }
			);

			const auto dist = distance(__data.begin(), r);

			if (r == __data.begin() + load)
				return -1; // nothing found

			if (!__data[dist].is_equal(e, k_lower, k_higher))
				return -1;

			return dist;
		}
	}

//	size_t search_parallel(const Element &e) {
//        auto r = std::find_if(std::execution::par, __data.begin(), __data.end(),
//                              [&e](const Element *c) { return c->get_value() == e.get_value(); });
//        if (r == std::end(__data)) { // nothing found
//            return -1;
//        }
//
//        return distance(__data.begin(), r);
//    }


//    uint64_t search_parallel_level(const Element &e, const uint64_t level, const std::vector<uint64_t> &vec) {
//        uint64_t k_lower, k_higher;
//        translate_level(&k_lower, &k_higher, level, vec);
//
//        auto r = std::find_if(std::execution::par, __data.begin(), __data.end(),
//                              [e, k_lower, k_higher](const Element *c) {
//                                    return e.is_equal(*c, k_lower, k_higher);
//                                });
//        if (r == std::end(__data)) { // nothing found
//            return -1;
//        }
//
//        return distance(__data.begin(), r);
//    }

    /// \param e
    /// \return	a tuple indicating the start and end indices within the list. start == end == load indicating nothing found,
    std::pair<uint64_t, uint64_t> search_boundaries(const Element &e, const uint64_t k_lower, const uint64_t k_higher) {
        uint64_t end_index;
		uint64_t start_index;
        if constexpr (!Element::binary()){
			start_index = search_level(e, k_lower, k_higher);
		} else {
			start_index = search_level_binary(e, k_lower, k_higher);
		}


		if (start_index == uint64_t(-1))
            return std::pair<uint64_t, uint64_t>(load, load);

        // get the upper index
        end_index = start_index + 1;
        while (end_index < load && (__data[start_index].is_equal(__data[end_index], k_lower, k_higher)))
            end_index += 1;

        return std::pair<uint64_t, uint64_t>(start_index, end_index);
    }

    /// A little helper function to check if a list is sorted. This is very useful to assert specific states within
    /// complex cryptanalytic algorithms.
    bool is_sorted(const uint64_t k_lower, const uint64_t k_higher) const {
	    for (uint64_t i = 1; i < get_load(); ++i) {
		    if (__data[i-1].is_equal(__data[i], k_lower, k_higher)) {
			    continue;
		    }

#if !defined(SORT_INCREASING_ORDER)
		    if (!__data[i-1].is_lower(__data[i], k_lower, k_higher)){
			    return false;
		    }
#else
		    if (!__data[i-1].is_greater(__data[i], k_lower, k_higher)){
			    return false;
		    }
#endif
	    }

	    return true;
	}

    /// appends the element e to the list. Note that the list keeps track of its load. So you dont have to do anyting.
    /// Note: if the list is full, every new element is silently discarded.
    /// \param e	Element to add
    void append(Element &e) {
        if (load < size()) {
            // wrong, increases effective size of container, use custom element copy function instead
            __data[load] = e;
        } else {
            __data.push_back(e);
        }

        load++;
    }

    /// append e1+e2|full_length to list
    /// \param e1 first element.
    /// \param e2 second element
    void add_and_append(const Element &e1, const Element &e2, const uint32_t norm=-1) {
       add_and_append(e1, e2, 0, LabelLENGTH, norm);
    }

	/// Same as the function above, but with a `constexpr` size factor. This may allow further optimisations to the
	/// compiler. Additionaly this functions adds e1 and e2 together and daves the result.
	/// \tparam approx_size Size of the list
	/// \param e1 first element
	/// \param e2 second element
	/// \param norm norm of the element e1+e2. If the calculated norm is bigger than `norm` the element is discarded.
	/// \return
	template<const uint64_t approx_size>
	int add_and_append(const Element &e1, const Element &e2, const uint32_t norm=-1) {
		if (load < approx_size) {
			Element::add(__data[load], e1, e2, 0, LabelLENGTH, norm);
			load += 1;
		}
		return approx_size-load;
		// ignore every element which could not be added to the list.
	}

	/// same as above, but the label is only added between k_lower and k_higher
	/// \param e1 first element to add
	/// \param e2 second element
	/// \param k_lower lower dimension to add the label on
	/// \param k_higher higher dimension to add the label on
	/// \param norm filter norm. If the norm of the resulting element is higher than `norm` it will be discarded.
	void add_and_append(const Element &e1, const Element &e2,
	                    const uint32_t k_lower, const uint32_t k_higher, const uint32_t norm=-1) {
		if (load < this->size()) {
			auto b = Element::add(__data[load], e1, e2, k_lower, k_higher, norm);
			// 'add' returns true if a overflow, over the given norm occurred. This means that at least coordinate 'r'
			// exists for which it holds: |data[load].value[r]| >= norm
			if (b == true)
				return;
		} else {
			Element t{};
			auto b = Element::add(t, e1, e2, k_lower, k_higher, norm);
			if (b == true)
				return;

			// this __MUST__ be a copy.
			__data.push_back(t);
		}

		// we do not increase the 'load' of our internal data structure if one of the add functions above returns true.
		load++;
	}

	/// print some information. Note: this will print the whole list. So be careful if its big.
	/// \param k_lower
	/// \param k_upper
	void print(const uint32_t k_lower, const uint32_t k_upper) const {
		for (const auto &e : __data) {
			e.print(k_lower, k_upper);
		}
	}

	/// set the element at position i to zero.
	/// \param i
	void zero(const size_t i) {
		ASSERT(i < get_load()); __data[i].zero();
	}

	/// remove the element at pos i.
	/// \param i
	void erase(const size_t i) {
		ASSERT(i < get_load()); __data.erase(__data.begin()+i); load -= 1;
	}

	/// resize the list. Note only the sanity check, if the list is already big enough, is made. If not enough mem
	/// is allocatable it will through an exception.
	/// \param size		new size
	void resize(const size_t size) {
        if (size > __data.size() && size > load) {
            __data.resize(size);
        }
    }

	constexpr size_t size() const { return __data.size(); }
	constexpr size_t size(const uint32_t tid) const { return __data.size(); } // TODO not correct
    void set_data(Element &e, const uint64_t i) { ASSERT(i <size()); __data[i] = e; }

	size_t get_load() const { return load; }
    void set_load(size_t l) { load = l; }

    Element &operator[](size_t i) {
        ASSERT(i < load && "wrong index");
        return this->__data[i];
    }
    const Element &operator[](const size_t i) const {
        ASSERT(i < load && "wrong index");
        return this->__data[i];
    }

    // Get a const pointer. Sometimes useful if one ones to tell the kernel how to access memory.
    const Element* data(){ return __data.data(); }
	auto* data2(){ return __data.data(); }
	const auto* data2() const { return __data.data(); }

	inline ValueType& data_value(const size_t i){  ASSERT(i < __data.size()); return __data[i].get_value(); }
	inline const ValueType& data_value(const size_t i) const { ASSERT(i < __data.size()); return __data[i].get_value(); }
	inline LabelType& data_label(const size_t i){ ASSERT(i < __data.size()); return __data[i].get_label(); }
	inline const LabelType& data_label(const size_t i) const { ASSERT(i < __data.size()); return __data[i].get_label(); }

	// TODO
	// returning the range in which one thread is allowed to operate
	inline uint64_t start_pos(const uint32_t tid) const { return 0; };
	inline uint64_t end_pos(const uint32_t tid) const { return( __data.size()); };

private:
    // just a small counter for the '_generate' function. __MUST__ be ignored.
    size_t internal_counter = 0;

	/// internal data representation of the list.
	alignas(PAGE_SIZE) std::vector<Element> __data;
    size_t load;
};


template<class Element>
std::ostream &operator<<(std::ostream &out, const List_T<Element> &obj) {
    for (uint64_t i = 0; i < obj.get_load(); ++i) {
        out << obj[i];
    }

    return out;
}

template<class Element>
std::ostream &operator<<(std::ostream &out, const Parallel_List_T<Element> &obj) {
	for (uint64_t i = 0; i < obj.size(); ++i) {
		out << obj.data_value(i) << " " << obj.data_label(i) << " " << i << "\n" << std::flush;
	}

	return out;
}

template<class Element>
std::ostream &operator<<(std::ostream &out, const Parallel_List_FullElement_T<Element> &obj) {
	for (uint64_t i = 0; i < obj.size(); ++i) {
		out << i << " " << obj.data(i) << std::flush;
	}

	return out;
}

#endif //SMALLSECRETLWE_LIST_H
