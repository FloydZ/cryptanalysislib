#ifndef CRYPTANALYISLIB_LIST_PARALLEL_INDEX_H
#define CRYPTANALYISLIB_LIST_PARALLEL_INDEX_H

#include "list/common.h"

template<class InnerElement, const uint32_t nri>
class Parallel_List_IndexElement_Wrapper_T {
public:
	using LabelType = typename InnerElement::LabelType;

	/// additional Typedefs
	using IndexType = std::array<uint32_t, nri>;
	using InternalElementType = std::pair<LabelType, IndexType>;
};


/// nearly the same as
///		 `Parallel_List_T`
/// with the main difference that this list only saves the label,
/// and not the Value. But instead a counter/index is saved which
/// NOTE: does not track the load
/// \tparam Element
template<class Element, uint32_t nri>
class Parallel_List_IndexElement_T :
	public MetaListT<Element>
{
public:
	/// needed typedefs
	using typename MetaListT<Element>::ElementType;
	using typename MetaListT<Element>::ValueType;
	using typename MetaListT<Element>::LabelType;
	using typename MetaListT<Element>::ValueLimbType;
	using typename MetaListT<Element>::LabelLimbType;
	using typename MetaListT<Element>::ValueContainerType;
	using typename MetaListT<Element>::LabelContainerType;
	using typename MetaListT<Element>::ValueDataType;
	using typename MetaListT<Element>::LabelDataType;
	using typename MetaListT<Element>::MatrixType;

	/// additional Typedefs
	using IndexType = std::array<uint32_t, nri>;
	using InternalElementType = std::pair<LabelType, IndexType>;

	/// needed values
	using MetaListT<Element>::__load;
	using MetaListT<Element>::__size;
	using MetaListT<Element>::__threads;
	using MetaListT<Element>::__thread_block_size;

	using MetaListT<Element>::ValueLENGTH;
	using MetaListT<Element>::LabelLENGTH;

	using MetaListT<Element>::ElementBytes;
	using MetaListT<Element>::ValueBytes;
	using MetaListT<Element>::LabelBytes;

	/// needed functions
	using MetaListT<Element>::size;
	using MetaListT<Element>::set_size;
	using MetaListT<Element>::threads;
	using MetaListT<Element>::set_threads;
	using MetaListT<Element>::thread_block_size;
	using MetaListT<Element>::set_thread_block_size;
	using MetaListT<Element>::resize;
	using MetaListT<Element>::load;
	using MetaListT<Element>::set_load;
	using MetaListT<Element>::start_pos;
	using MetaListT<Element>::end_pos;
	using MetaListT<Element>::data;
	using MetaListT<Element>::data_value;
	using MetaListT<Element>::data_label;
	using MetaListT<Element>::at;
	using MetaListT<Element>::set;
	using MetaListT<Element>::print;
	using MetaListT<Element>::print_binary;
	using MetaListT<Element>::begin;
	using MetaListT<Element>::end;
	using MetaListT<Element>::zero;
	using MetaListT<Element>::reset;
	using MetaListT<Element>::erase;

private:
	// disable the empty constructor. So you have to specify a rough size of the list. This is for optimisations reasons.
	Parallel_List_IndexElement_T() : MetaListT<Element>() {};

public:

	/// \param size total number of elements in the list
	/// \param threads
	/// \param thread_block
	constexpr explicit Parallel_List_IndexElement_T(const size_t size,
										  			const uint32_t threads) noexcept :
	       MetaListT<Element>(size, threads, false)
	{
		__data.resize(size);
	}

	/// not implemented
	void sort() noexcept {
		ASSERT(0);
	}

	/// overwrites every element with the byte sym
	/// \param tid thread number
	/// \param sym byte to overwrite the memory with.
	void zero(const uint32_t tid, const uint8_t sym=0) noexcept {
		ASSERT(tid < threads());

		uint64_t s = start_pos(tid);
		uint64_t l = end_pos(tid) - s;
		memset((void *) (uint64_t(__data.data()) + s), sym, l);
	}

	/// zero out the i-th element.
	/// \param i
	void zero_element(size_t i) noexcept {
		ASSERT(i < size());
		__data[i].first.zero();
		__data[i].second = 0;
	}

	/// add l1 and l2, and stores the result ad load positon with in the
	/// frame of a thread
	/// \param l1 first label
	/// \param l2 second label
	/// \param i2 first index
	/// \param i2 second indes
	/// \param load current load factor. Must be for each thread seperate
	/// \param tid threadid
	inline void add_and_append(const LabelType &l1,
	                           const LabelType &l2,
							   const uint32_t i1,
	                           const uint32_t i2,
							   uint64_t &load,
	                           const uint32_t tid) noexcept {
		ASSERT(tid < __threads);

		if (load >= thread_block_size())
			return;

		LabelType::add(__data[start_pos(tid) + load].first, l1, l2);
		__data[start_pos(tid) + load].second[0] = i1;
		__data[start_pos(tid) + load].second[1] = i2;
		load += 1;
	}

	/// data container
	alignas(CUSTOM_PAGE_SIZE) std::vector<InternalElementType> __data;
};


/// prints the list
template<class Element, const uint32_t nri>
std::ostream &operator<<(std::ostream &out,
                         const Parallel_List_IndexElement_T<Element, nri> &obj) {
	for (size_t i = 0; i < obj.get_load(); ++i) {
		out << obj[i];
	}

	return out;
}
#endif//DECODING_PARALLEL_INDEX_H
