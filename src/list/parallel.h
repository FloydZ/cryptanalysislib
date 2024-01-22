#ifndef DECODING_LIST_PARALLEL_H
#define DECODING_LIST_PARALLEL_H

#include <cstddef>
#include <cstdint>

#include "list/common.h"
#include "sort/sort.h"

/// This implements a data struct which can hold arbitrary amount of labels and values in two different lists.
///  	To separate the labels and values from each others, means that we are able to faster enumerate over only the labels
/// \tparam Element
template<class Element>
class Parallel_List_T : public MetaListT<Element> {
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

	using typename MetaListT<Element>::LoadType;

	/// needed values
	using MetaListT<Element>::__load;
	using MetaListT<Element>::__size;
	using MetaListT<Element>::__data;
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
	using MetaListT<Element>::zero;
	using MetaListT<Element>::reset;

	using MetaListT<Element>::print;
	using MetaListT<Element>::print_binary;
private:
	// disable the empty constructor. So you have to specify a rough size of the list. This is for optimisations reasons.
	Parallel_List_T() : MetaListT<Element>() {};

public:

	/// \param size of the whole list
	/// \param threads number of threads access this list
	/// \param thread_block size of each block for each thread.
	/// \param no_value do not allocate the value const_array
	constexpr explicit Parallel_List_T(const size_t size,
									   const uint32_t threads,
									   bool no_values=false) noexcept :
	   MetaListT<Element>(size, threads, false),
	   no_values(no_values)
	{
		if (no_values == false) {
			__data_value.resize(size);
			memset(__data_value.data(), 0, size * sizeof(ValueType));
		}

		__data_label.resize(size);
		memset(__data_label.data(), 0, size*sizeof(LabelType));
	}

	/// single threaded memcpy
	/// \param other
	/// \return
	inline Parallel_List_T<Element>& operator=(const Parallel_List_T<Element>& other) noexcept {
		// Guard self assignment
		if (this == &other) {
			return *this;
		}

		__size = other.size();
		__thread_block_size = other.__thread_block_size;
		__threads = other.__threads;

		memcpy(__data_value, other.__data_value, size()*sizeof(ValueType));
		memcpy(__data_label, other.__data_label, size()*sizeof(LabelType));
		return *this;
	}

	/// parallel memcopy
	/// \param out
	/// \param in
	/// \param tid
	inline void static copy(Parallel_List_T &out,
	                        const Parallel_List_T &in,
	                        const uint32_t tid) noexcept {
		out.nr_elements = in.size();
		out.thread_block = in.thread_block;

		const std::size_t s = tid*in.threads1;
		const std::size_t c = ((tid == in.threads1 - 1) ? in.thread_block : in.nr_elements- (in.threads1-1)*in.thread_block);

		memcpy(out.__data_value+s, in.__data_value+s, c*sizeof(ValueType));
		memcpy(out.__data_label+s, in.__data_value+s, c*sizeof(LabelType));
	}

	/// not implemented
	constexpr void sort() {
		ASSERT(0);
	}

	///
	/// \tparam Hash hash function type, needed for the bucket sort
	/// \param start start point= first element to sort
	/// \param end  end point = last element to sort
	/// \param hash hash function
	template<typename Hash>
	void sort(const size_t start, const size_t end, Hash hash) noexcept {
		ska_sort(__data_value + start, __data_value + end, hash);
	}

	///
	/// \tparam Hash hash function type, needed for the bucket sort
	/// \param tid thread id 
	/// \param hash hash function
	template<typename Hash>
	void sort(uint32_t tid, Hash hash) noexcept {
		const size_t start = start_pos(tid);
		const size_t end = start_pos(tid);
		ska_sort(__data_value + start, __data_value + end, hash);
	}

	/// zero a list
	/// \param tid
	constexpr void zero(const uint32_t tid=0) noexcept {
		ASSERT(tid < __threads);
		for (size_t i = start_pos(tid); i < end_pos(tid); ++i) {
			__data_value[i].zero();
			__data_label[i].zero();
		}
	}

	/// zeros a single element
	/// \param i
	constexpr void zero_element(const size_t i) noexcept {
		ASSERT(i < size());
		__data_value[i].zero();
		__data_label[i].zero();
	}

	/// iterator are useless in this class
	auto begin() noexcept { return nullptr; }
	auto end() noexcept { return nullptr; }

	constexpr inline ValueType* data_value() noexcept { return (ValueType *)__data_value.data() ; }
	constexpr inline const ValueType* data_value() const noexcept { return (ValueType *)__data_value.data(); }
	constexpr inline LabelType* data_label() noexcept { return (LabelType *)__data_label.data(); }
	constexpr inline const LabelType* data_label() const noexcept { return (const LabelType *)__data_label.data(); }

	constexpr inline ValueType& data_value(const size_t i) noexcept {  ASSERT(i < __size); return __data_value[i]; }
	constexpr inline const ValueType& data_value(const size_t i) const noexcept { ASSERT(i < __size); return __data_value[i]; }
	constexpr inline LabelType& data_label(const size_t i) noexcept { ASSERT(i < __size); return __data_label[i]; }
	constexpr inline const LabelType& data_label(const size_t i) const noexcept { ASSERT(i < __size); return __data_label[i]; }

	/// number of bytes the list contains of
	/// \return
	[[nodiscard]] constexpr uint64_t bytes() const noexcept {
		if (__data_value == nullptr)
			return size() * sizeof(LabelType);

		if (__data_label == nullptr)
			return size() * sizeof(ValueType);

		return size() * (sizeof(ValueType) + sizeof(LabelType));
	}


	/// insert an element into the list past the load factor
	/// \param e element to insert
	/// \param pos is a relative position to the thread id
	/// \param tid thread id
	constexpr void insert(const Element &e, const size_t pos, const uint32_t tid=0) noexcept {
		const size_t spos = start_pos(tid);
		if (!no_values) {
			__data_value[spos + pos] = e.value;
		}
		__data_label[spos + pos] = e.label;
	}

private:
	alignas(PAGE_SIZE) std::vector<ValueType> __data_value;
	alignas(PAGE_SIZE) std::vector<LabelType> __data_label;

	const bool no_values = false;
};



///
/// \tparam Element
/// \param out
/// \param obj
/// \return
template<class Element>
std::ostream &operator<<(std::ostream &out, const Parallel_List_T<Element> &obj) {
	for (uint64_t i = 0; i < obj.size(); ++i) {
		out << obj.data_value(i) << " " << obj.data_label(i) << " " << i << "\n" << std::flush;
	}

	return out;
}
#endif//DECODING_PARALLEL_H
