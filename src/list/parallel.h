#ifndef DECODING_LIST_PARALLEL_H
#define DECODING_LIST_PARALLEL_H

#include <cstddef>
#include <cstdint>

#include "list/common.h"

/// This implements a data struct which can hold arbitrary amount of labels and values in two different lists.
///  	To separate the labels and values from each others, means that we are able to faster enumerate over only the labels
/// \tparam Element
template<class Element>
#if __cplusplus > 201709L
requires ListElementAble<Element>
#endif
class Parallel_List_T {
private:
	// disable the empty constructor. So you have to specify a rough size of the list. This is for optimisations reasons.
	Parallel_List_T() : nr_elements(0) {};

public:
	/// needed typedef
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

	/// internal data types lengths
	constexpr static uint32_t ValueLENGTH = ValueType::LENGTH;
	constexpr static uint32_t LabelLENGTH = LabelType::LENGTH;

	/// \param size of the whole list
	/// \param threads number of threads access this list
	/// \param thread_block size of each block for each thread.
	/// \param no_value do not allocate the value array
	constexpr explicit Parallel_List_T(const uint64_t size,
									   const uint32_t threads,
									   const uint64_t thread_block,
									   bool no_values=false) noexcept :
			nr_elements(size),
			thread_block(thread_block),
			threads(threads)
	{
		if (no_values == false) {
			__data_value = (ValueType *) cryptanalysislib_aligned_malloc(size * sizeof(ValueType), PAGE_SIZE);
			if (__data_value == NULL) {
				ASSERT("could not alloc __data_value");
				exit(1);
			}

			memset(__data_value, 0, size * sizeof(ValueType));
		}

		__data_label = (LabelType *)cryptanalysislib_aligned_malloc(size*sizeof(LabelType), PAGE_SIZE);
		if (__data_label == NULL) {
			ASSERT("could not alloc __data_label");
			exit(1);
		}

		memset(__data_label, 0, size*sizeof(LabelType));
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
	constexpr inline ValueType* data_value() noexcept { return __data_value; }
	constexpr inline const ValueType* data_value() const noexcept { return __data_value; }
	constexpr inline LabelType* data_label() noexcept { return __data_label; }
	constexpr inline const LabelType* data_label() const noexcept { return __data_label; }

	/// \return difference references to value and label
	constexpr inline ValueType& data_value(const size_t i) noexcept {  ASSERT(i < nr_elements); return __data_value[i]; }
	constexpr inline const ValueType& data_value(const size_t i) const noexcept { ASSERT(i < nr_elements); return __data_value[i]; }
	constexpr inline LabelType& data_label(const size_t i) noexcept { ASSERT(i < nr_elements); return __data_label[i]; }
	constexpr inline const LabelType& data_label(const size_t i) const noexcept { ASSERT(i < nr_elements); return __data_label[i]; }

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

	/// single threaded memcpy
	/// \param other
	/// \return
	inline Parallel_List_T<Element>& operator=(const Parallel_List_T<Element>& other) noexcept {
		// Guard self assignment
		if (this == &other) {
			return *this;
		}

		nr_elements = other.size();
		thread_block = other.thread_block;
		threads = other.threads;

		memcpy(__data_value, other.__data_value, nr_elements*sizeof(ValueType));
		memcpy(__data_label, other.__data_label, nr_elements*sizeof(LabelType));
		return *this;
	}

	/// parallel memcopy
	/// \param out
	/// \param in
	/// \param tid
	inline void static copy(Parallel_List_T &out, const Parallel_List_T &in , const uint32_t tid) noexcept {
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
	/// \param tid thread id
	/// \param start start point= first element to sort
	/// \param end  end point = last element to sort
	/// \param hash hash function
	template<typename Hash>
	void sort(uint32_t tid, const size_t start, const size_t end, Hash hash) noexcept {
		ASSERT(0);
	}

	/// zero a list
	/// \param tid
	constexpr void zero(const uint32_t tid=0) noexcept {
		ASSERT(tid < threads);
		for (size_t i = start_pos(tid); i < end_pos(tid); ++i) {
			__data_value[i].zero();
			__data_label[i].zero();
		}
	}

	/// zeros a single element
	/// \param i
	constexpr void zero_element(const size_t i) noexcept {
		ASSERT(i < nr_elements);
		__data_value[i].zero();
		__data_label[i].zero();
	}

	/// iterator are useless in this class
	auto begin() noexcept { return nullptr; }
	auto end() noexcept { return nullptr; }

	/// number of bytes the list contains of
	/// \return
	[[nodiscard]] constexpr uint64_t bytes() const noexcept {
		if (__data_value == nullptr)
			return nr_elements * sizeof(LabelType);

		if (__data_label == nullptr)
			return nr_elements * sizeof(ValueType);

		return nr_elements * (sizeof(ValueType) + sizeof(LabelType));
	}

public:
	ValueType *__data_value = nullptr;
	LabelType *__data_label = nullptr;

	/// total number of elements the list can hold
	size_t nr_elements;

	/// number of elements each thread needs to handle
	uint64_t thread_block;

	/// number of threads, which access the list in parallel.
	uint64_t threads;
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
