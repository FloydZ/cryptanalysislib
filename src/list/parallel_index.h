#ifndef DECODING_PARALLEL_INDEX_H
#define DECODING_PARALLEL_INDEX_H

#include "list/common.h"

/// nearly the same as
///		 `Parallel_List_T`
/// with the main difference that this list only saves the label,
/// and not the Value. But instead a counter/index is saved which
/// NOTE: does not track the load
/// \tparam Element
template<class Element, uint32_t nri>
#if __cplusplus > 201709L
	requires ListElementAble<Element>
#endif
class Parallel_List_IndexElement_T {
private:
	// disable the empty constructor. So you have to specify a rough size of the list. This is for optimisations reasons.
	Parallel_List_IndexElement_T() : nr_elements(0) {};

public:
	/// Basic typedefs
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

	/// additional Typedefs
	using IndexType = std::array<uint32_t, nri>;
	using InternalElementType = std::pair<LabelType, IndexType>;

	/// internal data types lengths
	constexpr static uint32_t ValueLENGTH = ValueType::LENGTH;
	constexpr static uint32_t LabelLENGTH = LabelType::LENGTH;

	///
	constexpr ~Parallel_List_IndexElement_T() noexcept {}

	/// \param size total number of elements in the list
	/// \param threads
	/// \param thread_block
	constexpr explicit Parallel_List_IndexElement_T(const size_t size,
										  			const uint32_t threads,
										  			const size_t thread_block) noexcept :
			nr_elements(size), thread_block(thread_block), threads(threads)
	{
		ASSERT(threads > 0);
		ASSERT(size > 0);
		ASSERT(thread_block > 0);
		ASSERT(thread_block <= size);
		__data.resize(size);
	}

	/// \return size the size of the list
	[[nodiscard]] constexpr size_t size() const noexcept { return nr_elements; }
	/// \return the number of elements each thread enumerates
	[[nodiscard]] constexpr inline size_t size(const uint32_t tid) const noexcept { return thread_block; }

	/// Useless function, as this list class does not track its load
	[[nodiscard]] constexpr size_t get_load() const noexcept { return 0; }
	constexpr void set_load(const size_t l) noexcept { (void)l; }

	// returning the range in which one thread is allowed to operate
	[[nodiscard]] constexpr inline size_t start_pos(const uint32_t tid) const noexcept { ASSERT(tid < threads); return tid * thread_block; };
	[[nodiscard]] constexpr inline size_t end_pos(const uint32_t tid) const noexcept  { ASSERT(tid < threads); return (tid+1) * thread_block; };

	/// useless, as not really implementable, so actually they return the index
	inline auto& data_value(const size_t i) noexcept {  ASSERT(i < nr_elements); return __data[i].second; }
	inline const auto& data_value(const size_t i) const noexcept { ASSERT(i < nr_elements); return __data[i].second; }
	/// Only these function do make sense
	inline LabelType& data_label(const size_t i) noexcept { ASSERT(i < nr_elements); return __data[i].first; }
	inline const LabelType& data_label(const size_t i) const noexcept { ASSERT(i < nr_elements); return __data[i].first; }

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

		std::cout << data_value(pos) << std::endl;
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

		std::cout << data_value(pos) << std::endl;
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

	/// copy operator
	/// \param other
	/// \return
	inline Parallel_List_IndexElement_T& operator=(const Parallel_List_IndexElement_T& other) noexcept {
		// Guard self assignment
		if (this == &other)
			return *this;

		nr_elements = other.size();
		thread_block = other.thread_block;
		threads = other.threads;

		memcpy(__data, other.data(), nr_elements*sizeof(InternalElementType));
		return *this;
	}

	/// copy function
	/// \param out
	/// \param in
	/// \param tid
	inline void static copy(Parallel_List_IndexElement_T &out,
	                        const Parallel_List_IndexElement_T &in,
	                        const uint32_t tid) noexcept {
		out.nr_elements = in.size();
		out.thread_block = in.thread_block;

		const std::size_t s = in.start_pos(tid);
		const std::size_t c = in.end_pos(tid);

		memcpy(out.__data+s, in.__data+s, c*sizeof(InternalElementType));
	}

	/// not implemented
	void sort() noexcept {
		ASSERT(0);
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
		ASSERT(tid < threads);

		if (load >= thread_block)
			return;

		LabelType::add(__data[start_pos(tid) + load].first, l1, l2);
		__data[start_pos(tid) + load].second[0] = i1;
		__data[start_pos(tid) + load].second[1] = i2;
		load += 1;
	}

	/// some useful stuff
	auto begin() noexcept { return __data.begin(); }
	auto end() noexcept { return __data.end(); }

	/// return the number of bytes needed for this list
	[[nodiscard]] constexpr uint64_t bytes() const noexcept {
		return __data.size() * sizeof(InternalElementType);
	}

public:

	/// total numbers of elements the list can holds
	size_t nr_elements;

	/// number of elements each thread needs to handle
	size_t thread_block;

	/// number of threads, which can access the list in parallel
	uint32_t threads;

	/// data container
	alignas(PAGE_SIZE) std::vector<InternalElementType> __data;
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
