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
	/// \param no_value do not allocate the value array
	constexpr explicit Parallel_List_T(const size_t size,
									   const uint32_t threads,
									   bool no_values=false) noexcept :
	   MetaListT<Element>(size, threads, false),
	   no_values(no_values)
	{
		if (!no_values) {
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

		if (other.data_value()) {
			memcpy(__data_value, other.__data_value, size() * sizeof(ValueType));
		}
		memcpy(__data_label, other.__data_label, size()*sizeof(LabelType));
		return *this;
	}

	/// parallel copy
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

		if (out.data_value()) {
			memcpy(out.__data_value + s, in.__data_value + s, c * sizeof(ValueType));
		}
		memcpy(out.__data_label+s, in.__data_value+s, c*sizeof(LabelType));
	}


	///
	void random(const size_t i) {
		ASSERT(i < size());
		__data_value[i].random();
		__data_label[i].random();
	}

	/// generate a random list
	constexpr void random() {
		MatrixType m;
		m.random();
		random(size(), m);
	}

	///
	constexpr void random(const size_t list_size,
						  const MatrixType &m) {
		for (size_t i = 0; i < list_size; ++i) {
			Element e{};
			e.random(m);
			__data_value[i] = e.get_value();
			__data_label[i] = e.get_label();
		}

		set_load(list_size);
	}

	/// checks if all elements in the list fulfill the equation:
	// 				label == value*matrix
	/// \param m 		the matrix.
	/// \param rewrite 	if set to true, all labels within each element will we overwritten by the recalculated.
	/// \return 		true if ech element is correct.
	constexpr bool is_correct(const MatrixType &m,
							  const bool rewrite = false) noexcept {
		for (size_t i = 0; i < load(); ++i) {
			LabelType tmp;
			m.mul(tmp, data_value(i));

			bool ret = tmp.is_equal(data_label(i));
			if ((!ret) && (!rewrite)) {
				return ret;
			}
		}

		return true;
	}

	/// NOTE: single threaded
	/// \tparam Hash hash function type, needed for the bucket sort
	/// \param start start point= first element to sort
	/// \param end  end point = last element to sort
	void sort(const size_t start=0,
	          const size_t end=load(),
	          const uint32_t k_lower=0,
			  const uint32_t k_higher=LabelLENGTH) noexcept {
		ASSERT(start < end);
		size_t _end = end;
		if (_end > size()) {
			_end = load();
		}
		std::sort(__data_label.begin() + start, __data_label.begin() + _end,
		          [k_lower, k_higher](const auto &e1, const auto &e2) {
#if !defined(SORT_INCREASING_ORDER)
			          return e1.is_lower(e2, k_lower, k_higher);
#else
			          return e1.is_greater(e2, k_lower, k_higher);
#endif
		          });

		ASSERT(is_sorted(k_lower, k_higher));
	          }

	/// NOTE: single threded
	/// \tparam Hash hash function type, needed for the bucket sort
	/// \param tid thread id 
	/// \param hash hash function
	template<typename Hash>
	void sort(const uint32_t tid=0) noexcept {
		const size_t start = start_pos(tid);
		const size_t end = start_pos(tid);
		auto hash = [](const LabelType &e) -> uint64_t {
		  return e.hash();
		};
		ska_sort(__data_label.begin() + start,
		         __data_label.begin() + end, hash);
	}

	/// zero a list
	/// \param tid
	constexpr void zero(const uint32_t tid=0) noexcept {
		ASSERT(tid < __threads);
		for (size_t i = start_pos(tid); i < end_pos(tid); ++i) {
			if (data_value()) {
				__data_value[i].zero();
			}

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
	[[nodiscard]] auto begin() noexcept { ASSERT(false); return nullptr; }
	[[nodiscard]] auto end() noexcept { ASSERT(false); return nullptr; }

	[[nodiscard]] constexpr inline ValueType* data_value() noexcept { return (ValueType *)__data_value.data() ; }
	[[nodiscard]] constexpr inline const ValueType* data_value() const noexcept { return (ValueType *)__data_value.data(); }
	[[nodiscard]] constexpr inline LabelType* data_label() noexcept { return (LabelType *)__data_label.data(); }
	[[nodiscard]] constexpr inline const LabelType* data_label() const noexcept { return (const LabelType *)__data_label.data(); }

	[[nodiscard]] constexpr inline ValueType& data_value(const size_t i) noexcept {  ASSERT(i < __size); return __data_value[i]; }
	[[nodiscard]] constexpr inline const ValueType& data_value(const size_t i) const noexcept { ASSERT(i < __size); return __data_value[i]; }
	[[nodiscard]] constexpr inline LabelType& data_label(const size_t i) noexcept { ASSERT(i < __size); return __data_label[i]; }
	[[nodiscard]] constexpr inline const LabelType& data_label(const size_t i) const noexcept { ASSERT(i < __size); return __data_label[i]; }

	/// \return number of bytes the list contains of
	[[nodiscard]] constexpr inline uint64_t bytes() const noexcept {
		if (__data_value == nullptr) {
			return size() * sizeof(LabelType);
		}

		if (__data_label == nullptr) {
			return size() * sizeof(ValueType);
		}

		return size() * (sizeof(ValueType) + sizeof(LabelType));
	}


	/// insert an element into the list past the load factor
	/// \param e element to insert
	/// \param pos is a relative position to the thread id
	/// \param tid thread id
	constexpr void insert(const Element &e,
	                      const size_t pos,
	                      const uint32_t tid=0) noexcept {
		const size_t spos = start_pos(tid);
		if (!no_values) {
			__data_value[spos + pos] = e.value;
		}

		__data_label[spos + pos] = e.label;
	}


	constexpr bool is_sorted(const uint64_t k_lower=0,
							 const uint64_t k_higher=LabelBytes) const {

		for (size_t i = 1; i < load(); ++i) {
			if (__data_label[i - 1].is_equal(__data_label[i], k_lower, k_higher)) {
				continue;
			}

#if !defined(SORT_INCREASING_ORDER)
			if (!__data_label[i - 1].is_lower(__data_label[i], k_lower, k_higher)) {
				return false;
			}
#else
			if (!__data_label[i - 1].is_greater(__data_label[i], k_lower, k_higher)) {
				return false;
			}
#endif
		}
		return true;
	}

private:
	alignas(CUSTOM_PAGE_SIZE) std::vector<ValueType> __data_value;
	alignas(CUSTOM_PAGE_SIZE) std::vector<LabelType> __data_label;

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
