#ifndef DECODING_LIST_COMMON_H
#define DECODING_LIST_COMMON_H

#include "element.h"
#include <cstdint>

#if __cplusplus > 201709L

/// This concept enforces the needed
/// 	- functions
/// 	- typedefs
/// 	- elements
/// each list needs
/// \tparam Element
template<class Element>
concept ListElementAble = requires(Element a) {
	typename Element::ValueType;
	typename Element::LabelType;
	typename Element::MatrixType;

	typename Element::ValueContainerType;
	typename Element::LabelContainerType;

	typename Element::ValueLimbType;
	typename Element::LabelLimbType;

	typename Element::ValueDataType;
	typename Element::LabelDataType;

	requires ElementAble<typename Element::ValueType,
	                     typename Element::LabelType,
	                     typename Element::MatrixType>;

	a.label;
	a.value;

	requires requires(const size_t i, const typename Element::MatrixType &m) {
		a.bytes();
		a.binary();// checks if the underlying container is binary
		a.zero();

		a.add(a, a, a);
		a.add(a, a, a, i, i);
		a.sub(a, a, a);

		a.is_equal(a, i, i);
		a.is_greater(a, i, i);
		a.is_lower(a, i, i);
		a.is_zero();
		a.print();
		a.print_binary(i, i, i, i);

		a.random();
		a.random(m);
	};
};


/// this concept enforces the needed
/// 	- functions
/// 	- typedefs
/// a list must implement
/// \tparam List
template<class List>
concept ListAble = requires(List l) {
	typename List::ElementType;

	/// insert//append stuff
	requires requires(const size_t pos, const uint32_t tid, const typename List::ElementType &e) {
		l.insert(e, pos, tid);
	};

	/// size stuff
	requires requires(const uint32_t i) {
		/// returns the size
		l.size();
		/// returns the size each thread needs to enumerate
		l.size(i);

		/// start/end pos of each block for each thread
		l.start_pos(i);
		l.end_pos(i);

		l.load();
		l.set_load(i);
		l.inc_load();
	};

	/// access functions
	requires requires(const uint32_t i) {
		l.data_value();
		l.data_label();
		l.data_value(i);
		l.data_label(i);
		l[i];
		l.at(i);
	};

	/// printing stuff
	requires requires(const uint32_t i) {
		/// print single elements
		l.print_binary(i, i, i, i, i);
		l.print(i, i, i, i, i);

		/// print parts of the list
		l.print_binary(i, i, i, i, i, i);
		l.print(i, i, i, i, i, i);
	};

	/// arithmetic/algorithm stuff
	requires requires(const uint32_t i) {
		l.sort();
		/// i = thread id
		l.zero(i);
		l.random(i);
		l.random();

		l.bytes();
	};
};
#endif


template<class Element>
#if __cplusplus > 201709L
    requires ListElementAble<Element>
#endif
class MetaListT {
private:
	// disable the empty constructor. So you have to specify a rough size of the list.
	// This is for optimisations reasons.
	MetaListT() : __load(0), __size(0), __threads(1){};

protected:
	/// load factor of the list
	std::vector<size_t> __load;

	/// total size of the list
	size_t __size;

	/// number of threads the
	uint32_t __threads;

	/// number of elements each thread needs to work on
	size_t __thread_block_size;

	/// internal data representation of the list.
	alignas(PAGE_SIZE) std::vector<Element> __data;

public:
	/// only valid constructor
	///
	constexpr MetaListT(const size_t size, const uint32_t threads = 1, bool init_data = true) noexcept
	    : __load(threads), __size(size), __threads(threads), __thread_block_size(size / threads) {
		if (init_data) {
			__data.resize(size);
			for (uint32_t i = 0; i < threads; i++) {
				__load[i] = 0;
			}
		}
	}

	/// parallel copy
	/// \param out
	/// \param in
	/// \param tid
	/// \return
	constexpr inline void static copy(MetaListT &out,
	                                  const MetaListT &in,
	                                  const uint32_t tid) noexcept {
		out.set_size(in.size());
		out.set_load(in.load());
		out.set_threads(in.threads());
		out.set_thread_block_size(in.thread_block_size());

		const std::size_t s = tid * in.threads();
		const std::size_t c = ((tid == in.threads - 1) ? in.thread_block : in.nr_elements - (in.threads - 1) * in.thread_block);

		memcpy(out.__data_value + s, in.__data_value + s, c * sizeof(ValueType));
		memcpy(out.__data_label + s, in.__data_value + s, c * sizeof(LabelType));
	}

	typedef Element ElementType;
	typedef typename Element::ValueType ValueType;
	typedef typename Element::LabelType LabelType;

	typedef typename Element::ValueType::LimbType ValueLimbType;
	typedef typename Element::LabelType::LimbType LabelLimbType;

	typedef typename Element::ValueContainerType ValueContainerType;
	typedef typename Element::LabelContainerType LabelContainerType;

	typedef typename Element::ValueDataType ValueDataType;
	typedef typename Element::LabelDataType LabelDataType;

	typedef typename Element::MatrixType MatrixType;

	using LoadType = size_t;

	// internal data types lengths
	constexpr static uint32_t ValueLENGTH = ValueType::LENGTH;
	constexpr static uint32_t LabelLENGTH = LabelType::LENGTH;

	/// size in bytes
	constexpr static uint64_t ElementBytes = Element::bytes();
	constexpr static uint64_t ValueBytes = ValueType::bytes();
	constexpr static uint64_t LabelBytes = LabelType::bytes();

	/// \return size the size of the list
	[[nodiscard]] constexpr size_t size() const noexcept { return __size; }
	/// \return the number of elements each thread enumerates
	[[nodiscard]] constexpr size_t size(const uint32_t tid) const noexcept {
		if (tid == threads() - 1) {
			return std::max(thread_block_size() * threads(), size());
		}

		return __thread_block_size;
	}

	/// set the size
	constexpr void set_size(const size_t new_size) noexcept {
		resize(new_size);
		__size = new_size;
	}

	/// resize the internal data container
	constexpr void resize(const size_t new_size) noexcept { return __data.resize(new_size); }

	/// set/get the load factor
	[[nodiscard]] constexpr size_t load(const uint32_t tid = 0) const noexcept {
		ASSERT(tid < threads());
		return __load[tid];
	}
	constexpr void set_load(const size_t l, const uint32_t tid = 0) noexcept {
		ASSERT(tid < threads());
		__load[tid] = l;
	}
	constexpr void inc_load(const uint32_t tid = 0) noexcept {
		ASSERT(tid < threads());
		__load[tid] += 1;
	}

	/// returning the range in which one thread is allowed to operate
	[[nodiscard]] constexpr inline size_t start_pos(const uint32_t tid) const noexcept { return tid * (__data.size() / __threads); };
	[[nodiscard]] constexpr inline size_t end_pos(const uint32_t tid) const noexcept {
		if (tid == threads() - 1) {
			return std::max(thread_block_size() * tid, size());
		}
		return (tid + 1) * (__data.size() / __threads);
	};

	/// some setter/getter
	[[nodiscard]] uint32_t threads() const noexcept { return __threads; }
	[[nodiscard]] size_t thread_block_size() const noexcept { return __thread_block_size; }
	/// NOTE: this functions resets the load factors
	constexpr void set_threads(const uint32_t new_threads) noexcept {
		__threads = new_threads;
		__thread_block_size = size() / __threads;
		__load.resize(new_threads);
		for (uint32_t i = 0; i < new_threads; i++) {
			__load[i] = 0;
		}
	}
	constexpr void set_thread_block_size(const size_t a) noexcept { __thread_block_size = a; }

	/// Get a const pointer. Sometimes useful if one ones to tell the kernel how to access memory.
	constexpr inline auto *data() noexcept { return __data.data(); }
	constexpr const auto *data() const noexcept { return __data.data(); }

	/// wrapper
	constexpr inline ValueType *data_value() noexcept { return (ValueType *) (((uint8_t *) ptr()) + LabelBytes); }
	constexpr inline const ValueType *data_value() const noexcept { return (ValueType *) (((uint8_t *) ptr()) + LabelBytes); }
	constexpr inline LabelType *data_label() noexcept { return (LabelType *) __data.data(); }
	constexpr inline const LabelType *data_label() const noexcept { return (const LabelType *) __data.data(); }
	constexpr inline ValueType &data_value(const size_t i) noexcept {
		ASSERT(i < __size);
		return __data[i].get_value();
	}
	constexpr inline const ValueType &data_value(const size_t i) const noexcept {
		ASSERT(i < __size);
		return __data[i].get_value();
	}
	constexpr inline LabelType &data_label(const size_t i) noexcept {
		ASSERT(i < __size);
		return __data[i].get_label();
	}
	constexpr inline const LabelType &data_label(const size_t i) const noexcept {
		ASSERT(i < __size);
		return __data[i].get_label();
	}

	/// operator overloading
	constexpr inline Element &at(const size_t i) noexcept {
		ASSERT(i < size());
		return __data[i];
	}
	constexpr inline const Element &at(const size_t i) const noexcept {
		ASSERT(i < size());
		return __data[i];
	}
	constexpr inline Element &operator[](const size_t i) noexcept {
		ASSERT(i < size());
		return __data[i];
	}
	constexpr inline const Element &operator[](const size_t i) const noexcept {
		ASSERT(i < size());
		return __data[i];
	}

	void set(Element &e, const uint64_t i) {
		ASSERT(i < size());
		__data[i] = e;
	}

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
		ASSERT(end <= __data.size());
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
		ASSERT(end <= __data.size());
		ASSERT(value_k_lower < value_k_higher);
		ASSERT(value_k_higher <= ValueLENGTH);
		ASSERT(label_k_lower < label_k_higher);
		ASSERT(label_k_higher <= LabelLENGTH);

		for (size_t i = start; i < end; ++i) {
			print_binary(i, value_k_lower, value_k_higher,
			             label_k_lower, label_k_higher);
		}
	}

	/// zeros the whole list
	/// and resets the load
	void zero(const uint32_t tid = 0) {
		const size_t spos = start_pos(tid);
		const size_t epos = end_pos(tid);

		for (size_t i = spos; i < epos; ++i) {
			__data[i].zero();
		}

		set_load(0, tid);
	}

	/// this only sets the load counter to zero
	void reset(const uint32_t tid = 0) {
		set_load(0, tid);
	}

	/// remove the element at pos i.
	/// \param i
	void erase(const size_t i, const uint32_t tid = 0) {
		ASSERT(i < size());
		__data.erase(__data.begin() + i);
		__load[tid] -= 1;
	}

	/// generates a random element
	/// NOTE: this random elements, does not fulfill any property (e.g. label = matrix*value)
	void random(const size_t i) {
		ASSERT(i < size());
		__data[i].random();
	}

	/// generate a full random list
	void random() {
		for (size_t i = 0; i < size(); ++i) {
			random(i);
		}
	}

	/// iterator are useless in this class
	auto begin() noexcept { return __data.begin(); }
	auto end() noexcept { return __data.end(); }

	/// returns a pointer to the internal data structure
	[[nodiscard]] constexpr Element *ptr() noexcept { return __data.data(); }

	/// return the bytes this list allocates (only the data)
	[[nodiscard]] constexpr size_t bytes() const noexcept {
		return size() * sizeof(Element);
	}

	/// insert an element into the list past the load factor
	/// \param e element to insert
	/// \param pos is a relative position to the thread id
	/// \param tid thread id
	constexpr void insert(const Element &e, const size_t pos, const uint32_t tid = 0) noexcept {
		const size_t spos = start_pos(tid);
		__data[spos + pos] = e;
	}
};


template<typename Element>
std::ostream &operator<<(std::ostream &out, const MetaListT<Element> &obj) {
	for (size_t i = 0; i < obj.size(); ++i) {
		out << obj[i] << std::endl;
	}
	return out;
}

#endif//DECODING_LIST_COMMON_H
