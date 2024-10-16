#ifndef DECODING_LIST_COMMON_H
#define DECODING_LIST_COMMON_H

#include "element.h"
#include "memory/memory.h"
#include "alloc/alloc.h"

struct ListConfig : public AlignmentConfig {
public:
	// if `true` all internal sorting algorithms are `std::sort`
	constexpr static bool use_std_sort = true;

	// if `true`, the call to `binary_search` will be remapped to
	// the standard implementation
	constexpr static bool use_std_binary_search = true;

	// if `true`
	constexpr static bool use_interpolation_search = false;

	// if `true` sorting is increasing, else decresing
	constexpr static bool sort_increasing_order = true;

	// if 'true', add_and_append will be allow to resize
	constexpr static bool allow_resize = true;

	static void info() noexcept {
		std::cout << " { name=\"ListConfig\""
				  << " , use_std_sort:" << use_std_sort
		          << " , use_std_sort:" << use_std_binary_search
				  << " , use_interpolation_search:" << use_interpolation_search
				  << " , sort_increasing_order:" << sort_increasing_order
				  << " }\n";
	};

} listConfig;



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

	// static functions
	Element::info();

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

	// we are good c++ devs
	typename List::value_type;
	typename List::allocator_type;
	typename List::size_type;
	typename List::difference_type;
	typename List::reference;
	typename List::const_reference;
	typename List::pointer;
	typename List::const_pointer;
	typename List::iterator;
	typename List::const_iterator;

	// static functions
	List::info();

	/// insert//append stuff
	requires requires(const size_t pos, 
			const uint32_t tid, 
			const typename List::ElementType &e) {
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
		l.random();
		l.random(i); // create a random element

		l.bytes();
	};
};
#endif


template<class List>
#if __cplusplus > 201709L
	//requires ListAble<List>
#endif
class Listview_t {
private:
	using Element = typename List::ElementType;

	// needed types
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
	constexpr Listview_t() = default;

	const size_t start;
	const size_t end;
	const List *list ;
public:
	constexpr Listview_t(const List *list,
						 const size_t start,
						 const size_t end) :
			start(start), end(end), list(list) {
		ASSERT(start < end);
		ASSERT(end < list->size());
	}

	[[nodiscard]] constexpr inline size_t size() const noexcept {
		return end - start;
	}

	[[nodiscard]] constexpr inline Element &operator[](const size_t i) noexcept {
		return list->at(start + i);
	}

	[[nodiscard]] constexpr inline const Element &operator[](const size_t i) const noexcept {
		return list->at(start + i);
	}
};

template<class List>
#if __cplusplus > 201709L
	//requires ListAble<List>
#endif
class Listview_const_t {
private:
	using Element = typename List::ElementType;

	// needed types
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
	constexpr Listview_const_t() = default;

	const size_t start;
	const size_t end;
	const List *list ;
public:
	constexpr Listview_const_t(const List *list,
	                     	   const size_t start,
	                     	   const size_t end) :
		list(list), start(start), end(end) {}

};

template<class Element,
         class Allocator=cryptanalysislib::alloc::allocator,
		 const ListConfig &config=listConfig>
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
	alignas(config.alignment*8) std::vector<Element> __data;

	/// options
	constexpr static bool use_std_sort = config.use_std_sort;
	constexpr static bool use_std_binary_search = config.use_std_binary_search;
	constexpr static bool sort_increasing_order = config.sort_increasing_order;
	constexpr static bool use_interpolation_search = config.use_interpolation_search;
	constexpr static bool allow_resize = config.allow_resize;

public:

	using L = MetaListT<Element, Allocator, listConfig>;

	// needed types
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

	// todo do the same for all other list classes
	// we are good c++ defs
	typedef Element value_type;
	typedef Allocator allocator_type; // TODO use
	typedef size_t size_type;
	typedef size_t difference_type;
	typedef value_type& reference;
	typedef const value_type& const_reference;
	typedef value_type* pointer;
	typedef const value_type* const_pointer;
	typedef typename std::vector<Element>::iterator iterator;
	typedef typename std::vector<Element>::const_iterator const_iterator ;

	using LoadType = size_t;

	// internal data types lengths
	constexpr static uint32_t ValueLENGTH = ValueType::length();
	constexpr static uint32_t LabelLENGTH = LabelType::length();

	/// size in bytes
	constexpr static uint64_t ElementBytes = Element::bytes();
	constexpr static uint64_t ValueBytes = ValueType::bytes();
	constexpr static uint64_t LabelBytes = LabelType::bytes();

	/// only valid constructor
	constexpr MetaListT(const size_t size,
	                    const uint32_t threads = 1,
	                    bool init_data = true) noexcept
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
	constexpr inline void static copy(MetaListT &out,
	                                  const MetaListT &in,
	                                  const uint32_t tid=0) noexcept {
		out.set_size(in.size());
		out.set_load(in.load());
		out.set_threads(in.threads());
		out.set_thread_block_size(in.thread_block_size());

		const std::size_t s = tid * in.threads();
		const std::size_t c = ((tid == in.threads - 1) ? in.thread_block : in.nr_elements - (in.threads - 1) * in.thread_block);
		memcpy(out.__data.data() + s, in.__data.data() + s, c * sizeof(ValueType));
	}

	/// checks if all elements in the list fulfill the equation:
	// 				label == value*matrix
	/// \param m 		the matrix.
	/// \param rewrite 	if set to true, all labels within each element will we overwritten by the recalculated.
	/// \return 		true if ech element is correct.
	constexpr bool is_correct(const MatrixType &m,
							  const bool rewrite = false) noexcept {
		bool ret = false;
		for (size_t i = 0; i < load(); ++i) {
			ret |= __data[i].is_correct(m, rewrite);
			if ((ret) && (!rewrite)) {
				return ret;
			}
		}

		return ret;
	}

	/// A little helper function to check if a list is sorted.
	/// This is very useful to assert specific states within
	/// complex cryptanalytic algorithms.
	/// \param k_lower lower bound
	/// \param k_higher upper bound
	/// \param start first index to check
	/// \param end last index to check
	/// \return if its sorted
	[[nodiscard]] constexpr bool is_sorted(const uint64_t k_lower=0,
										   const uint64_t k_higher=LabelBytes,
										   const size_t start=0,
										   const size_t end=-1ull) const noexcept {
		const size_t end_ = end==-1ull ? load() : end;
		ASSERT(start <= end_);

		for (size_t i = start+1; i < end_; ++i) {
			if (__data[i - 1].is_equal(__data[i], k_lower, k_higher)) {
				continue;
			}

			if constexpr (sort_increasing_order) {
				if (!__data[i - 1].is_lower(__data[i], k_lower, k_higher)) {
					std::cout << *this;
					return false;
				}
			} else {
				if (!__data[i - 1].is_greater(__data[i], k_lower, k_higher)) {
					return false;
				}
			}
		}

		return true;
	}

	/// same function as above, but adds/subs `t` into the list
	/// \param t element to add (on the gly)
	/// \param sub if `true` will compute `t - L[i]` instead of `t + L[i]`
	/// \param k_lower lower bound
	/// \param k_higher upper bound
	/// \param start first index to check
	/// \param end last index to check
	/// \return if its sorted
	[[nodiscard]] constexpr bool is_sorted(const LabelType &t,
	                                       const bool sub=false,
	         							   const uint64_t k_lower=0,
	                                       const uint64_t k_higher=LabelBytes,
	                                       const size_t start=0,
	                                       const size_t end=-1ull) const noexcept {
		const size_t end_ = end==-1ull ? load()-1 : end;
		ASSERT(start < end_);

		auto op = [&t, sub](const LabelType &a){
			LabelType tmp;
			if (sub) {LabelType::sub(tmp, t, a);
			} else {  LabelType::add(tmp, t, a); }
			return tmp;
		};

		for (size_t i = start+1; i < end_; ++i) {
			const auto e1 = op(__data[i-1].label);
			const auto e2 = op(__data[i].label);

			if (e1.is_equal(e2, k_lower, k_higher)) {
				continue;
			}

			if constexpr (sort_increasing_order) {
				if (!e1.is_lower(e2, k_lower, k_higher)) {
					return false;
				}
			} else {
				if (!e1.is_greater(e2, k_lower, k_higher)) {
					return false;
				}
			}
		}

		return true;
	}

	/// \return size the size of the list
	[[nodiscard]] constexpr size_t size() const noexcept { return __size; }
	/// \return the number of elements each thread enumerates
	[[nodiscard]] constexpr size_t size(const uint32_t tid) const noexcept {
		ASSERT(tid < threads());

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
	constexpr void resize(const size_t new_size) noexcept {
		if (__size == new_size) {
			return;
		}
		return __data.resize(new_size);
	}

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
	[[nodiscard]] constexpr inline size_t start_pos(const uint32_t tid=0) const noexcept {
		ASSERT(tid < threads());

		if (threads() == 1) {
			return 0;
		}

		return tid * (__data.size() / __threads);
	};
	[[nodiscard]] constexpr inline size_t end_pos(const uint32_t tid=0) const noexcept {
		ASSERT(tid < threads());
		if (threads() == 1) {
			return load();
		}

		if (tid == threads() - 1) {
			return std::max(thread_block_size() * tid, size());
		}
		return (tid + 1) * (__data.size() / __threads);
	};

	/// some setter/getter
	[[nodiscard]] constexpr inline uint32_t threads() const noexcept { return __threads; }
	[[nodiscard]] constexpr inline size_t thread_block_size() const noexcept { return __thread_block_size; }

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
	[[nodiscard]] constexpr inline auto *data() noexcept { return __data.data(); }
	[[nodiscard]] constexpr inline const auto *data() const noexcept { return __data.data(); }

	/// wrapper
	[[nodiscard]] constexpr inline ValueType *data_value() noexcept { return (ValueType *) (((uint8_t *) ptr()) + LabelBytes); }
	[[nodiscard]] constexpr inline const ValueType *data_value() const noexcept { return (ValueType *) (((uint8_t *) ptr()) + LabelBytes); }
	[[nodiscard]] constexpr inline LabelType *data_label() noexcept { return (LabelType *) __data.data(); }
	[[nodiscard]] constexpr inline const LabelType *data_label() const noexcept { return (const LabelType *) __data.data(); }
	[[nodiscard]] constexpr inline ValueType &data_value(const size_t i) noexcept {
		ASSERT(i < __size);
		return __data[i].get_value();
	}
	[[nodiscard]] constexpr inline const ValueType &data_value(const size_t i) const noexcept {
	 	ASSERT(i < __size);
	 	return __data[i].get_value();
	}
	[[nodiscard]] constexpr inline LabelType &data_label(const size_t i) noexcept {
		ASSERT(i < __size);
		return __data[i].get_label();
	}
	[[nodiscard]] constexpr inline const LabelType &data_label(const size_t i) const noexcept {
		ASSERT(i < __size);
		return __data[i].get_label();
	}

	/// operator overloading
	[[nodiscard]] constexpr inline Element &at(const size_t i) noexcept {
		ASSERT(i < size());
		return __data[i];
	}
	[[nodiscard]] constexpr inline const Element &at(const size_t i) const noexcept {
		ASSERT(i < size());
		return __data[i];
	}
	[[nodiscard]] constexpr inline Element &operator[](const size_t i) noexcept {
		ASSERT(i < size());
		return __data[i];
	}
	[[nodiscard]] constexpr inline const Element &operator[](const size_t i) const noexcept {
		ASSERT(i < size());
		return __data[i];
	}

	[[nodiscard]] constexpr inline Listview_t<L> view(const size_t i,
	                                             	   const size_t j) noexcept {
		ASSERT(j <= size());
		ASSERT(i < j);
		return Listview_t<L>(this, i, j);
	}
	[[nodiscard]] constexpr inline const Element view(const size_t i,
	                                                   const size_t j) const noexcept {
		ASSERT(j <= size());
		ASSERT(i < j);
		return Listview_const_t<L>(this, i, j);
	}

	///
	/// \param e
	/// \param i
	constexpr inline void set(Element &e, const size_t i) noexcept {
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
	constexpr void print_binary(const uint64_t pos,
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
	constexpr void print(const uint64_t pos,
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
	constexpr void print(const uint32_t value_k_lower,
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
	constexpr void print_binary(const uint32_t value_k_lower,
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
	constexpr void zero(const uint32_t tid = 0) noexcept {
		const size_t spos = start_pos(tid);
		const size_t epos = end_pos(tid);

		for (size_t i = spos; i < epos; ++i) {
			__data[i].zero();
		}

		set_load(0, tid);
	}

	/// this only sets the load counter to zero
	constexpr inline void reset(const uint32_t tid = 0) noexcept {
		set_load(0, tid);
	}

	/// remove the element at pos i.
	/// \param i
	constexpr void erase(const size_t i,
	                     const uint32_t tid = 0) noexcept {
		ASSERT(i < size());
		__data.erase(__data.begin() + i);
		__load[tid] -= 1;
	}

	/// generates a rng element
	/// NOTE: this rng elements, does not fulfill any property (e.g. label = matrix*value)
	void random(const size_t i) noexcept {
		ASSERT(i < size());
		__data[i].random();
	}

	/// generate a rng list
	constexpr void random() noexcept {
		MatrixType m;
		m.random();
		random(size(), m);
	}

	/// single threaded
	constexpr void random(const size_t list_size,
						  const MatrixType &m) noexcept {
		__data.resize(list_size);
		set_size(list_size);
		set_load(list_size);

		for (size_t i = 0; i < list_size; ++i) {
			this->at(i).random(m);
		}
	}

	// mutl
	constexpr void random(const MatrixType &m,
						  const uint32_t tid) noexcept {
		ASSERT(tid < threads());
		const size_t sp = start_pos(tid);
		const size_t ep = start_pos(tid);

		Element e{};
		set_load(ep - sp, tid);
		for (size_t i = sp; i < sp; ++i) {
			e.random(m);
			this->at(i) = e;
		}
	}

	/// iterator are useless in this class
	[[nodiscard]] constexpr inline auto begin() noexcept { return __data.begin(); }
	[[nodiscard]] constexpr inline auto end() noexcept { return __data.end(); }
	[[nodiscard]] constexpr inline auto begin() const noexcept { return __data.begin(); }
	[[nodiscard]] constexpr inline auto end() const noexcept { return __data.end(); }

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
	constexpr void insert(const Element &e,
	                      const size_t pos,
	                      const uint32_t tid = 0) noexcept {
		const size_t spos = start_pos(tid);
		ASSERT((spos+pos) < size());
		__data[spos + pos] = e;
	}

	static void info() noexcept {
		std::cout << " { name=\"MetaListT\""
				  << " , sizeof(LoadType):" << sizeof(LoadType)
				  << " , ValueLENGTH:" << ValueLENGTH
				  << " , LabelLENGTH:" << LabelLENGTH
		          << " }" << std::endl;
		ListConfig::info();
		ElementType::info();
	}
};


template<typename Element>
std::ostream &operator<<(std::ostream &out,
                         const MetaListT<Element> &obj) {
	const size_t size = obj.load() > 0 ? obj.load() : obj.size();
	for (size_t i = 0; i < size; ++i) {
		out << obj[i] << "\t pos:" << i << "\n";
	}
	return out;
}

template<typename List>
std::ostream &operator<<(std::ostream &out,
                         const Listview_t<List> &obj) {
	for (size_t i = 0; i < obj.size(); ++i) {
		out << obj[i] << "\t pos:" << i << "\n";
	}
	return out;
}

#endif//DECODING_LIST_COMMON_H
