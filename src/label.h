#ifndef SMALLSECRETLWE_LABEL_H
#define SMALLSECRETLWE_LABEL_H

#include "helper.h"
#include "container.h"

#if __cplusplus > 201709L
///
/// \tparam Container
template<class Container>
concept LabelAble = requires(Container c) {
	typename Container::DataType;
	
	// TODO this is not true for kAryType
	// requires std::integral<typename Container::DataType>;

	// we need to enforce the existence of some fields
	//{ Container::LENGTH } -> std::convertible_to<uint32_t>;

	requires requires(const uint32_t i) {
		c[i];
		c.random();
		c.zero();
		c.is_equal(c, i, i);
		c.is_greater(c, i, i);
		c.is_lower(c, i, i);
		Container::add(c, c, c, i, i);
		Container::sub(c, c, c, i, i);
		Container::set(c, c, i, i);
		Container::cmp(c, c, i, i);
		c.neg(i, i);
		c.print(i, i);
		c.data();
		c.is_zero();

		c.print_binary(i, i);
		c.print(i, i);
	};

	// we also have to enforce the existence of some constexpr functions.
	{ Container::binary() } -> std::convertible_to<bool>;
	{ Container::size() } -> std::convertible_to<uint32_t>;
	{ Container::limbs() } -> std::convertible_to<uint32_t>;
	{ Container::bytes() } -> std::convertible_to<uint32_t>;
};
#endif

/// the class 'Label' represent A*v, where 'A' is a NTRU shaped Matrix:
///         [[ h_1, ...,      h_n],
///          [ h_n, h_1, ..., h_n-1],
///					...
///			 [ h_2, ...,      h_1]]
template<class Container>
#if __cplusplus > 201709L
	requires LabelAble<Container>
#endif
class Label_T {
public:
	typedef typename Container::DataType DataType;
	typedef typename Container::ContainerLimbType ContainerLimbType;
	typedef Container ContainerType;

	constexpr static uint32_t LENGTH = Container::LENGTH;

	void zero() { __data.zero(); }

	/// set the whole data array on random data.
	void random() { __data.random(); }

	// add v1 + v2 = v3 on the coordinates between [k_lower, k_upper)
	inline static void add(Label_T &v3, Label_T const &v1, Label_T const &v2,
	                const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) noexcept {
		Container::add(v3.__data, v1.__data, v2.__data, k_lower, k_upper);
	}

	// subtract v1 - v2 = v3 on the coordinates between [k_lower, k_upper)
	inline static void sub(Label_T &v3, Label_T const &v1, Label_T const &v2,
	                       const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) noexcept {
		Container::sub(v3.__data, v1.__data, v2.__data, k_lower, k_upper);
	}

	// negate all coordinates
	inline void neg(const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) noexcept {
		__data.neg(k_lower, k_upper);
	}

	// i think this does a 3 way comparison.
    inline static bool cmp(Label_T const &v1, Label_T const &v2,
                           const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) noexcept {
	    return Container::cmp(v1.__data, v2.__data, k_lower, k_upper);

    }

    // v1 = v3 between the coordinates [k_lower, k_upper)
    inline static void set(Label_T &v1, Label_T const &v2,
                           const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) noexcept {
	    Container::set(v1.__data, v2.__data, k_lower, k_upper);
    }

	/// checks whether this == obj on the interval [k_lower, ..., k_upper]
	/// the level of the calling 'list' object.
	/// \param obj		the object o compare with
	/// \param k_lower	lower bound to start comparison
	/// \param k_upper	upper bound
	/// \return
	inline bool is_equal(const Label_T &obj, const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) const noexcept {
		return Container::cmp(__data, obj.__data, k_lower, k_upper);
	}

	inline bool is_greater(const Label_T &obj, const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) const noexcept {
		return __data.is_greater(obj.__data, k_lower, k_upper);
	}

	inline bool is_lower(const Label_T &obj, const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) const noexcept {
	    return __data.is_lower(obj.__data, k_lower, k_upper);
    }

	/// \return true/false
	template<const uint32_t k_lower, const uint32_t k_upper>
	inline bool is_equal(const Label_T &obj) const noexcept {
		return __data.template is_equal<k_lower, k_upper>(obj.__data);
	}

	/// \return this > obj between the coordinates [k_lower, ..., k_upper]
	template<const uint32_t k_lower, const uint32_t k_upper>
	inline bool is_greater(const Label_T &obj) const noexcept {
		return __data.template is_greater<k_lower, k_upper>(obj.__data);
	}

	/// \return this < obj between the coordinates [k_lower, ..., k_upper]
	template<const uint32_t k_lower, const uint32_t k_upper>
	inline bool is_lower(const Label_T &obj) const noexcept {
		return __data.template is_lower<k_lower, k_upper>(obj.__data);
	}

	Label_T& operator =(Label_T const &obj) noexcept {
	    // fast path: do nothing if they are the same.
    	if (likely(this != &obj)) {
		    __data = obj.__data;
	    }

        return *this;
    }

	///
	bool is_zero() const noexcept{
		return __data.is_zero();
	}
	/// print the data
	void print(const uint32_t k_lower=0, const uint32_t k_upper=LENGTH) const noexcept { __data.print(k_lower, k_upper); }

	// size information
	__FORCEINLINE__ constexpr static bool binary() noexcept { return Container::binary(); }
	__FORCEINLINE__ constexpr static uint32_t size() noexcept { return Container::size(); }
	__FORCEINLINE__ constexpr static uint32_t limbs() noexcept { return Container::limbs(); }
	__FORCEINLINE__ constexpr static uint32_t bytes() noexcept { return Container::bytes(); }

	__FORCEINLINE__ auto* ptr() noexcept { return __data.data().data(); };
	__FORCEINLINE__ const auto* ptr() const noexcept { return __data.data().data(); };

	__FORCEINLINE__ Container& data() noexcept { return __data; };
	__FORCEINLINE__ const Container& data() const noexcept { return __data; };

	__FORCEINLINE__ auto data(const uint64_t i) noexcept { ASSERT(i < __data.size()); return __data[i]; };
	__FORCEINLINE__ const auto data(const uint64_t i) const noexcept { ASSERT(i < __data.size()); return __data[i]; };

	__FORCEINLINE__ auto operator [](const uint64_t i) noexcept { return data(i); };
	__FORCEINLINE__ const auto operator [](const uint64_t i) const noexcept { return __data.data(i); };
private:
	Container __data;
};

template<class Container>
std::ostream& operator<< (std::ostream &out, const Label_T<Container> &obj) {
//	int k=0;
    for (uint64_t i = 0; i < obj.size(); ++i) {
//		if (i != 0) {
//			if (__level_translation_array[k] == i) {
//				++k;
//				out << "";
//			}
//		}
		out << obj.data()[i];
	}
    return out;
}
#endif //SMALLSECRETLWE_LABEL_H
