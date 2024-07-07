#ifndef SMALLSECRETLWE_ELEMENT_H
#define SMALLSECRETLWE_ELEMENT_H

// global includes
#include <array>
#include <cstdint>

#ifdef USE_FPLLL
// dependencies include
#include "fplll/nr/matrix.h"
#include "fplll/util.h"// needed for 'vector_matrix_product'
using namespace fplll;
#endif

// local includes
#include "container/binary_packed_vector.h"
#include "helper.h"
#include "matrix/matrix.h"

#if __cplusplus > 201709L

///
/// \tparam Container
template<class Container>
concept LabelAble = requires(Container c) {
	typename Container::DataType;

	// we need to enforce the existence of some fields/functions
	Container::length;
	Container::modulus;

	requires requires(const uint32_t i,
	                  const size_t s) {
		c[i];
		c.get(i);
		c.set(i, i);
		c.random();
		c.zero();
		c.data();
		c.ptr();
		c.ptr(s);

		c.is_equal(c, i, i);
		c.is_equal(c);
		c.is_greater(c, i, i);
		c.is_greater(c);
		c.is_lower(c, i, i);
		c.is_lower(c);
		c.is_zero(i, i);
		c.is_zero();
		c.neg(i, i);

		Container::add(c, c, c, i, i);
		Container::sub(c, c, c, i, i);
		Container::set(c, c, i, i);
		Container::cmp(c, c, i, i);

		c.print_binary(i, i);
		c.print(i, i);
	};

	/// limb arithmetic stuff
	requires requires(const typename Container::LimbType a,
	                  const uint8x32_t b) {
		Container::add_T(a, a);
		Container::sub_T(a, a);
		Container::mod_T(a);
		Container::mul_T(a, a);
		Container::neg_T(a);
		Container::scalar_T(a, a);
		Container::popcnt_T(a);

		Container::add256_T(b, b);
		Container::sub256_T(b, b);
		Container::mod256_T(b);
		Container::mul256_T(b, b);
		Container::neg256_T(b);
	};

	// we also have to enforce the existence of some constexpr functions.
	{ Container::binary() } -> std::convertible_to<bool>;
	{ Container::length() } -> std::convertible_to<uint32_t>;
	{ Container::size() } -> std::convertible_to<uint32_t>;
	{ Container::limbs() } -> std::convertible_to<uint32_t>;
	{ Container::bytes() } -> std::convertible_to<uint32_t>;
};

/// Requirements for a base data container.
/// \tparam Container
template<class Container>
concept ValueAble = requires(Container c) {
	typename Container::DataType;
	typename Container::LimbType;

	// we need to enforce the existence of some fields/functions
	Container::modulus;
	Container::length;

	requires requires(const uint32_t i,
	                  const size_t s) {
		/// init/getter/setter
		c[i];
		c.get(i);
		c.set(i, i);
		Container::set(c, c, i, i);
		c.random();
		c.zero();
		c.data();
		c.ptr();
		c.ptr(s);

		/// comparison
		Container::cmp(c, c, i, i);
		c.is_equal(c, i, i);
		c.is_equal(c);
		c.is_greater(c, i, i);
		c.is_greater(c);
		c.is_lower(c, i, i);
		c.is_lower(c);
		c.is_zero(i, i);
		c.is_zero();

		/// arithmetic
		Container::add(c, c, c, i, i);
		Container::sub(c, c, c, i, i);
		c.neg(i, i);

		/// printing stuff
		c.print_binary(i, i);
		c.print(i, i);
	};

	/// limb arithmetic stuff
	requires requires(const typename Container::LimbType a,
	                  const uint8x32_t b) {
		Container::add_T(a, a);
		Container::sub_T(a, a);
		Container::mod_T(a);
		Container::mul_T(a, a);
		Container::neg_T(a);
		Container::scalar_T(a, a);
		Container::popcnt_T(a);

		// simd stuff 
		Container::add256_T(b, b);
		Container::sub256_T(b, b);
		Container::mod256_T(b);
		Container::mul256_T(b, b);
		Container::neg256_T(b);
	};

	// we also have to enforce the existence of some constexpr functions.
	{ Container::binary() } -> std::convertible_to<bool>;
	{ Container::length() } -> std::convertible_to<uint32_t>;
	{ Container::size() } -> std::convertible_to<uint32_t>;
	{ Container::limbs() } -> std::convertible_to<uint32_t>;
	{ Container::bytes() } -> std::convertible_to<uint32_t>;
};

template<class Value, class Label, class Matrix>
concept ElementAble = requires(Value v, Label l) {
	/// needed typedefs
	typename Value::ContainerType;
	typename Label::ContainerType;

	requires ValueAble<typename Value::ContainerType>;
	requires LabelAble<typename Label::ContainerType>;

	requires MatrixAble<Matrix>;
};
#endif

/// \tparam Value
/// \tparam Label
template<class Value, class Label, class Matrix>
#if __cplusplus > 201709L
    requires ElementAble<Value, Label, Matrix>
#endif
class Element_T {
public:
	// Internal Datatype
	typedef Value ValueType;
	typedef Label LabelType;
	typedef Matrix MatrixType;

	typedef typename Value::ContainerType ValueContainerType;
	typedef typename Label::ContainerType LabelContainerType;

	typedef typename Value::DataType ValueDataType;
	typedef typename Label::DataType LabelDataType;

	typedef typename Value::LimbType ValueLimbType;
	typedef typename Label::LimbType LabelLimbType;

	// internal data types lengths
	constexpr static uint32_t ValueLENGTH = Value::length();
	constexpr static uint32_t LabelLENGTH = Label::length();


	/// normal constructor. Initialize everything with zero.
	Element_T() noexcept : label(), value() { this->zero(); }

	/// copy constructor
	Element_T(const Element_T &a) noexcept
	    : label(a.label), value(a.value) {}

	/// zero out the element.
	void zero() noexcept {
		value.zero();
		label.zero();
	}

	///
	void random() noexcept {
		value.random();
		label.random();
	}

	/// generate a random element.
	/// \param m 	Matrix
	void random(const MatrixType &m) noexcept {
		value.random();
		recalculate_label(m);
	}

	/// returns the position of the i-th which is zero counted from left, where the first start '0' are skipped
	/// \param i		pos
	/// \param start
	/// \return
	size_t ith_value_left_zero_position(const size_t i, const size_t start = 0) const noexcept {
		uint64_t count = 0;
		for (uint64_t j = 0; j < value_size(); ++j) {
			if (get_value().data()[j] == 0)
				count += 1;

			if (count == (i + start + 1))
				return j;
		}

		return uint64_t(-1);
	}

	/// same as the function above. Only counting from right.
	/// \param i
	/// \param start
	/// \return
	size_t ith_value_right_zero_position(const size_t i, const size_t start = 0) const noexcept {
		uint64_t count = 0;
		for (uint64_t j = value_size(); j > 0; --j) {
			if (get_value().data()[j - 1] == 0)
				count += 1;

			if (count == (i + start + 1))
				return j - 1;
		}

		return uint64_t(-1);
	}

	/// recalculated the label. Useful if vou have to negate/change some coordinates of the label for an easier merging
	/// procedure.
	/// \param m Matrix
	constexpr void recalculate_label(const MatrixType &m) noexcept {
		m.mul(label, value);
	}

	/// checks if label == value*m
	/// \param m
	/// \param rewrite if set to true, it will overwrite the old label with the new recalculated one.
	/// \return true if the label is correct under the given matrix.
	constexpr bool is_correct(const MatrixType &m,
	                          const bool rewrite = false) noexcept {
		Label tmp;
		m.mul(tmp, value);

		bool ret = tmp.is_equal(label, 0, label_size());
		if (rewrite) {
			label = tmp;
		}

		return ret;
	}

	/// e3 = e1 + e2 iff \forall r \in \[k_lower, ..., k_upper\] |e3[r]| < norm
	/// return 'true' if the element needs to be filtered out.
	///	__IMPORTANT__ This functions interrupts the current calculations if the 'norm' limit is violated.
	///				So it stops the calculation resulting in a __NOT__ correct 'e3'
	/// \param e3	Output Label
	/// \param e1	Input Label
	/// \param e2	InputLabel
	/// \param norm
	/// \return 	'true' if the elements needs to be filtered out. E.g. it \exists a coordinate r s.t. |Value[r]| > norm.
	///					it also stops the calculation so the result __MUST__ __NOT__ be correct in this case.
	///				else 'false'
	constexpr static bool add(Element_T &e3,
	                          Element_T const &e1,
	                          Element_T const &e2,
	                          const uint32_t k_lower,
	                          const uint32_t k_upper,
	                          const uint32_t norm = -1) noexcept {
		Label::add(e3.label, e1.label, e2.label, k_lower, k_upper);
		return Value::add(e3.value, e1.value, e2.value, 0, ValueLENGTH, norm);
	}

	///  Useful if you do not want to filter in your tree and want additional performance.
	constexpr static void add(Element_T &e3,
	                          Element_T const &e1,
	                          Element_T const &e2) noexcept {
		Label::add(e3.label, e1.label, e2.label);
		Value::add(e3.value, e1.value, e2.value);
	}

	///  Useful if you do not want to filter in your tree and want additional performance.
	constexpr static void sub(Element_T &e3,
	                          Element_T const &e1,
	                          Element_T const &e2) noexcept {
		LabelContainerType::sub(e3.label, e1.label, e2.label);
		ValueContainerType::sub(e3.value, e1.value, e2.value);
	}
	/// checks if this.label == obj.label on the coordinates [k_lower, k_upper]
	/// \param obj		second element
	/// \param k_lower  lower coordinate
	/// \param k_upper  higher coordinate
	/// \return true/false
	constexpr inline bool is_equal(const Element_T &obj, const uint32_t k_lower = 0, const uint32_t k_upper = LabelLENGTH) const noexcept {
		// No need to assert, because everything will be done inside the called function 'value.is_equal(...)'
		return label.is_equal(obj.label, k_lower, k_upper);
	}

	/// \return this->label > obj.label between the coordinates [k_lower, ..., k_upper]
	constexpr inline bool is_greater(const Element_T &obj, const uint32_t k_lower = 0, const uint32_t k_upper = LabelLENGTH) const noexcept {
		// No need to assert, because everything will be done inside the called function 'value.is_greater(...)'
		return label.is_greater(obj.label, k_lower, k_upper);
	}

	/// \return this->label < obj.label between the coordinates [k_lower, ..., k_upper]
	constexpr inline bool is_lower(const Element_T &obj, const uint32_t k_lower = 0, const uint32_t k_upper = LabelLENGTH) const noexcept {
		// No need to assert, because everything will be done inside the called function 'value.is_lower(...)'
		return label.is_lower(obj.label, k_lower, k_upper);
	}

	/// \return true/false
	template<const uint32_t k_lower, const uint32_t k_upper>
	constexpr inline bool is_equal(const Element_T &obj) const noexcept {
		// No need to assert, because everything will be done inside the called function 'value.is_equal(...)'
		return label.template is_equal<k_lower, k_upper>(obj.label);
	}

	/// \return this->label > obj.label between the coordinates [k_lower, ..., k_upper]
	template<const uint32_t k_lower, const uint32_t k_upper>
	constexpr inline bool is_greater(const Element_T &obj) const noexcept {
		return label.template is_greater<k_lower, k_upper>(obj.label);
	}

	/// \return this->label < obj.label between the coordinates [k_lower, ..., k_upper]
	template<const uint32_t k_lower, const uint32_t k_upper>
	constexpr inline bool is_lower(const Element_T &obj) const noexcept {
		return label.template is_lower<k_lower, k_upper>(obj.label);
	}

	/// Assignment operator implementing copy assignment
	/// see https://en.cppreference.com/w/cpp/language/operators
	///
	/// \param obj to copy from
	/// \return this
	constexpr inline Element_T &operator=(Element_T const &obj) noexcept {
		// self-assignment check expected
		if (this != &obj) {
			// now we can copy it
			label = obj.label;
			value = obj.value;
		}

		return *this;
	}

	///
	/// \return true if either the value or label is zero on all coordinates
	[[nodiscard]] constexpr bool is_zero() const noexcept {
		bool ret = false;

		ret |= value.is_zero();
		ret |= label.is_zero();
		return ret;
	}

	/// Assignment operator implementing move assignment
	/// see https://en.cppreference.com/w/cpp/language/move_assignment
	/// \param obj
	/// \return
	Element_T &operator=(Element_T &&obj) noexcept {
		if (this != &obj) {// self-assignment check expected really?
			value = std::move(obj.value);
			label = std::move(obj.label);
		}

		return *this;
	}

	/// \param obj
	/// \return
	inline bool operator!=(Element_T const &obj) const noexcept {
		return !label.is_equal(obj.label);
	}

	///
	/// \param obj
	/// \return
	inline bool operator==(Element_T const &obj) const noexcept {
		return label.is_equal(obj.label);
	}

	///
	/// \param obj
	/// \return
	inline bool operator>(Element_T const &obj) const noexcept {
		return label.is_greater(obj.label);
	}

	///
	/// \param obj
	/// \return
	inline bool operator>=(Element_T const &obj) const noexcept {
		return !label.is_lower(obj.label);
	}

	///
	/// \param obj
	/// \return
	inline bool operator<(Element_T const &obj) const noexcept {
		return label.is_lower(obj.label);
	}

	///
	/// \param obj
	/// \return
	inline bool operator<=(Element_T const &obj) const noexcept {
		return !label.is_greater(obj.label);
	}

	/// prints stuff
	constexpr void print() const noexcept {
		label.print();
		value.print();
	}

	/// print the internal data
	/// \param k_lower lower dimension
	/// \param k_upper upper dimension
	constexpr void print(const uint64_t k_lower_label,
	                     const uint64_t k_upper_label,
	                     const uint64_t k_lower_value,
	                     const uint64_t k_upper_value) const noexcept {
		label.print(k_lower_label, k_upper_label);
		value.print(k_lower_value, k_upper_value);
	}

	/// prints stuff
	constexpr void print_binary() const noexcept {
		label.print_binary();
		value.print_binary();
	}

	/// print the internal data
	/// \param k_lower_label
	/// \param k_upper_label
	/// \param k_lower_value
	/// \param k_upper_value
	constexpr void print_binary(const uint64_t k_lower_label,
	                            const uint64_t k_upper_label,
	                            const uint64_t k_lower_value,
	                            const uint64_t k_upper_value) const noexcept {
		label.print_binary(k_lower_label, k_upper_label);
		value.print_binary(k_lower_value, k_upper_value);
	}

	constexpr Value &get_value() noexcept { return value; }
	constexpr const Value &get_value() const noexcept { return value; }
	constexpr auto get_value(const size_t i) noexcept {
		ASSERT(i < value.size());
		return value.data(i);
	}
	constexpr auto get_value(const size_t i) const noexcept {
		ASSERT(i < value.size());
		return value.data(i);
	}

	constexpr Label &get_label() noexcept { return label; }
	constexpr const Label &get_label() const noexcept { return label; }
	constexpr auto get_label(const uint64_t i) noexcept {
		ASSERT(i < label.size());
		return label.data(i);
	}
	constexpr auto get_label(const uint64_t i) const noexcept {
		ASSERT(i < label.size());
		return label.data(i);
	}

	constexpr void set_value(const Value &v) noexcept { value = v; }
	constexpr void set_label(const Label &l) noexcept { label = l; }


	/// returns true of both underlying data structs are binary
	constexpr static bool binary() noexcept { return Label::binary() && Value::binary(); }
	constexpr static uint32_t label_size() noexcept { return Label::size(); }
	constexpr static uint32_t value_size() noexcept { return Value::size(); }
	constexpr static uint32_t size() noexcept { return Value::size() + Label::size(); }
	constexpr static uint32_t bytes() noexcept { return ValueContainerType::copyable_ssize() + LabelContainerType::copyable_ssize(); }

	///
	__FORCEINLINE__ constexpr auto *label_ptr() noexcept { return label.ptr(); }
	__FORCEINLINE__ constexpr auto *value_ptr() noexcept { return value.ptr(); }

	__FORCEINLINE__ constexpr auto label_ptr(const size_t i) const noexcept { return label.ptr(i); }
	__FORCEINLINE__ constexpr auto value_ptr(const size_t i) const noexcept { return value.ptr(i); }

public:
	Label label;
	Value value;
};

/// print operator
/// \tparam Value type of a value
/// \tparam Label type of label
/// \param out
/// \param obj	input element
/// \return
template<class Value, class Label, class Matrix>
std::ostream &operator<<(std::ostream &out, const Element_T<Value, Label, Matrix> &obj) {
	out << "V: " << obj.get_value();
	out << ", L: " << obj.get_label() << "\n";
	return out;
}
#endif//SMALLSECRETLWE_ELEMENT_H
