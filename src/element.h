#ifndef SMALLSECRETLWE_ELEMENT_H
#define SMALLSECRETLWE_ELEMENT_H

// global includes
#include <array>
#include <cstdint>

#ifdef USE_FPLLL
// dependencies include
#include "fplll/nr/matrix.h"
#include "fplll/util.h"         // needed for 'vector_matrix_product'
using namespace fplll;
#endif

// local includes
#include "helper.h"
#include "container.h"
#include "value.h"
#include "label.h"
#include "matrix.h"

#if __cplusplus > 201709L
template<class Value, class Label, class Matrix>
concept ElementAble = requires(Value v, Label l) {
	typename Value::ContainerType;
	typename Label::ContainerType;

	requires ValueAble<typename Value::ContainerType>;
	requires LabelAble<typename Label::ContainerType>;

	// Value requirements
	requires requires(const uint32_t i) {
		v[i];
		v.random();
		v.zero();
		v.is_equal(v, i, i);
		v.is_greater(v, i, i);
		v.is_lower(v, i, i);
		Value::add(v, v, v, i, i, i);
		Value::sub(v, v, v, i, i);
		Value::set(v, v, i, i);
		Value::cmp(v, v, i, i);
		v.neg(i, i);
		v.print(i, i);
		v.size();
		v.data();
		v.is_zero();
	};

	// Label requirements
	requires requires(const uint32_t i) {
		l[i];
		l.random();
		l.zero();
		l.is_equal(l, i, i);
		l.is_greater(l, i, i);
		l.is_lower(l, i, i);
		Label::add(l, l, l, i, i);
		Label::sub(l, l, l, i, i);
		Label::set(l, l, i, i);
		Label::cmp(l, l, i, i);
		l.neg(i, i);
		l.print(i, i);
		l.size();
		l.data();
		v.is_zero();
	};


	// we also have to enforce the existence of some constexpr functions.
	{ Value::binary() } -> std::convertible_to<bool>;
	{ Value::size() } -> std::convertible_to<uint32_t>;
	{ Value::limbs() } -> std::convertible_to<uint32_t>;
	{ Value::bytes() } -> std::convertible_to<uint32_t>;
	{ Label::binary() } -> std::convertible_to<bool>;
	{ Label::size() } -> std::convertible_to<uint32_t>;
	{ Label::limbs() } -> std::convertible_to<uint32_t>;
	{ Label::bytes() } -> std::convertible_to<uint32_t>;
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
	// Internal Datatypes
	typedef Value ValueType;
	typedef Label LabelType;
	typedef Matrix MatrixType;

	typedef typename Value::ContainerType ValueContainerType;
	typedef typename Label::ContainerType LabelContainerType;

	typedef typename Value::DataType ValueDataType;
	typedef typename Label::DataType LabelDataType;

	typedef typename Value::ContainerLimbType ValueContainerLimbType;
	typedef typename Label::ContainerLimbType LabelContainerLimbType;

	// internal data types lengths
	constexpr static uint32_t ValueLENGTH = Value::LENGTH;
	constexpr static uint32_t LabelLENGTH = Label::LENGTH;


	/// normal constructor. Initialize everything with zero.
    Element_T() : value(), label() { this->zero(); }

    /// copy constructor
	Element_T(const Element_T& a) : value(a.value), label(a.label) {}

	/// zero out the element.
    void zero() {
        value.zero();
        label.zero();
    }

    /// returns the position of the i-th which is zero counted from left, where the first start '0' are skipped
	/// \param i		pos
	/// \param start
	/// \return
    uint64_t ith_value_left_zero_position(const uint64_t i, const uint64_t start = 0) const {
	    uint64_t count = 0;
	    for (uint64_t j = 0; j < value_size(); ++j) {
			if (get_value().data()[j] == 0)
				count += 1;

			if (count == (i+start+1))
				return j;
	    }

	    return uint64_t(-1);
	}

	/// same as the function above. Only counting from right.
	/// \param i
	/// \param start
	/// \return
	uint64_t ith_value_right_zero_position(const uint64_t i, const uint64_t start = 0) const {
		uint64_t count = 0;
		for (uint64_t j = value_size(); j > 0; --j) {
			if (get_value().data()[j - 1] == 0)
				count += 1;

			if (count == (i+start+1))
				return j-1;
		}

		return uint64_t(-1);
	}

	/// generate a random element.
	/// \param m 	Matrix
	void random(const Matrix_T<Matrix> &m) {
		value.random();
		recalculate_label(m);
	}

	/// recalculated the label. Useful if vou have to negate/change some coordinates of the label for an easier merging
	/// procedure.
	/// \param m Matrix
    void recalculate_label(const Matrix_T<Matrix> &m){
	    new_vector_matrix_product<Label, Value, Matrix>(label, value, m);
	}

	/// checks if label == value*m
	/// \param m
	/// \param rewrite if set to true, it will overwrite the old label with the new recalculated one.
	/// \return true if the label is correct under the given matrix.
	bool is_correct(const Matrix_T<Matrix> &m, const bool rewrite=false) {
    	Label tmp = label;
    	recalculate_label(m);

    	bool ret = tmp.is_equal(label, 0, label_size());
    	if(!rewrite) {
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
    static bool add(Element_T &e3, Element_T const &e1, Element_T const &e2,
                    const uint32_t k_lower, const uint32_t k_upper, const uint32_t norm=-1) {
        Label::add(e3.label, e1.label, e2.label, k_lower, k_upper);
	    return Value::add(e3.value, e1.value, e2.value, 0, ValueLENGTH, norm);
    }

	/// same as the function above but always return false, meaning to NOT filter out this element.
	///  Useful if you do not want to filter in your tree and want additional performance.
	static void add(Element_T &e3, Element_T const &e1, Element_T const &e2) {
		LabelContainerType::add(e3.label.data(), e1.label.data(), e2.label.data());
		ValueContainerType::add(e3.value.data(), e1.value.data(), e2.value.data());
	}

    /// checks if this.label == obj.label on the coordinates [k_lower, k_upper]
    /// \param obj		second element
    /// \param k_lower  lower coordinate
    /// \param k_upper  higher coordinate
    /// \return true/false
    inline bool is_equal(const Element_T &obj, const uint32_t k_lower=0, const uint32_t k_upper=LabelLENGTH) const {
	    // No need to assert, because everything will be done inside the called function 'value.is_equal(...)'
		return label.is_equal(obj.label, k_lower, k_upper);
    }

	/// \return this->label > obj.label between the coordinates [k_lower, ..., k_upper]
	inline bool is_greater(const Element_T &obj, const uint32_t k_lower=0, const uint32_t k_upper=LabelLENGTH) const {
		// No need to assert, because everything will be done inside the called function 'value.is_greater(...)'
		return label.is_greater(obj.label, k_lower, k_upper);
	}

	/// \return this->label < obj.label between the coordinates [k_lower, ..., k_upper]
    inline bool is_lower(const Element_T &obj, const uint32_t k_lower=0, const uint32_t k_upper=LabelLENGTH) const {
	    // No need to assert, because everything will be done inside the called function 'value.is_lower(...)'
        return label.is_lower(obj.label, k_lower, k_upper);
    }

	/// \return true/false
	template<const uint32_t k_lower, const uint32_t k_upper>
	inline bool is_equal(const Element_T &obj) const {
		// No need to assert, because everything will be done inside the called function 'value.is_equal(...)'
		return label.template is_equal<k_lower, k_upper>(obj.label);
	}

	/// \return this->label > obj.label between the coordinates [k_lower, ..., k_upper]
	template<const uint32_t k_lower, const uint32_t k_upper>
	inline bool is_greater(const Element_T &obj) const {
		// No need to assert, because everything will be done inside the called function 'value.is_greater(...)'
		return label.template is_greater<k_lower, k_upper>(obj.label);
	}

	/// \return this->label < obj.label between the coordinates [k_lower, ..., k_upper]
	template<const uint32_t k_lower, const uint32_t k_upper>
	inline bool is_lower(const Element_T &obj) const {
		// No need to assert, because everything will be done inside the called function 'value.is_lower(...)'
		return label.template is_lower<k_lower, k_upper>(obj.label);
	}

	/// Assignment operator implementing copy assignment
    /// see https://en.cppreference.com/w/cpp/language/operators
    ///
    /// \param obj to copy from
    /// \return this
    Element_T& operator =(Element_T const &obj) {
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
	bool is_zero() {
		bool ret = false;

		ret |= value.is_zero();
		ret |= label.is_zero();
		return ret;
	}

	/// Assignment operator implementing move assignment
	/// see https://en.cppreference.com/w/cpp/language/move_assignment
	/// \param obj
	/// \return
	Element_T& operator =(Element_T &&obj) noexcept {
		if (this != &obj) { // self-assignment check expected really?
			value = std::move(obj.value);
			label = std::move(obj.label);
		}

		return *this;
	}

	/// print the internal data
	/// \param k_lower lower dimension
	/// \param k_upper upper dimension
	void print(const uint64_t k_lower, const uint64_t k_upper) const {
		label.print(k_lower, k_upper);
		value.print(k_lower, k_upper);
	}

	Value& get_value() { return value; }
	const Value& get_value() const { return value; }
	auto get_value(const size_t i) { ASSERT(i < value.size()); return value.data(i); }
	const auto get_value(const size_t i) const { ASSERT(i < value.size()); return value.data(i); }

	Label& get_label() { return label; }
	const Label& get_label() const { return label; }
	auto get_label(const uint64_t i) { ASSERT(i < label.size()); return label.data(i); }
	const auto get_label(const uint64_t i) const { ASSERT(i < label.size()); return label.data(i); }

	void set_value(const Value &v) { value = v; }
	void set_label(const Label &l) { label = l; }


	/// returns true of both underlying data structs are binary
	constexpr static bool binary() { return Label::binary() & Value::binary(); }
	constexpr static uint32_t label_size() { return Label::size(); }
	constexpr static uint32_t value_size() { return Value::size(); }
	constexpr static uint32_t size() { return Value::size()+Label::size(); }
	constexpr static uint32_t bytes() { return ValueContainerType::copyable_ssize()+LabelContainerType::copyable_ssize(); }

	__FORCEINLINE__ auto& get_label_container() { return label.data(); }
	__FORCEINLINE__ auto& get_value_container() { return value.data(); }

	__FORCEINLINE__ const auto& get_label_container() const { return label.data(); }
	__FORCEINLINE__ const auto& get_value_container() const { return value.data(); }

	__FORCEINLINE__ auto* get_label_container_ptr() { return label.data().data().data(); }
	__FORCEINLINE__ auto* get_value_container_ptr() { return value.data().data().data(); }

	__FORCEINLINE__ const auto* get_label_container_ptr() const { return label.data().data().data(); }
	__FORCEINLINE__ const auto* get_value_container_ptr() const { return value.data().data().data(); }
private:
    Value value;
    Label label;
};

/// print operator
/// \tparam Value type of a value
/// \tparam Label type of label
/// \param out
/// \param obj	input element
/// \return
template<class Value, class Label, class Matrix>
std::ostream& operator<< (std::ostream &out, const Element_T<Value, Label, Matrix> &obj) {
	out << "V: " << obj.get_value();
	out << ", L: " << obj.get_label() << "\n";
	return out;
}
#endif //SMALLSECRETLWE_ELEMENT_H
