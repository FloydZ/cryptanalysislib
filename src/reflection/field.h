#ifndef CRYPTANALYSISLIB_REFLECTION_FIELD_H
#define CRYPTANALYSISLIB_REFLECTION_FIELD_H

#include "string/stringliteral.h"
#include "reflection/internal.h"
#include "reflection/literal.h"

using namespace cryptanalysislib::internal;

namespace cryptanalysislib::reflection {

/// Used to define a field in the NamedTuple.
template <StringLiteral _name, class T>
struct Field {
	/// The underlying type.
	using Type = wrap_in_rfl_array_t<T>;

	/// The name of the field.
	using Name = Literal<_name>;

	constexpr Field(const Type& _value) noexcept : value_(_value) {}
	constexpr Field(Type&& _value) noexcept : value_(std::move(_value)) {}
	constexpr Field(Field<_name, T>&& _field) noexcept = default;
	constexpr Field(const Field<_name, T>& _field) = default;

	template <class U>
	constexpr Field(const Field<_name, U>& _field) noexcept : value_(_field.get()) {}

	template <class U>
	constexpr Field(Field<_name, U>&& _field) noexcept : value_(_field.get()) {}

	template <class U,
		     typename std::enable_if<std::is_convertible_v<U, Type>, bool>::type = true>
	constexpr Field(const U& _value) noexcept : value_(_value) {}

	template <class U,
		     typename std::enable_if<std::is_convertible_v<U, Type>, bool>::type = true>
	constexpr Field(U&& _value) noexcept : value_(std::forward<U>(_value)) {}

	template <class U,
		     typename std::enable_if<std::is_convertible_v<U, Type>, bool>::type = true>
	constexpr Field(const Field<_name, U>& _field) noexcept : value_(_field.value()) {}

	/// Assigns the underlying object to its default value.
	template <class U = Type,
	         typename std::enable_if<std::is_default_constructible_v<U>, bool>::type = true>
	constexpr Field(const Default& _default) noexcept : value_(Type()) { (void)_default; }

	~Field() = default;

	/// The name of the field, for internal use.
	constexpr static const StringLiteral name_ = _name;

	/// Returns the underlying object.
	constexpr const Type& get() const noexcept { return value_; }

	/// The name of the field.
	constexpr static std::string_view name() noexcept { return name_.string_view(); }

	/// Returns the underlying object.
	constexpr Type& operator()() noexcept { return value_; }

	/// Returns the underlying object.
	constexpr const Type& operator()() const noexcept { return value_; }

	/// Assigns the underlying object.
	constexpr auto& operator=(const Type& _value) noexcept {
		value_ = _value;
		return *this;
	}

	/// Assigns the underlying object.
	constexpr auto& operator=(Type&& _value) noexcept {
		value_ = std::move(_value);
		return *this;
	}

	/// Assigns the underlying object.
	template <class U,
		     typename std::enable_if<std::is_convertible_v<U, Type>, bool>::type = true>
	constexpr auto& operator=(const U& _value) noexcept {
		value_ = _value;
		return *this;
	}

	/// Assigns the underlying object to its default value.
	template <class U = Type,
	         typename std::enable_if<std::is_default_constructible_v<U>, bool>::type = true>
	constexpr auto& operator=(const Default& _default) noexcept {
		(void)_default;
		value_ = Type();
		return *this;
	}

	/// Assigns the underlying object.
	constexpr Field<_name, T>& operator=(const Field<_name, T>& _field) = default;

	/// Assigns the underlying object.
	constexpr Field<_name, T>& operator=(Field<_name, T>&& _field) = default;

	/// Assigns the underlying object.
	template <class U>
	constexpr auto& operator=(const Field<_name, U>& _field) noexcept {
		value_ = _field.get();
		return *this;
	}

	/// Assigns the underlying object.
	template <class U>
	constexpr auto& operator=(Field<_name, U>&& _field) noexcept {
		value_ = std::forward<T>(_field.value_);
		return *this;
	}

	/// Assigns the underlying object.
	constexpr void set(const Type& _value) noexcept { value_ = _value; }

	/// Assigns the underlying object.
	constexpr void set(Type&& _value) noexcept { value_ = std::move(_value); }

	/// Returns the underlying object.
	constexpr Type& value() noexcept { return value_; }

	/// Returns the underlying object.
	constexpr const Type& value() const noexcept { return value_; }

	/// The underlying value.
	Type value_;
};

/// Contains meta-information about a field in a struct.
class MetaField {
public:
	constexpr MetaField(const std::string& _name,
		                const std::string& _type) noexcept
		: name_(_name), type_(_type) {}

	constexpr ~MetaField() = default;

	/// The name of the field we describe.
	constexpr const std::string& name() const noexcept { return name_; };

	/// The type of the field we describe.
	constexpr const std::string& type() const noexcept { return type_; };

private:
	/// The name of the field we describe.
	std::string name_;

	/// The type of the field we describe.
	std::string type_;
};


}
#endif//CRYPTANALYSISLIB_REFLECTION_FIELD_H
