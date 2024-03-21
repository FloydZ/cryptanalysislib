#ifndef CRYPTANALYSISLIB_REFLECTION_LITERAL_H
#define CRYPTANALYSISLIB_REFLECTION_LITERAL_H

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <cstdint>
#include <type_traits>

#include "string/stringliteral.h"

using namespace cryptanalysislib::internal;
namespace cryptanalysislib::reflection {

	/// helper class
	/// \tparam _field
	template<internal::StringLiteral _field>
	struct LiteralHelper {
		constexpr static internal::StringLiteral field_ = _field;
	};

	///
	/// \tparam fields_
	template<internal::StringLiteral... fields_>
	class Literal {
		using FieldsType = std::tuple<LiteralHelper<fields_>...>;

	public:
		using ValueType = std::conditional_t<
		        sizeof...(fields_) <= std::numeric_limits<std::uint8_t>::max(),
		        std::uint8_t, std::uint16_t>;

		/// The number of different fields or different options that the literal
		/// can assume.
		static constexpr ValueType num_fields_ = sizeof...(fields_);

		using ReflectionType = std::string;

		/// Constructs a Literal from another literal.
		constexpr Literal(const Literal<fields_...> &_other) = default;

		/// Constructs a Literal from another literal.
		constexpr Literal(Literal<fields_...> &&_other) noexcept = default;

		constexpr Literal(const std::string &_str) : value_(find_value(_str).value()) {}

		/// A single-field literal is special because it
		/// can also have a default constructor.
		template<ValueType num_fields = num_fields_,
		         typename = std::enable_if_t<num_fields == 1>>
		constexpr Literal() noexcept : value_(0) {}

		constexpr ~Literal() = default;

		/// Constructs a new Literal.
		template<internal::StringLiteral _name>
		constexpr static Literal<fields_...> make() noexcept {
			return Literal(Literal<fields_...>::template value_of<_name>());
		}

		/// Constructs a new Literal, equivalent to make, for reasons of consistency.
		template<internal::StringLiteral _name>
		constexpr static Literal<fields_...> from_name() noexcept {
			return Literal<fields_...>::template make<_name>();
		}

		/// Constructs a new Literal.
		template<ValueType _value>
		constexpr static Literal<fields_...> from_value() noexcept {
			static_assert(_value < num_fields_,
			              "Value cannot exceed number of fields.");
			return Literal<fields_...>(_value);
		}

		/// Constructs a new Literal.
		constexpr static Literal<fields_...> from_value(ValueType _value) noexcept {
			if (_value >= num_fields_) {
				// return Error("Value cannot exceed number of fields.");
				std::cout << "Value cannot exceed number of fields." << std::endl;
				return Literal<fields_...>(num_fields_ - 1);
			}
			return Literal<fields_...>(_value);
		}

		/// Determines whether the literal contains the string.
		constexpr static bool contains(const std::string &_str) noexcept { return has_value(_str); }

		/// Determines whether the literal contains the string at compile time.
		template<internal::StringLiteral _name>
		static constexpr bool contains() noexcept {
			return find_value_of<_name>() != -1;
		}

		/// Determines whether the literal contains any of the strings in the other
		/// literal at compile time.
		template<class OtherLiteralType, int _i = 0>
		static constexpr bool contains_any() noexcept {
			if constexpr (_i == num_fields_) {
				return false;
			} else {
				constexpr auto name = find_name_within_own_fields<_i>();
				return OtherLiteralType::template contains<name>() ||
				       contains_any<OtherLiteralType, _i + 1>();
			}
		}

		/// Determines whether the literal contains all of the strings in the other
		/// literal at compile time.
		template<class OtherLiteralType, int _i = 0, int _n_found = 0>
		static constexpr bool contains_all() noexcept {
			if constexpr (_i == num_fields_) {
				return _n_found == OtherLiteralType::num_fields_;
			} else {
				constexpr auto name = find_name_within_own_fields<_i>();
				if constexpr (OtherLiteralType::template contains<name>()) {
					return contains_all<OtherLiteralType, _i + 1, _n_found + 1>();
				} else {
					return contains_all<OtherLiteralType, _i + 1, _n_found>();
				}
			}
		}

		/// Determines whether the literal has duplicate strings at compile time.
		/// These is useful for checking collections of strings in other contexts.
		static constexpr bool has_duplicates() noexcept { return has_duplicate_strings(); }

		/// Constructs a Literal from a string. Returns an error if the string
		/// cannot be found.
		constexpr static Literal from_string(const std::string &_str) noexcept {
			const auto to_literal = [](const auto &_v) {
				return Literal<fields_...>(_v);
			};
			return find_value(_str).transform(to_literal);
		};

		/// The name defined by the Literal.
		constexpr std::string name() const noexcept { return find_name(); }

		/// Returns all possible values of the literal as a std::vector<std::string>.
		constexpr static std::vector<std::string> names() noexcept{ return allowed_strings_vec(); }

		/// Helper function to retrieve a name at compile time.
		template<int _value>
		constexpr static auto name_of() noexcept {
			constexpr auto name = find_name_within_own_fields<_value>();
			return Literal<name>();
		}

		/// Assigns from another literal.
		constexpr Literal<fields_...> &operator=(const Literal<fields_...> &_other) = default;

		/// Assigns from another literal.
		constexpr Literal<fields_...> &operator=(Literal<fields_...> &&_other) noexcept =
		        default;

		/// Assigns the literal from a string
		constexpr Literal<fields_...> &operator=(const std::string &_str) noexcept {
			value_ = find_value(_str);
			return *this;
		}

		/// Equality operator other Literals.
		constexpr bool operator==(const Literal<fields_...> &_other) const noexcept {
			return value() == _other.value();
		}

		/// Alias for .name().
		constexpr std::string reflection() const noexcept { return name(); }

		/// Returns the number of fields in the Literal.
		static constexpr size_t size() noexcept { return num_fields_; }

		/// Alias for .name().
		constexpr std::string str() const noexcept { return name(); }

		/// Alias for .names().
		constexpr static std::vector<std::string> strings() noexcept { return allowed_strings_vec(); }

		/// Returns the value actually contained in the Literal.
		constexpr ValueType value() const noexcept { return value_; }

		/// Returns the value of the string literal in the template.
		template<internal::StringLiteral _name>
		static constexpr ValueType value_of() noexcept {
			constexpr auto value = find_value_of<_name>();
			static_assert(value >= 0, "String not supported.");
			return value;
		}

	private:
		/// Only the static methods are allowed to access this.
		constexpr Literal(const ValueType _value) noexcept : value_(_value) {}

		/// Returns all of the allowed fields.
		constexpr static std::string allowed_strings() noexcept {
			const auto vec = allowed_strings_vec();
			std::string str;
			for (size_t i = 0; i < vec.size(); ++i) {
				const auto head = "'" + vec[i] + "'";
				str += i == 0 ? head : (", " + head);
			}
			return str;
		}

		/// Returns all of the allowed fields.
		template<int _i = 0>
		constexpr static std::vector<std::string> allowed_strings_vec(
		        std::vector<std::string> _values = {}) noexcept {
			using FieldType = typename std::tuple_element<_i, FieldsType>::type;
			const auto head = FieldType::field_.str();
			_values.push_back(head);
			if constexpr (_i + 1 < num_fields_) {
				return allowed_strings_vec<_i + 1>(std::move(_values));
			} else {
				return _values;
			}
		}

		/// Whether the Literal contains duplicate strings.
		template<int _i = 1>
		constexpr static bool has_duplicate_strings() noexcept {
			if constexpr (_i >= num_fields_) {
				return false;
			} else {
				return is_duplicate<_i>() || has_duplicate_strings<_i + 1>();
			}
		}

		template<int _i, int _j = _i - 1>
		constexpr static bool is_duplicate() noexcept {
			using FieldType1 = typename std::tuple_element<_i, FieldsType>::type;
			using FieldType2 = typename std::tuple_element<_j, FieldsType>::type;
			if constexpr (FieldType1::field_ == FieldType2::field_) {
				return true;
			} else if constexpr (_j > 0) {
				return is_duplicate<_i, _j - 1>();
			} else {
				return false;
			}
		}

		/// Finds the correct index associated with
		/// the string at run time.
		template<int _i = 0>
		constexpr std::string find_name() const noexcept {
			if (_i == value_) {
				using FieldType = typename std::tuple_element<_i, FieldsType>::type;
				return FieldType::field_.str();
			}
			if constexpr (_i + 1 == num_fields_) {
				return "";
			} else {
				return find_name<_i + 1>();
			}
		}

		/// Finds the correct index associated with
		/// the string at compile time within the Literal's own fields.
		template<int _i>
		constexpr static auto find_name_within_own_fields() noexcept {
			using FieldType = typename std::tuple_element<_i, FieldsType>::type;
			return FieldType::field_;
		}

		/// Finds the correct value associated with
		/// the string at run time.
		template<int _i = 0>
		constexpr static int find_value(const std::string &_str) noexcept {
			using FieldType = typename std::tuple_element<_i, FieldsType>::type;
			if (FieldType::field_.str() == _str) {
				return _i;
			}
			if constexpr (_i + 1 == num_fields_) {
				std::cout << "Literal does not support string '" << _str << "'. The following strings are supported: " << allowed_strings() << ".";
				return -1;
			} else {
				return find_value<_i + 1>(_str);
			}
		}

		/// Finds the value of a string literal at compile time.
		template<internal::StringLiteral _name, int _i = 0>
		static constexpr int find_value_of() noexcept {
			using FieldType = typename std::tuple_element<_i, FieldsType>::type;
			if constexpr (FieldType::field_ == _name) {
				return _i;
			} else if constexpr (_i + 1 < num_fields_) {
				return find_value_of<_name, _i + 1>();
			} else {
				return -1;
			}
		}

		/// Whether the literal contains this string.
		template<int _i = 0>
		constexpr static bool has_value(const std::string &_str) noexcept {
			using FieldType = typename std::tuple_element<_i, FieldsType>::type;
			if (FieldType::field_.str() == _str) {
				return true;
			}
			if constexpr (_i + 1 == num_fields_) {
				return false;
			} else {
				return has_value<_i + 1>(_str);
			}
		}

		static_assert(sizeof...(fields_) > 0, "There must be at least one field in a Literal.");
		static_assert(sizeof...(fields_) <= std::numeric_limits<ValueType>::max(), "Too many fields.");
		static_assert(!has_duplicates(), "Duplicate strings are not allowed in a Literal.");

	private:
		/// The underlying value.
		ValueType value_;
	};
}

#endif//CRYPTANALYSISLIB_LITERAL_H
