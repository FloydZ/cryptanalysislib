#ifndef CRYPTANALYSISLIB_REFLECTION_FLATTEN_H
#define CRYPTANALYSISLIB_REFLECTION_FLATTEN_H

#include <algorithm>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace cryptanalysislib::reflection {
	/// Used to embed another struct into the generated output.
	template <class T>
	struct Flatten {
		/// The underlying type.
		using Type = std::remove_cvref_t<T>;

		constexpr Flatten(const Type& _value) : value_(_value) {}
		constexpr Flatten(Type&& _value) noexcept : value_(std::forward<Type>(_value)) {}
		constexpr Flatten(const Flatten<T>& _f) = default;
		constexpr Flatten(Flatten<T>&& _f) noexcept = default;

		template <class U>
		constexpr Flatten(const Flatten<U>& _f) noexcept : value_(_f.get()) {}

		template <class U>
		constexpr Flatten(Flatten<U>&& _f) noexcept : value_(_f.get()) {}

		template <class U,
		         typename std::enable_if<std::is_convertible_v<U, Type>, bool>::type = true>
		constexpr Flatten(const U& _value) noexcept : value_(_value) {}

		template <class U,
		         typename std::enable_if<std::is_convertible_v<U, Type>, bool>::type = true>
		constexpr Flatten(U&& _value) noexcept : value_(_value) {}

		constexpr ~Flatten() = default;

		/// Returns the underlying object.
		constexpr const Type& get() const noexcept { return value_; }

		/// Returns the underlying object.
		constexpr Type& operator()() noexcept { return value_; }

		/// Returns the underlying object.
		constexpr const Type& operator()() const noexcept { return value_; }

		/// Assigns the underlying object.
		constexpr Flatten<T>& operator=(const T& _value) {
			value_ = _value;
			return *this;
		}

		/// Assigns the underlying object.
		constexpr Flatten<T>& operator=(T&& _value) {
			value_ = std::forward<Type>(_value);
			return *this;
		}

		/// Assigns the underlying object.
		template <class U,
		         typename std::enable_if<std::is_convertible_v<U, Type>, bool>::type = true>
		constexpr Flatten<T>& operator=(const U& _value) noexcept {
			value_ = _value;
			return *this;
		}

		/// Assigns the underlying object.
		constexpr Flatten<T>& operator=(const Flatten<T>& _f) = default;

		/// Assigns the underlying object.
		constexpr Flatten<T>& operator=(Flatten<T>&& _f) = default;

		/// Assigns the underlying object.
		template <class U>
		constexpr Flatten<T>& operator=(const Flatten<U>& _f) noexcept {
			value_ = _f.get();
			return *this;
		}

		/// Assigns the underlying object.
		template <class U>
		constexpr Flatten<T>& operator=(Flatten<U>&& _f) noexcept{
			value_ = std::forward<U>(_f);
			return *this;
		}

		/// Assigns the underlying object.
		constexpr void set(const Type& _value) noexcept { value_ = _value; }

		/// Assigns the underlying object.
		constexpr void set(Type&& _value) noexcept { value_ = std::forward<Type>(_value); }

		/// The underlying value.
		Type value_;
	};
}  // namespace rfl

#endif//CRYPTANALYSISLIB_FLATTEN_H
