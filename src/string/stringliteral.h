#ifndef CRYPTANALYSISLIB_STRING_STRINGLITERAL_H
#define CRYPTANALYSISLIB_STRING_STRINGLITERAL_H

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

namespace cryptanalysislib::internal {
	/// Normal strings cannot be used as template
	/// parameters, but this can. This is needed
	/// for the parameters names in the NamedTuples.
	template <size_t N>
	struct StringLiteral {
		constexpr StringLiteral(const auto... _chars) noexcept : arr_{_chars..., '\0'} {}

		constexpr StringLiteral(const char (&_str)[N]) noexcept {
			std::copy_n(_str, N, std::data(arr_));
		}

		/// Returns the value as a string.
		constexpr inline std::string str() const noexcept { return std::string(std::data(arr_), N - 1); }

		/// Returns the value as a string.
		constexpr std::string_view string_view() const noexcept {
			return std::string_view(std::data(arr_), N - 1);
		}

		///
		std::array<char, N> arr_{};
	};

	/// spaceship operator
	/// \tparam N1
	/// \tparam N2
	/// \param _first
	/// \param _second
	/// \return
	template <size_t N1, size_t N2>
	constexpr inline auto operator<=>(const StringLiteral<N1>& _first,
	                                  const StringLiteral<N2>& _second) {
		if constexpr (N1 != N2) {
			return N1 <=> N2;
		}
		return _first.string_view() <=> _second.string_view();
	}

	template <size_t N1, size_t N2>
	constexpr inline bool operator==(const StringLiteral<N1>& _first,
	                                 const StringLiteral<N2>& _second) {
		if constexpr (N1 != N2) {
			return false;
		}
		return _first.string_view() == _second.string_view();
	}

	template <size_t N1, size_t N2>
	constexpr inline bool operator!=(const StringLiteral<N1>& _first,
	                                 const StringLiteral<N2>& _second) {
		return !(_first == _second);
	}
}  // namespace

#endif//CRYPTANALYSISLIB_STRINGLITERAL_H
