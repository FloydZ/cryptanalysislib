#pragma once

#include "loop_fusion/compiletime/basic_looper.hpp"
#include "loop_fusion/compiletime/basic_looper_merge.hpp"
#include "loop_fusion/compiletime/basic_looper_union.hpp"
#include "loop_fusion/compiletime/types.hpp"

#include <tuple>
#include <type_traits>

namespace loop_fusion::compiletime {

	/// Main looper class. Is a specialization of basic_looper of type std::size_t
	template <std::size_t Start, std::size_t End, typename... F>
	using looper = basic_looper<std::size_t, Start, End, F...>;

	/// Create a looper with a range of [start, end)
	template <std::size_t Start, std::size_t End, typename... F>
	[[nodiscard]] constexpr auto loop(F... args) noexcept {
	    return looper<Start, End, F...>(std::make_tuple(args...));
	}

	/// Create a looper with a range of [0, end)
	template <std::size_t End, typename... F>
	[[nodiscard]] constexpr auto loop_to(F... args) noexcept {
	    return looper<0ull, End, F...>(std::make_tuple(args...));
	}

	/// Alias for `loop<Start, End>(...)`.
	template <std::size_t Start, std::size_t End, typename... F>
	[[nodiscard]] constexpr auto loop_from_to(F... args) noexcept {
	    return looper<Start, End, F...>(std::make_tuple(args...));
	}

} // namespace loop_fusion::compiletime
