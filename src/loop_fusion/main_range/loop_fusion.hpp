#pragma once

#include "loop_fusion/common/range.hpp"
#include "loop_fusion/main_range/looper.hpp"
#include "loop_fusion/main_range/looper_union.hpp"

namespace loop_fusion::main_range {

/// Create a looper with a range of [start, end). Iterator type
/// depends on basic_range::iterator.
template <typename Iterator, typename... F>
[[nodiscard]] constexpr auto loop(::loop_fusion::common::basic_range<Iterator> _range, F... args)
{
    return looper(_range, std::make_tuple(args...));
}

/// Create a looper with a range of [0, end) with iterator type size_t
template <typename... F>
[[nodiscard]] constexpr auto loop_to(std::size_t to, F... args)
{
    return looper(range { 0ll, to }, std::make_tuple(args...));
}

/// Create a looper with a range of [start, end) with iterator type size_t
template <typename... F>
[[nodiscard]] constexpr auto loop_from_to(std::size_t from, std::size_t to, F... args)
{
    return looper(range { from, to }, std::make_tuple(args...));
}

} // namespace loop_fusion::main_range
