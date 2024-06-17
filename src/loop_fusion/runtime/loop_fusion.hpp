#pragma once

#include "loop_fusion/runtime/looper.hpp"
#include "loop_fusion/runtime/looper_union.hpp"
#include "loop_fusion/runtime/types.hpp"

namespace loop_fusion::runtime {

/// Create a looper with a range of [start, end)
template <typename... F>
[[nodiscard]] constexpr auto loop(range _range, F... args)
{
    return looper<F...>(_range, std::make_tuple(args...));
}

/// Create a looper with a range of [0, end)
template <typename... F>
[[nodiscard]] constexpr auto loop_to(size_t to, F... args)
{
    return looper<F...>(range { 0ll, to }, std::make_tuple(args...));
}

/// Create a looper with a range of [start, end)
template <typename... F>
[[nodiscard]] constexpr auto loop_from_to(size_t from, size_t to, F... args)
{
    return looper<F...>(range { from, to }, std::make_tuple(args...));
}

} // namespace loop_fusion::runtime
