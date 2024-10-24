#pragma once

#include <cstddef>
#include <type_traits>

namespace loop_fusion::common {

/// A range is used to store the bounds of a loop.
/// \tparam Iterator The iterator type, e.g. size_t or int
template <typename Iterator>
class basic_range {
public:
    // Pre-condition: We only support integral iteration types like std::size_t or int.
    static_assert(std::is_integral_v<Iterator>, "range must be an integral type");

    using iterator = Iterator;

public:
    constexpr basic_range() noexcept = default;
    constexpr basic_range(Iterator _start,
        Iterator _end) noexcept
        : start { _start }
        , end { _end }
    {}

    Iterator start {};
    Iterator end {};

    /// Returns the span width between start and end as a size_t.
    /// \return size_t difference between start and end.
    [[nodiscard]] constexpr size_t span() const
    {
        return static_cast<size_t>(end) - static_cast<size_t>(start);
    }
};

/// Common range class
using range = basic_range<std::size_t>;

} // namespace loop_fusion::common
