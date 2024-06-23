#pragma once

#include <cstddef>
#include <type_traits>

namespace loop_fusion::common {

template <typename T>
struct type_identity {
    constexpr type_identity() noexcept = default;
    using type = T;
};

} // namespace loop_fusion::common
