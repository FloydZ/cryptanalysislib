#pragma once

#include "loop_fusion/common/range.hpp"

#include <tuple>
#include <variant>

namespace loop_fusion::runtime {

using range = loop_fusion::common::range;

// Forward declaration
template <typename V>
class looper_union;

template <typename... F>
class looper {
public:
    constexpr explicit looper(range _range, F... _functions)
        : rng { _range }
        , functions(std::make_tuple(_functions...)) {};

    constexpr looper(range _range, std::tuple<F...> _functions)
        : rng { _range }
        , functions(_functions) {};

    constexpr void run()
    {
        run_loops(std::index_sequence_for<F...> {});
    }

    using self = looper<F...>;
    static constexpr auto function_count = sizeof...(F);

    /// Type of functions that can be used for loop fusion.
    /// Function must accept an argument of type Iter per value.
    template <typename loop_function>
    using is_loop_function =std::is_invocable<loop_function&, size_t>;

    static_assert(std::conjunction_v<is_loop_function<F>...>, "Invalid function signature");

    range rng;
    std::tuple<F...> functions;

    template<typename loop_function,
		     typename = std::enable_if_t<is_loop_function<loop_function>::value>>
    [[nodiscard]] constexpr auto operator|(loop_function rhs) const noexcept {
        using new_looper = looper<F..., loop_function>;
        return new_looper(rng, std::tuple_cat(functions, std::make_tuple(rhs)));
    }

    template <typename... F2>
    [[nodiscard]] constexpr auto operator|(const looper<F2...>& rhs) const noexcept {
        return looper_union<std::variant<looper<F...>>>(*this) | rhs;
    }

private:
    template <std::size_t... Idx>
    constexpr void run_loops(std::index_sequence<Idx...>) noexcept {
        for (auto i = rng.start; i < rng.end; ++i) {
            (std::get<Idx>(functions)(i), ...);
        }
    }
};

} // namespace loop_fusion::runtime
