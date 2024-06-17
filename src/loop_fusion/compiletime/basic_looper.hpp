#pragma once

#include "loop_fusion/compiletime/basic_looper_merge.hpp"

#include <tuple>
#include <type_traits>

namespace loop_fusion::compiletime {

template <typename Iter, Iter Start, Iter End, typename... F>
class basic_looper {
public:
    // Pre-condition: We only support integral iteration types like std::size_t or int.
    static_assert(std::is_integral_v<Iter>, "loop value type must be an integral value type");

    using self = basic_looper<Iter, Start, End, F...>;
    using iterator = Iter;
    static constexpr iterator start = Start;
    static constexpr iterator end = End;
    static constexpr auto function_count = sizeof...(F);

    /// Type of functions that can be used for loop fusion.
    /// Function must accept an argument of type Iter per value.
    template <typename loop_function>
    using is_loop_function = std::is_invocable<loop_function&, iterator>;
    template <typename loop_function>
    static constexpr bool is_loop_function_v = is_loop_function<loop_function>::value;

    static_assert(std::conjunction_v<is_loop_function<F>...>, "Invalid function signature");

    std::tuple<F...> functions;

public:
    constexpr explicit basic_looper(std::tuple<F...> _functions)
        : functions(_functions) {};

    constexpr void run() noexcept {
        run_loops(std::index_sequence_for<F...> {});
    }

    template <iterator RhsStart, iterator RhsEnd, typename... RhsF>
    constexpr auto operator|(basic_looper<iterator, RhsStart, RhsEnd, RhsF...> rhs) const {
        if constexpr (Start == RhsStart && End == RhsEnd) {
            using new_looper = basic_looper<iterator, Start, End, F..., RhsF...>;
            return new_looper(std::tuple_cat(functions, rhs.functions));
        } else {
            // merging is a bit more complicated and returns a
            // type basic_looper_union
            return make_union(*this, rhs);
        }
    }

    template <typename loop_function, typename = std::enable_if_t<is_loop_function_v<loop_function>>>
    constexpr auto operator|(loop_function rhs) const {
        using new_looper = basic_looper<iterator, Start, End, F..., loop_function>;
        return new_looper(std::tuple_cat(functions, std::make_tuple(rhs)));
    }

private:
    template <std::size_t... Idx>
    constexpr void run_loops(std::index_sequence<Idx...>) noexcept {
        for (auto i = Start; i != End; ++i) {
            (std::get<Idx>(functions)(i), ...);
        }
    }
};

} // namespace loop_fusion::compiletime
