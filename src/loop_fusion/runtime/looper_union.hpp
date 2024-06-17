#pragma once

#include "loop_fusion/runtime/looper.hpp"
#include "loop_fusion/runtime/types.hpp"

#include <algorithm>
#include <tuple>
#include <variant>
#include <vector>

namespace loop_fusion::runtime {

namespace detail {
    template <typename V1, typename V2, typename Looper>
    [[nodiscard]] constexpr std::vector<V1> add_looper(const std::vector<V2>& loopers, const Looper& rhs)
    {
        std::vector<V1> new_loopers;

        auto r2 = rhs.rng;
        bool included = false;
        for (auto&& l : loopers) {
            std::visit(
                [&](auto&& lhs) {
                    auto r1 = lhs.rng;

                    if (r1.start == r2.start && r1.end == r2.end) {
                        // Equal ranges
                        new_loopers.push_back(looper(r1, std::tuple_cat(lhs.functions, rhs.functions)));
                        included = true;
                    } else if (r1.end <= r2.start || r1.start >= r2.end) {
                        // Disjunct ranges
                        new_loopers.push_back(lhs);
                    } else {
                        // Intersecting ranges

                        // Before intersection
                        if (r1.start < r2.start) {
                            new_loopers.push_back(looper(range { r1.start, r2.start }, lhs.functions));
                        } else if (r2.start < r1.start) {
                            new_loopers.push_back(looper(range { r2.start, r1.start }, rhs.functions));
                        }

                        // Intersection
                        auto p1 = std::max(r1.start, r2.start);
                        auto p2 = std::min(r1.end, r2.end);
                        if (p1 != p2) {
                            new_loopers.push_back(
                                looper(range { p1, p2 }, std::tuple_cat(lhs.functions, rhs.functions)));
                        }

                        // After intersection
                        if (r1.end < r2.end) {
                            r2.start = r1.end;
                        } else if (r2.end < r1.end) {
                            new_loopers.push_back(looper(range { r2.end, r1.end }, lhs.functions));
                        }
                        included = r2.end <= r1.end;
                    }
                },
                l);
        }

        if (!included) {
            new_loopers.push_back(looper(range { r2.start, r2.end }, rhs.functions));
        }

        return new_loopers;
    }
} // namespace detail

template <typename V>
class looper_union {
public:
    template <typename Looper>
    constexpr explicit looper_union(const Looper& looper)
        : loopers { std::variant<Looper>(looper) }
    {}

    constexpr explicit looper_union(const std::vector<V>& _loopers)
        : loopers(_loopers)
    {}

    template <typename V2>
    constexpr explicit looper_union(const looper_union<V2>& u)
        : loopers {}
    {
        // Unfortunately, a vector of variants can't just be assigned to another vector that contains supersets of its
        // variants. As a workaround, each variant in the source vector has to be visited to copy it to the destination
        // vector.
        for (auto&& l : u.loopers) {
            std::visit([this](auto&& looper) { loopers.push_back(looper); }, l);
        }
    }

    constexpr void run() noexcept {
        for (auto&& l : loopers) {
            std::visit([](auto&& looper) { looper.run(); }, l);
        }
    }

    template <typename... F>
    [[nodiscard]] constexpr auto operator|(const looper<F...>& rhs) const noexcept {
        using new_variant = types::variant_append_t<V, F...>;
        using new_union = looper_union<new_variant>;
        return new_union(detail::add_looper<new_variant>(loopers, rhs));
    }

    /**
     * An almost complete implementation for merging a union with another union.
     * Unfortunately, it is missing a final part:
     * As `u.loopers` uses `new_variant` for its elements, the branches of
     * `detail::add_looper` that merge two function tuples will fail for certain
     * visitted variants. For example, there are branches that potentially add
     * functions in an order that actually isn't even possible, but the compiler
     * can't rule them out and therefore fails to compile.
     * Hopefully one day, in the near or distant future, a brave soul is able to solve
     * this issue and bring glorious union with union merging to this implementation.
     */
    template <typename... L>
    [[nodiscard]] constexpr auto operator|(const looper_union<std::variant<L...>>& rhs) const noexcept {
        using new_variant = types::variant_append_all_t<V, L...>;
        using new_union = looper_union<new_variant>;
        new_union u(*this);
        for (auto&& l : rhs.loopers) {
            std::visit(
                [&u](auto&& looper) {
                    u.loopers = detail::add_looper<new_variant>(u.loopers, looper);
                },
                l);
        }
        return u;
    }

    /**
     * public so it can be accessed from operator| for merging unions with unions
     */
    std::vector<V> loopers;
};

} // namespace loop_fusion::runtime
