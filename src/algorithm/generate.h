#pragma once

/// NOTE: tests in `tests/algorithm/generate.cpp`
/// NOTE: parallel version of generator makes no sense.

namespace cryptanalysislib {

/// Assigns `g()` to every element of [first, last), in order
/// (like `std::generate`)
/// \tparam ForwardIt Forward iterator type
/// \tparam Generator callable without arguments, its result is assigned
/// \param first[in]: Iterator to the beginning of the range
/// \param last[in]: Iterator to the end of the range
/// \param g[in]: generator, called once per element
template<class ForwardIt, 
         class Generator>
constexpr 
void generate(ForwardIt first, 
              ForwardIt last, 
              Generator g) {
    for (; first != last; ++first) {
        *first = g();
    }
}

/// Assigns `g()` to the first `count` elements starting at `first`
/// (like `std::generate_n`)
/// \tparam OutputIt Output iterator type
/// \tparam Size integral type of `count`
/// \tparam Generator callable without arguments, its result is assigned
/// \param first[in]: Iterator to the first element to assign
/// \param count[in]: number of elements, nothing happens if `count <= 0`
/// \param g[in]: generator, called `count` times
/// \return Iterator one past the last assigned element
template<class OutputIt,
         class Size,
         class Generator>
constexpr
OutputIt generate_n(OutputIt first,
                    const Size count,
                    Generator g) {
    for (Size i = 0; i < count; ++i, ++first) {
        *first = g();
    }
 
    return first;
}
} // end namespace cryptanalysislib
