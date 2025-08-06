#pragma once

/// TODO tests
/// NOTE: parallel version of generator makes no sense.

// TODO doc
/// \tparam ForwardIt
/// \tparam Generator
/// \param[in]: first 
/// \param[in]: last 
/// \param[in]: G
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

/// \tparam ForwardIt
/// \tparam Size
/// \tparam Generator
/// \param[in]: first 
/// \param[in]: count 
/// \param[in]: G
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
