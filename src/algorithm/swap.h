#pragma once

/// TODO doc
template<class ForwardIt1,
         class ForwardIt2>
constexpr
void iter_swap(ForwardIt1 a,
               ForwardIt2 b) {
    using std::swap;
    swap(*a, *b);
}

/// TODO all other functions
