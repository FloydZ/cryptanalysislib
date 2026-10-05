#pragma once

#include <cstddef>
#include <utility>

// Sort f[] (ascending order).
// Algorithm is O(n*n), use for short arrays only.
template <typename Type>
constexpr void selection_sort(const Type *f,
                              const size_t n) noexcept {
    for (size_t i=0; i<n; ++i) {
        Type v = f[i];
        // position of minimum
        size_t m = i;
        size_t j = n;
        // search (index of) minimum
        while (--j > i) {
            if (f[j] < v){
                m = j;
                v = f[m];
            }
        }

        std::swap(f[i], f[m]);
    }
}
