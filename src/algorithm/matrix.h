#pragma once
#include <cstddef>

// this file is not in `matrix.h` as this file implements algorithms which
// solve matrix equation systems.

namespace cryptanalysislib {
/// 1.5. Tridiagonal Matrix Algorithm. Solves a tridiagonal
/// system of linear equations
/// 	a_i x_{i-1} + b_i x_i + c_i x_{i+1} = d_i, for i = 0, ..., n-1,
/// where a_0 = c_{n-1} = 0. Beware of numerical instability.
/// NOTE: `c` and `d` are overwritten.
/// \param x output: the solution, n elements
/// \param a lower diagonal, n elements
/// \param b diagonal, n elements
/// \param c upper diagonal, n elements, will be overwritten
/// \param d right hand side, n elements, will be overwritten
/// \param n number of equations, >= 2
template <typename T>
constexpr void solve(T *x,
                     const T *a,
                     const T *b,
                     T *c,
                     T *d,
                     const size_t n) noexcept {
    c[0] /= b[0]; d[0] /= b[0];
    for (size_t i = 1; i < n-1; i++) { c[i] /= b[i] - a[i]*c[i-1]; }
    for (size_t i = 1; i < n; i++) {  d[i] = (d[i] - a[i]*d[i-1]) / (b[i] - a[i]*c[i-1]); }
    x[n-1] = d[n-1];
    for (size_t i = n-1; i--;) { x[i] = d[i] - c[i]*x[i+1]; }
}

};
