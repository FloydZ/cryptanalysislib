#pragma once
#include <stdlib.h>

// this file is not in `matrix.h` as this file implements algorithms which 
// solve matrix equation systems.

namespace cryptanalysislib {
// 1.5. Tridiagonal Matrix Algorithm. Solves a tridiagonal
// system of linear equations
// aixi−1 + bixi + cixi+1 = di
// where a1 = cn = 0. Beware of numerical instability.
#define MAXN 5000

double A[MAXN], B[MAXN], C[MAXN], D[MAXN], X[MAXN];

template <typename T>
void solve(int n) {
    C[0] /= B[0]; D[0] /= B[0];
    for (size_t i = 1; i < n-1; i++) { C[i] /= B[i] - A[i]*C[i-1]; }
    for (size_t i = 1; i < n; i++) {  D[i] = (D[i] - A[i]*D[i-1]) / (B[i] - A[i]*C[i-1]); }
    X[n-1] = D[n-1];
    for (int i = n-1; i--;) { X[i] = D[i] - C[i]*X[i+1]; }
}

};
