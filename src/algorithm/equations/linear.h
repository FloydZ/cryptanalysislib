#pragma once

// Number of Integer Points under Line. Count the
// number of integer solutions to Ax + By ≤ C, 0 ≤ x ≤ n,
// 0 ≤ y. In other words, evaluate the sum ∑n x=0 ⌊ C−Ax B + 1⌋. To
// count all solutions, let n = ⌊ c/a ⌋. In any case, it must hold that
// C − nA ≥ 0. Be very careful about overflows.
template <typename T>
T floor_sum(T n, T a, T b, T c) {
    if (c == 0) { return 1; }
    if (c < 0) { return 0; }
    if (a % b == 0) { return (n+1)*(c/b+1)-n*(n+1)/2*a/b; }
    if (a >= b) { return floor_sum(n,a%b,b,c)-a/b*n*(n+1)/2; }
    T t = (c-a*n+b)/b;
    return floor_sum((c-b*t)/b,b,a,c-b*t)+t*(n+1); 
}
