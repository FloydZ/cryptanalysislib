#pragma once

#include <utility>

namespace cryptanalysislib {

    // Chinese Remainder Theorem: returns (u, v) s.t.
    // x=u (mod v) <=> x=a (mod n) and x=b (mod m)
    template<typename T>
    std::pair<T, T> crt(T a, T n, T b, T m) {
        T s, t, d = egcd(n, m, s, t); //n,m<=1e9
        if ((a - b) % d) {
            return { 0, -1 };
        }

        return { mod(s*b%m*n + t*a%n*m, n*m)/d, n*m/d };
    }
};
