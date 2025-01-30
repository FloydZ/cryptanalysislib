#pragma once

#include <vector>

namespace cryptanalysislib {
    // phi[i] = #{ 0 < j <= i | gcd(i, j) = 1 } sieve
    template<typename T>
    std::vector<T> totient(int N) noexcept {
    std::vector<T> phi(N);
        for (int i = 0; i < N; i++) phi[i] = i;
        for (int i = 2; i < N; i++) if (phi[i] == i)
        for (int j = i; j < N; j+=i) phi[j] -= phi[j]/i;
        return phi;
    }
};
