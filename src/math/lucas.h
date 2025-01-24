#pragma once

namespace cryptanalysislib {
    // calculate nCk % p (p prime!)
    template <typename T>
    T lucas(T n, T k, T p) {
        T ans = 1;
        while (n) {
            T np = n % p, kp = k % p;
            if (np < kp) return 0;
            ans = mod(ans * binom(np, kp), p); // (np C kp)
            n /= p; k /= p;
        }
        return ans;
    }
};
