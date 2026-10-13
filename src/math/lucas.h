#pragma once

#include "math/math.h"

namespace cryptanalysislib {
    // calculate nCk % p (p prime!)
    template <typename T>
    T lucas(T n, T k, T p) {
        T ans = 1;
        while (n) {
            T np = n % p, kp = k % p;
            if (np < kp) return 0;
            // NOTE: was `mod(...)`, which does not exist (did not compile);
            //  np, kp < p, so `binom(np, kp)` is computed exactly for small p
            ans = mulmod<T>(ans, T(binom(np, kp) % p), p); // (np C kp)
            n /= p; k /= p;
        }
        return ans;
    }
};
