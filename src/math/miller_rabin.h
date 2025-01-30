#pragma once 

namespace cryptanalysislib {
    // src: https://github.com/ludopulles/tcr
    // returns if n is prime for n < 3e24 (>2^64)
    // but use mul_mod for n > 2e9.
    template<typename T>
    bool millerRabin(T n){
		constexpr T data[] = { 2, 3, 5, 7, 11, 13,17, 19, 23, 29, 31, 37, 41 };
        if (n < 2 || n % 2 == 0) { 
            return n == 2;
        }
        T d = n - 1, ad, s = 0, r;
        for (; d % 2 == 0; d /= 2) { s++; }
        for (int a : data) {
            if (n == a) { return true; }
            if ((ad = mod_pow(a, d, n)) == 1) { continue; }
            
            for (r = 0; r < s && ad + 1 != n; r++) {
                ad = (ad * ad) % n;
            }
            if (r == s) {
				return false;
			}
        }
        return true;
    }
};
