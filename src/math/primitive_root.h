#pragma once

#include <vector>
#include <cstdint>

namespace cryptanalysislib {

    //  O(√m). Returns a generator of F^*_m.
	//  If m not prime, replace m − 1 by totient of m
    template<typename T>
    T primitive_root(T m) {
        std::vector<T> div;
        for (T i = 1; i*i < m; i++) {
            if ((m-1) % i == 0) {
                if (i < m-1) { div.pb(i); }
                if ((m-1)/i < m) { div.pb((m-1)/i); }
            }
        }

        for (uint32_t x = 2; x < m; x++) {
            bool ok = true;
            for (T d : div) { 
                if (mod_pow(x, d, m) == 1){ 
                    ok = false; 
                    break; 
                }
            }
            if (ok) { return x; }
        }
        return -1;
    }
}
