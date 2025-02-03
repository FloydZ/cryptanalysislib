#pragma once 

#include <cstdint>

template<typename T>
class revbin_upd {
private:
    constexpr static inline uint32_t BITS = 64;
    // mask for updating bit-reversed values
    T utab[BITS];
    
    // Initialize lookup table used by revbin_tupd()
    constexpr inline void make_revbin_upd_tab(T h) {
        utab[0] = h;  // == n/2 == 1UL<<(ldn-1);
        for (T i=1; h!=0; ++i)
        {
            h >>= 1;
            utab[i] = utab[i-1] ^ h;
        }
    }
public:
    explicit revbin_upd(T h) noexcept {
        make_revbin_upd_tab(h);
    }

    // Let n=2**ldn and h=n/2.
    // Then, with r == revbin(x, ldn) at entry, return revbin(x+1, ldn)
    // NOTE: routine will hang if called with r the all-ones word
    constexpr inline T revbin_upd_(T r, T h) {
        while ( ! ((r^=h) & h) )  h >>= 1;
        return  r;
    }
    

    // Let r==revbin(k, ldn) then
    // return revbin(k+1, ldn).
    // NOTE 1: need to call make_revbin_upd_tab(ldn) before usage
    //         where ldn=log_2(n)
    // NOTE 2: different argument structure than revbin_upd()
    constexpr inline T revbin_tupd(T r, T k) {
        k = lowest_one_idx( ~k );  // lowest zero idx
        r ^= utab[k];
        return r;
    }


};
