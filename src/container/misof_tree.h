#pragma once 

#include <cstdlib>
#include <cstring>

struct misof_tree {
    constexpr static size_t BITS = 15;
	int cnt[BITS][1ul<<BITS];


	misof_tree() { 
        memset(cnt,0,sizeof(cnt)); 
    }

    /// \param x[in]: 
	void insert(int x) {
		for (size_t i=0; i<BITS; cnt[i++][x]++, x >>= 1); 
    }

    /// \param x[in]: 
	void erase(int x) {
		for (size_t i=0; i<BITS; cnt[i++][x]--, x >>= 1); 
    }

    /// \param n[in]: 
	int nth(int n) {
		int res = 0;
		for (int i = BITS-1; i >= 0; i--) {
			if (cnt[i][res <<= 1] <= n) {
				n -= cnt[i][res], res |= 1;
            }
        }

		return res;
	}
};
