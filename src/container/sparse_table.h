#pragma once

#include <vector>
template<typename T>
struct sparse_table {
    std::vector<std::vector<T>> m;

    /// \param arr[in]:
    sparse_table(std::vector<T> &arr) noexcept {
		m.pb(arr);
		for (int k=0; (1<<(++k)) <= sz(arr); ) {
			int w = (1<<k), hw = w/2;
			m.pb(vi(sz(arr) - w + 1));
			for (int i = 0; i+w <= sz(arr); i++) {
				m[k][i] = min(m[k-1][i], m[k-1][i+hw]);
			}
        }
	}

    // query min in [l,r]
	int query(int l, int r) { 
		int k = 31 - __builtin_clz(r-l);
		return min(m[k][l], m[k][r-(1<<k)+1]);
	}
};
