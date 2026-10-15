#pragma once

#include <vector>
#include "alloc/alloc.h"

template<typename T,
        template<class N> class Allocator = cryptanalysislib::allocator>
struct sparse_table {
    std::vector<std::vector<T>> m;

    /// \param arr[in]:
    sparse_table(const std::vector<T> &arr) noexcept {
		m.push_back(arr);
		for (size_t k=0; (size_t(1)<<(++k)) <= arr.size(); ) {
			const size_t w = (size_t(1)<<k), hw = w/2;
			m.push_back(std::vector<T>(arr.size() - w + 1));
			for (size_t i = 0; i+w <= arr.size(); i++) {
				const T a = m[k-1][i], b = m[k-1][i+hw];
				m[k][i] = a < b ? a : b;
			}
        }
	}

    // query min in [l,r] (both inclusive, l <= r)
	T query(const size_t l, const size_t r) const noexcept { 
		// largest k with 2^k <= r-l+1
		const size_t k = 63 - __builtin_clzll(r - l + 1);
		const T a = m[k][l], b = m[k][r-(size_t(1)<<k)+1];
		return a < b ? a : b;
	}
};
