#pragma once
#include <cstddef>

namespace cryptanalysislib {
	namespace internal {
		template<typename T> 
 		constexpr void memset(T *out, 
				const T in, 
				const size_t len,
				const size_t pos) {
			if (pos == len) {
				return;
			}

			out[pos] = in;
			memcpy(out, in, len , pos+1);
		}
 	}

	template<typename T>
	constexpr void memset(T *out, T in, size_t len) {
		internal::memset(out, in, len, 0);
	}
}
