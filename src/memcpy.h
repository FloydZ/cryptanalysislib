#pragma once
#include <cstddef>

namespace cryptanalysislib {
	namespace internal {
		template<typename T> 
 		constexpr void memcpy(T *out, 
				const T *in, 
				const size_t len,
				const size_t pos) {
			if (pos == len) {
				return;
			}

			out[pos] = in[pos];
			memcpy(out, in, len , pos+1);
		}
 	}

	template<typename T>
	constexpr void memcpy(T *out, T *in, size_t len) {
		internal::memcpy(out, in, len, 0);
	}
}
