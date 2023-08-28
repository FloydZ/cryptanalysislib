#pragma once
#include <cstddef>

namespace cryptanalysislib {
	namespace internal {

		///
		/// \tparam T
		/// \param out
		/// \param in
		/// \param len
		/// \param pos
		/// \return
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

	///
	/// \tparam T
	/// \param out
	/// \param in
	/// \param len
	/// \return
	template<typename T>
	constexpr void memset(T *out, T in, size_t len) {
		internal::memset(out, in, len, 0);
	}
}
