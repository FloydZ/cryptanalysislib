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

			for (size_t i = pos; i < len; ++i) {
				out[i] = in;
			}
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
