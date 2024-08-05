#ifndef CRYPTANALYSISLIB_METRIC_H
#define CRYPTANALYSISLIB_METRIC_H

#include "popcount/popcount.h"
#include <cstdint>

namespace cryptanalysislib::metric {

#if __cplusplus > 201709L
/// describes the needed function to be row
template<class T>
concept RowAble = requires(T t) {
	typename T::LimbType;

	requires std::is_integral<typename T::LimbType>::value;

	requires requires(const uint32_t i,
					  typename T::LimbType l) {
		t[i];
		// number of underlying limbs
		t.limbs();
		T::modulus();
		// number of elements in each row
		T::length();
		T::add(t, t, t);
		T::add(t, t, t, i, i);
		t.rol(i);
		l.rol(i);
	};
};
#endif

class HammingMetric {
	/// :tparam T: limb type
	template<typename T>
#if __cplusplus > 201709L
		requires std::is_integral<T>::value
#endif
	constexpr static inline uint64_t wt(const T in) noexcept {
		return cryptanalysislib::popcount::popcount(in);
	}

	/// tparam T: row type
	template<typename T>
#if __cplusplus > 201709L
		requires RowAble<T>
#endif
	constexpr static inline uint32_t wt(const T &in) noexcept {
		uint64_t ret = 0;
		for (uint32_t i = 0; i < in.limbs(); i++) {
			ret += HammingMetric::wt(in[i]);
		}

		return ret;
	}

	/// tparam T: row type
	template<typename T>
#if __cplusplus > 201709L
		requires RowAble<T>
#endif
	constexpr static inline uint32_t d(const T &a, const T &b) noexcept {
		T c;
		T::add(c, a, b);
		return HammingMetric::wt(c);
	}

	static void info() noexcept {
		std::cout << " { name: \"HammingMetrix\""
			      << " }" << std::endl;
	}
};


}
#endif//CRYPTANALYSISLIB_METRIC_H
