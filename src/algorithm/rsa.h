#ifndef CRYPTANALYSISLIB_ALGORITHM_RSA_H
#define CRYPTANALYSISLIB_ALGORITHM_RSA_H

#include <cstdint>

#include "helper.h"
#include "algorithm/gcd.h"

using namespace cryptanalysislib;

struct RSA_instance {
	const uint64_t N;
};

template<const RSA_instance &config>
struct RSACmp {
	using T = TypeTemplate<config.N>;
	constexpr static T one{1};
	constexpr static T N{config.N};

	using TT = T::LimbType;
	decltype(auto) operator()(const T,
	                          const T &a2,
	                          const T,
	                          const T &b2) const {
		auto t = gcd<TT>((a2.value() + N - b2.value()) % N, N);
		return t > 1;
	}
};

template<const RSA_instance &config>
class rsa_pollard_rho {
	using T = TypeTemplate<config.N>;

	[[nodiscard]] bool run() noexcept {
		T a, b;
		a.set(5,0);
		b.set(26,0);

		PollardRho<RSACmp<config>, T>::run([](const T &in){
			const auto t = in*in + T(1);
			return t;
		}, a, b);

		std::cout << a << std::endl;
		std::cout << b << std::endl;

	}
};
#endif
