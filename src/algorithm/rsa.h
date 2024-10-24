#ifndef CRYPTANALYSISLIB_ALGORITHM_RSA_H
#define CRYPTANALYSISLIB_ALGORITHM_RSA_H

#include <cstdint>

#include "helper.h"
#include "algorithm/gcd.h"

using namespace cryptanalysislib;

struct RSA_instance {
	/// TODO generic Fq numbers
	const uint64_t N;
};

template<const RSA_instance &config>
struct RSACmp {
public:
	// using L = TypeTemplate<config.N>;
	using T = kAry_Type_T<config.N>;
	using L = T::LimbType;

	constexpr static T one{1};
	constexpr static T N{config.N};

	decltype(auto) operator()(const T,
	                          const T &a2,
	                          const T,
	                          const T &b2) const {
		auto t = gcd<L>((a2.value() + config.N - b2.value()) % config.N, config.N);
		return t > 1;
	}
};

template<const RSA_instance &config>
class rsa_pollard_rho {
public:
	using L = TypeTemplate<config.N>;
	using T = kAry_Type_T<config.N>;

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
