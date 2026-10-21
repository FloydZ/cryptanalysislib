#ifndef CRYPTANALYSISLIB_ALGORITHM_RSA_H
#define CRYPTANALYSISLIB_ALGORITHM_RSA_H

#include <cstdint>

#include "helper.h"
#include "math/gcd.h"

using namespace cryptanalysislib;

/// \tparam L unsigned type of the modulus, the numbers mod N are
///		`kAry_Type_T_big<L, N>`
template<typename L = uint64_t>
struct RSA_instance_T {
	const L N;
};
using RSA_instance = RSA_instance_T<uint64_t>;

template<const auto &config>
struct RSACmp {
public:
	using L = std::remove_cv_t<decltype(config.N)>;
	using T = kAry_Type_T_big<L, config.N>;

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

template<const auto &config>
class rsa_pollard_rho {
public:
	using L = std::remove_cv_t<decltype(config.N)>;
	using T = kAry_Type_T_big<L, config.N>;

	[[nodiscard]] bool run() noexcept {
		T a, b;
		a.set(5,0);
		b.set(26,0);

		const bool found = PollardRho<RSACmp<config>, T>::run([](const T &in){
			const auto t = in*in + T(1);
			return t;
		}, a, b);

		std::cout << a << std::endl;
		std::cout << b << std::endl;
		return found;
	}
};
#endif
