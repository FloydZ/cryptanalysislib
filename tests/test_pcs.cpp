#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>
#include <array>

#include "helper.h"
#include "kAry_type.h"
#include "pcs.h"
#include "algorithm/gcd.h"
#include "algorithm/random_index.h"
#include "algorithm/int2weight.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// max n = 15
constexpr uint32_t q = 8051;
using T  = kAry_Type_T<q>;
using TT = T::LimbType;

struct RSACmp {
	constexpr static T one{1};
	constexpr static T nn{q};
	decltype(auto) operator()(T &a, T &b) const {
		auto t = gcd<TT>((a.value() + q - b.value()) % q, q);
		return t > 1;
	}
};

/// TODO move somewhere useful
/// generates a new random subset sum instance
template<typename T>
void generate_random_subset_sum(std::vector<T> &in,
                                T &target,
                                const uint32_t weight) noexcept {
	if (in.size() == 0) {
		return;
	}

	if (weight > in.size()) {
		return;
	}

   	for (size_t i = 0; i < in.size(); i++) {
		in[i] = T::random();
	}

	std::vector<uint32_t> weights(weight);
	generate_random_indices(weights, in.size());

	target = T(0);
	for (auto &w: weights) {
		target += in[w];
	}
}

TEST(PCS, RhoFactorise) {
	T a, b;
	a.set(5,0);
	b.set(26,0);
	PollardRho<RSACmp, T>::run([](const T &in){
		const auto t = in*in + T(1);
		return t;
	}, a, b);

	EXPECT_EQ(a.value(), 26);
	EXPECT_EQ(b.value(), 2839);
}

TEST(PCS, RhoSubSetSum) {
	/// Simple example of how to use the PollardRho class
	/// to solve a subset sum problem.
	/// As a function distinuisher we use the first bit a number
	constexpr static uint32_t n = 16;
	constexpr static uint32_t p = 2999;
	using T = kAry_Type_T<p>;

	T target, a = T(1734), b = T(319);
	std::vector<T> instance((size_t) n);
	generate_random_subset_sum(instance, target, n/2);

	struct SubSetSumCmp {
		decltype(auto) operator()(const T &a,
		                          const T &b) const {
			return a == b;
		}
	};

	while(true) {
		/// restart every X runs
		const bool val = PollardRho<SubSetSumCmp, T>::run([&instance, &target](const T &in) {
			static uint32_t weights[n/4];
			int2weights<T::LimbType>(weights, in.value(), n, n/4, n/4);
			T ret = T(0);

			// super simple distinguish function
			if (in.value() & 1u) {
				ret += target;
			}

			for (uint32_t i = 0; i < n / 4; i++) {
				ret += instance[weights[i]];
			}

			return ret;
		}, a, b, 1u << 6);

		if (val) {
			break;
		}

		// resample the flavour
		a = fastrandombytes_T<T::LimbType>() % p;
		b = fastrandombytes_T<T::LimbType>() % p;
	}

}



int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
