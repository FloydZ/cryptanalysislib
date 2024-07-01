#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>
#include <array>

#include "helper.h"
#include "kAry_type.h"
#include "pcs.h"
#include "algorithm/gcd.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// max n = 15
constexpr uint32_t q    = 8051;
using T    		= kAry_Type_T<q>;
using TT 		= T::LimbType;

struct RSACmp {
	constexpr static T one{1};
	constexpr static T nn{q};
	decltype(auto) operator()(T &a, T &b) const {
		auto t = gcd<TT>((a.value() + q - b.value()) % q, q);
		return t > 1;
	}
};

template<typename T>
void generate_random_subset_sum(std::vector<T> &in, T &target, const uint32_t weight) {
	if (in.size() == 0) {
		return;
	}

   	target =  T::random();
	T sum = T(0);
   	for (size_t i = 0; i < in.size() - 1; i++) {
		target[i] = T::random();
		// TODO random weight agor in seperate file
	}
}

TEST(PCS, RhoFactorise) {
	T c(q), a, b;
	a.set(5,0);
	b.set(26,0);
	PollardRho<RSACmp, T> rho(c);
	rho.run([](const T &in){
		const auto t = in*in + T(1);
		return t;
	}, a, b);

	EXPECT_EQ(a.value(), 26);
	EXPECT_EQ(b.value(), 2839);
}

TEST(PCS, RhoSubSetSum) {
	using T = kAry_Type_T<1u << 16u>;
	T target =, a, b;
	a.set(5,0);
	b.set(26,0);


	struct RSACmp {
		decltype(auto) operator()(T &a, T &b) const {
		}
	};
	PollardRho<RSACmp, T> rho(target);
	rho.run([](const T &in){
	  const auto t = in*in + T(1);
	  return t;
	}, a, b);

	EXPECT_EQ(a.value(), 26);
	EXPECT_EQ(b.value(), 2839);
}



int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
