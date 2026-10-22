#include <gtest/gtest.h>
#include <cstdint>

#include "algorithm/pcs.h"
#include "container/kAry_type.h"
#include "algorithm/rsa.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

TEST(RSA, uint64) {
	constexpr static RSA_instance instance{21};
	rsa_pollard_rho<instance> rsa;
	EXPECT_TRUE(rsa.run());
}

TEST(RSA, uint32) {
	// 83 * 97
	constexpr static RSA_instance_T<uint32_t> instance{8051};
	using R = rsa_pollard_rho<instance>;
	static_assert(std::is_same_v<R::T::LimbType, uint32_t>);
	R rsa;
	EXPECT_TRUE(rsa.run());
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
