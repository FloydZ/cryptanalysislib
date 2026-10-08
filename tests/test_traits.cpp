#include <gtest/gtest.h>

#include "traits.h"

using ::testing::InitGoogleTest;

TEST(CallableMetadata, generatePointer) {
	auto make = [](int k) { return [k](int x) { return x + k; }; };
	auto c1 = make(1), c2 = make(100);
	using M = cryptanalysislib::CallableMetadata<decltype(c1)>;

	auto *p1 = M::generatePointer(c1);
	EXPECT_EQ(p1(0), 1);

	// one copy per closure type: the pointer calls the last closure
	auto *p2 = M::generatePointer(c2);
	EXPECT_EQ(p2(0), 100);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
