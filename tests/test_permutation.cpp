#include <gtest/gtest.h>
#include <utility>

#include "permutation/permutation.h"

using ::testing::InitGoogleTest;

TEST(Permutation, copy) {
	Permutation a(8);
	{
		Permutation b = a;
		b.values[0] = 7;
		EXPECT_EQ(a.values[0], 0u);
		EXPECT_EQ(b.length, 8u);
	}

	Permutation c(4);
	c = a;
	EXPECT_EQ(c.length, 8u);
	for (uint32_t i = 0; i < 8; i++) { EXPECT_EQ(c.values[i], i); }
}

TEST(Permutation, move) {
	Permutation a(8);
	Permutation b = std::move(a);
	EXPECT_EQ(a.values, nullptr);
	EXPECT_EQ(b.length, 8u);

	Permutation c(2);
	c = std::move(b);
	EXPECT_EQ(c.length, 8u);
	EXPECT_EQ(c.values[7], 7u);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
