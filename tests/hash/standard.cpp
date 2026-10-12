#include <gtest/gtest.h>
#include <cstdint>

#include "hash/standard.h"

using ::testing::InitGoogleTest;

static_assert(StandardHashAble<uint32_t> && StandardHashAble<int64_t>);
static_assert(!StandardHashAble<double> && !StandardHashAble<int *>);

TEST(standard_hash, values) {
	EXPECT_EQ(hash<uint64_t>{}(0), 0x9e3779b9ull);
	EXPECT_EQ(hash<uint32_t>{}(1u), size_t(1u ^ (1u + 0x9e3779b9u + (1u << 6u) + (1u >> 2u))));
	EXPECT_NE(hash<uint64_t>{}(1), hash<uint64_t>{}(2));
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
