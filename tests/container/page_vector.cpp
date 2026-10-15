#include <gtest/gtest.h>
#include <cstdint>

#include "container/vector.h"

using ::testing::InitGoogleTest;

TEST(page_vector, push_back) {
	constexpr size_t size = 1000;
	page_vector<uint64_t, size> v;
	for (uint64_t i = 0; i < size; i++) { v.push_back(i * 3); }
	EXPECT_EQ(v.size(), size);
	for (uint64_t i = 0; i < size; i++) { EXPECT_EQ(v[i], i * 3); }

	// copy, clear and reuse: the freed pages are recycled
	page_vector<uint64_t, size> w = v;
	EXPECT_EQ(w.size(), size);
	EXPECT_EQ(w[size - 1], (size - 1) * 3);
	v.clear();
	v.shrink_to_fit();
	for (uint64_t i = 0; i < 10; i++) { v.push_back(i); }
	EXPECT_EQ(v[9], 9u);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
