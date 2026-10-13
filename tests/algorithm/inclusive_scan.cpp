#include <cstdint>
#include <numeric>
#include <vector>
#include <gtest/gtest.h>

#include "algorithm/inclusive_scan.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace cryptanalysislib;

/// all `inclusive_scan` overloads against std::inclusive_scan
TEST(InclusiveScan, overloads) {
	for (const size_t s : {size_t(0), size_t(1), size_t(13), size_t(1u << 20)}) {
		std::vector<uint64_t> in(s), e(s), o(s + 1, 0xdeadu);
		for (size_t i = 0; i < s; ++i) { in[i] = (i * 2654435761u) & 0xffu; }
		const auto op = [](const uint64_t a, const uint64_t b) { return a + b; };

		std::inclusive_scan(in.begin(), in.end(), e.begin(), op);
		EXPECT_EQ(cryptanalysislib::inclusive_scan(in.begin(), in.end(), o.begin(), op), o.begin() + s);
		for (size_t i = 0; i < s; ++i) { ASSERT_EQ(o[i], e[i]) << i; }
		EXPECT_EQ(o[s], 0xdeadu);

		std::inclusive_scan(in.begin(), in.end(), e.begin(), op, uint64_t(5));
		EXPECT_EQ(cryptanalysislib::inclusive_scan(in.begin(), in.end(), o.begin(), uint64_t(5), op), o.begin() + s);
		for (size_t i = 0; i < s; ++i) { ASSERT_EQ(o[i], e[i]) << i; }
		EXPECT_EQ(cryptanalysislib::inclusive_scan(par_if(true), in.begin(), in.end(), o.begin(), uint64_t(5), op), o.begin() + s);
		for (size_t i = 0; i < s; ++i) { ASSERT_EQ(o[i], e[i]) << i; }
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
