#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "container/hashmap.h"
#include "random.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib::hh;

/// rounding of the bucket count and the range/distribution of the buckets
TEST(fibonacci_growth_policy, buckets) {
	size_t c = 0;
	fibonacci_growth_policy p0(c);
	EXPECT_EQ(c, 0u);
	for (size_t h = 0; h < 1000; ++h) { EXPECT_EQ(p0.bucket_for_hash(h), 0u); }

	for (const size_t want : {1ul, 2ul, 3ul, 17ul, 1024ul, 1000000ul}) {
		size_t count = want;
		fibonacci_growth_policy p(count);
		EXPECT_GE(count, want);
		EXPECT_GE(count, 2u);
		EXPECT_EQ(count & (count - 1u), 0u) << "power of two " << want;
		EXPECT_EQ(p.next_bucket_count(), 2 * count);

		std::vector<size_t> hist(count, 0);
		const size_t samples = 64 * count;
		for (size_t i = 0; i < samples; ++i) {
			const size_t b = p.bucket_for_hash(i); // consecutive hashes: the worst case for a mask
			ASSERT_LT(b, count);
			hist[b] += 1;
		}
		// every bucket is used about equally often
		for (size_t b = 0; b < count; ++b) {
			EXPECT_GT(hist[b], 32u) << want << " " << b;
			EXPECT_LT(hist[b], 96u) << want << " " << b;
		}
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
