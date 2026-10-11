#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <set>
#include <vector>

#include "random.h"
#include "algorithm/set_intersection.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// sorted, duplicate free random uint16_t
static std::vector<uint16_t> sorted_unique(const size_t n, const uint32_t mod) {
	std::set<uint16_t> s;
	while (s.size() < n) { s.insert(uint16_t(rng() % mod)); }
	return {s.begin(), s.end()};
}

TEST(set_intersection, serial_and_simd_count) {
	for (const uint32_t mod : {300u, 2000u, 65536u}) {
		for (size_t n1 = 0; n1 <= 260; n1 += 13) {
			for (size_t n2 = 0; n2 <= 260; n2 += 17) {
				const auto a = sorted_unique(n1, mod), b = sorted_unique(n2, mod);
				std::vector<uint16_t> e;
				std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(e));
				EXPECT_EQ(internal::intersect_u16_serial(a.data(), b.data(), n1, n2), e.size());
				EXPECT_EQ(internal::galloping_intersection_size(a.data(), b.data(), n1, n2), e.size());
#if defined(USE_AVX512F) && defined(USE_AVX512BW)
				uint64_t c = 0;
				internal::simsimd_intersect_u16_ice(a.data(), b.data(), n1, n2, &c);
				ASSERT_EQ(c, e.size()) << n1 << " " << n2 << " " << mod;
#endif
			}
		}
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
