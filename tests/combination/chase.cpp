#include <cstddef>
#include <bitset>
#include <gtest/gtest.h>

#define private public

#include "combination/chase.h"
#include "random.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

uint64_t d = 0;
auto print_change = [](const uint16_t a,
                       const uint16_t b) {
	d ^= 1ull << a;
  	d ^= 1ull << b;
	std::bitset<10> dd(d);
	std::cout << dd << ":" << a << " " << b << std::endl;
};

TEST(BinaryChaseSequence, simple) {
	std::vector<std::pair<uint16_t, uint16_t>> ret{};
	constexpr uint32_t n = 4;
	constexpr uint32_t p = 1;
	BinaryChaseEnumerator<n, p>::changelist(ret);
	d = (1u << p) - 1u;

	std::bitset<n> dd(d);
	std::cout << dd << std::endl;
	for (const auto &l: ret) {
		d ^= 1ull << l.first;
		d ^= 1ull << l.second;
		EXPECT_EQ(cryptanalysislib::popcount::popcount(d), p);
		std::bitset<n> dd(d);
		std::cout << dd << ":" << l.first << ":" << l.second << std::endl;
	}
}

int main(int argc, char **argv) {
	rng_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
