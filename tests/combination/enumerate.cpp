#include <cstddef>
#include <bitset>
#include <gtest/gtest.h>

#define private public

#include "combination/chase.h"
#include "random.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

uint64_t d = 0;
auto print_change = [](const uint16_t *a) {
	d ^= 1ull << a[0];
	d ^= 1ull << a[1];
	std::bitset<10> dd(d);
	std::cout << dd << ":" << a[0] << " " << a[1] << std::endl;
};

TEST(enumerate_t, enumerate2_simple) {
	auto cf = enumerate_t<10, 2>();
	std::cout << d << std::endl;
	cf.enumerate(print_change);
}

int main(int argc, char **argv) {
	rng_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
