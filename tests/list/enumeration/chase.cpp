#include <cstddef>
#include <bitset>
#include <gtest/gtest.h>

#define private public

#include "combination/chase2.h"
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

TEST(Chase, enumerate1_simple) {
	auto cf = chase_full<10, 1>{};
	d= 1;
	cf.enumerate1(print_change);
}

TEST(Chase, enumerate1_simple_backwards) {
	auto cf = chase_full<10, 1>{};
	d=1ull << (9);
	cf.enumerate1(print_change, 0, 10, false);
}

TEST(Chase, enumerate2) {
	d = 1;
	auto cf = chase_full<10, 3>{};
	std::cout << d << std::endl;
	cf.enumerate(print_change);
}


TEST(Chase, enumerate_kek) {
	d = 1;
	auto cf = chase_full<10, 1>{};
	std::cout << d << std::endl;
	cf.enumerate_v2(print_change);
}

int main(int argc, char **argv) {
	random_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
