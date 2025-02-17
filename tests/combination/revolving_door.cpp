#include <cstddef>
#include <bitset>
#include <gtest/gtest.h>

#define private public

#include "combination/revolving_door.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

TEST(revolvingDoor, simple) {
    combination_revdoor c(10, 2);
    uint32_t k1 = 0, k2 = 1;
    do {
    	c.print_deltaset();
    	std::cout << " " << k1 << " " << k2 << std::endl;
    } while (c.next(&k1, &k2));
    return;
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
