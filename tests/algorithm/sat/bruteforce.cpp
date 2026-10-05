#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>


#include "algorithm/sat/bruteforce.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


template <typename T>
class SAT : public testing::Test {};

TYPED_TEST_SUITE_P(SAT);

TYPED_TEST_P(SAT, bruteforce) {
    constexpr static uint32_t n = 3;
    constexpr static uint32_t c = 6;

    uint32_t clauses[c] = {0b101, 0b110, 0b111, 0b100, 0b010, 0b000};
    sat_bruteforce<TypeParam, n>(clauses, c);
}

REGISTER_TYPED_TEST_SUITE_P(SAT, bruteforce);
using MyTypes = ::testing::Types<uint32_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, SAT, MyTypes);


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
