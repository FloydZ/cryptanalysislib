#include <cstddef>
#include <bitset>
#include <gtest/gtest.h>

#include "combination/bit_subset.h"
#include "container/binary_packed_vector.h"
#include "container/fq_packed_vector.h"
#include "random.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace cryptanalysislib;


using B = FqPackedVector<100, 2>;

TEST(BitSubset, simple) {
    bit_subset b(0b1101);
    for (uint32_t i =0; i < 10; i++) {
        b.next();
        std::cout << b.current() << std::endl;
    }
}

TEST(BitSubset, BinaryVector) {
    B bb;
    bb.set(100); 
    bit_subset<B> b(bb);
    for (uint32_t i =0; i < 10; i++) {
        b.next();
        // std::cout << b.current() << std::endl;
    }
}

int main(int argc, char **argv) {
	rng_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
