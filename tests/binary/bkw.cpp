#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "helper.h"
#include "binary.h"
#include "bkw.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// TODO unfinished
TEST(BKWTest, simple) {
	mzd_t *A_ = mzd_init(n, n);
	Matrix_T<mzd_t *> A((mzd_t *)A_);
	A.gen_identity(n);
	BinaryList L{0};
	constexpr uint64_t size = 100;
	L.generate_base_random(size, A);
	BKW<BinaryList>(L, 3, 5, 5);
}


#ifndef EXTERNAL_MAIN
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
#endif
