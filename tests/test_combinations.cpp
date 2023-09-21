#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include <array>

#include "m4ri/m4ri.h"
#include "combination/binary.h"
#include "test.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

/// weight to enumerate
constexpr uint64_t kk = 3;


TEST(Combinations_Chase_Binary_BinaryContainer, Left) {
	uint64_t nperm = 0;
	using BinaryContainer2 = BinaryContainer<n>;
	BinaryContainer2 e1, e2; e1.zero(); e2.zero();
	Combinations_Chase_Binary c{n, kk};
	c.left_init(e1.data().data());
	c.left_step(e1.data().data(), true);
	uint64_t rt = 1;

	while(rt != 0) {
		rt = c.left_step(e1.data().data());

		nperm += 1;
		EXPECT_EQ(false, e2.is_equal(e1, 0, n));
		e2 = e1;
	}

	EXPECT_EQ(nperm, bc(n, kk));
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
