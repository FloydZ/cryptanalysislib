#include <gtest/gtest.h>
#include <iostream>

#include "helper.h"
#include "binary.h"
#include "tree.h"

using BinaryExtendedTree = ExtendedTree<BinaryList>;

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


constexpr size_t ListSize = 10;

TEST(extendedTree, join2lists) {
	constexpr uint32_t l = 0, h = 4;
	constexpr uint32_t threads = 1;
	BinaryMatrix A;
	A.identity();

	BinaryList L1{ListSize, threads}, L2{ListSize, threads}, out{2*ListSize, threads};

	L1.generate_base_random(ListSize, A);
	L2.generate_base_random(ListSize, A);

	BinaryExtendedTree::template join2lists
	        <l, h, 5>
	        (out, L1, L2);

	for (uint32_t tid = 0; tid < threads; tid++) {
		std::cout << "tid: " << tid << ", load:" << out.load(tid) << std::endl;
		EXPECT_GE(out.load(tid), 0);
		for (size_t i = 0; i < out.load(tid); ++i) {
			for (uint32_t j = l; j < h; ++j) {
				ASSERT_EQ(out[tid*out.thread_block_size() + i].label[j], 0);
			}
		}
	}
}

TEST(extendedTree, stream_join) {
	constexpr uint32_t threads = 1;
	BinaryMatrix A;
	A.identity();

	BinaryLabel target;
	target.zero();

	BinaryList L1{ListSize, threads}, L2{ListSize, threads};
	L1.generate_base_random(ListSize, A);
	L2.generate_base_random(ListSize, A);

	BinaryExtendedTree::streamjoin(L1, L2, target);
}


#ifndef EXTERNAL_MAIN
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
#endif
