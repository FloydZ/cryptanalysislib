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


constexpr uint64_t ListSize = 10;

constexpr uint64_t b10 = 0;
constexpr uint64_t b11 = 3;
constexpr uint64_t b12 = 3;
constexpr uint64_t b20 = 3;
constexpr uint64_t b21 = 5;
constexpr uint64_t b22 = 5;

constexpr uint64_t NRB1 = 1ull << b12;
constexpr uint64_t NRB2 = 1ull << b22;
constexpr uint64_t SRB1 = 100ull;
constexpr uint64_t SRB2 = 100ull;
constexpr uint64_t threads  = 1;

constexpr uint64_t l  = 5;

using T = uint64_t;
using IndexType = uint64_t;
using LPartType = T;

template<const uint32_t l, const uint32_t h>
static T Hash(T a){
	constexpr T mask            = (~((T(1) << l) - 1u)) & ((T(1) << h) - 1u);
	return (a&mask)>>l;
}

//TEST(extendedTree, join2lists) {
//	mzd_t *A_ = mzd_init(n, n);
//	Matrix_T<mzd_t *> A((mzd_t *) A_);
//	A.gen_identity(n);
//
//	BinaryList L1{ListSize}, L2{ListSize};
//
//	L1.generate_base_random(ListSize, A);
//	L2.generate_base_random(ListSize, A);
//	using Extractor = WindowExtractor<BinaryLabel, LPartType>;
//
//	constexpr static ConfigParallelBucketSort chm1{b10, b11, b12, SRB1, NRB1, threads, 1, n - l, l, 0, 0, true, false, false, true};
//	constexpr static ConfigParallelBucketSort chm2{b10, b21, b22, SRB2, NRB2, threads, 1, n - l, l, 0, 0, true, false, false, true};
//
//	using HM1Type = ParallelBucketSort<chm1, BinaryList, T, IndexType, &Hash<b10, b12>>;
//	using HM2Type = ParallelBucketSort<chm2, BinaryList, T, IndexType, &Hash<b20, b22>>;
//	using TestExtendedTree = ExtendedTree<BinaryList, HM1Type>;
//	HM1Type *hm1; HM2Type *hm2;
//
//	hm1 = new HM1Type();
//	hm2 = new HM2Type();
//
//
//	auto extractor = [](const BinaryLabel &label) -> LPartType {
//		auto data = Extractor::template extract<n-l, n>(label);
//		return data;
//	};
//	TestExtendedTree::template join2lists<HM2Type>(*hm2, *hm1, L1, L2, extractor);
//	hm2->print();
//
//}

TEST(extendedTree, stream_join) {
	mzd_t *A_ = mzd_init(n, n);
	Matrix_T<mzd_t *> A((mzd_t *) A_);
	A.gen_identity(n);

	BinaryList L1{ListSize}, L2{ListSize};

	L1.generate_base_random(ListSize, A);
	L2.generate_base_random(ListSize, A);

	constexpr static ExtendenTreeConfig config{};
	constexpr static ConfigParallelBucketSort chm0{config.l[0], config.l[1], config.l[1], SRB1, NRB1, threads, 1, n - l, l, 0, 0, true, false, false, true};
	constexpr static ConfigParallelBucketSort chm1{config.l[1], config.l[2], config.l[2], SRB1, NRB1, threads, 1, n - l, l, 0, 0, true, false, false, true};
	constexpr static ConfigParallelBucketSort chm2{config.l[2], config.l[3], config.l[3], SRB1, NRB1, threads, 1, n - l, l, 0, 0, true, false, false, true};
	using HM0Type = ParallelBucketSort<chm0, BinaryList, T, IndexType, &Hash<config.l[0], config.l[1]>>;
	using HM1Type = ParallelBucketSort<chm1, BinaryList, T, IndexType, &Hash<config.l[1], config.l[2]>>;
	using HM2Type = ParallelBucketSort<chm2, BinaryList, T, IndexType, &Hash<config.l[2], config.l[3]>>;
	auto *hm0 = new HM0Type();
	auto *hm1 = new HM1Type();
	auto *hm2 = new HM2Type();

	using TestExtendedTree = ExtendedTree<BinaryList>;


	BinaryLabel target; target.random();
	//TestExtendedTree::template stream_join<1, limits>(L1, L2, target);
	//TestExtendedTree::stream_join(L1, L2, target);
	TestExtendedTree::template streamjoin<config, 0, HM0Type, HM1Type, HM2Type>(L1, L2, target);
}


#ifndef EXTERNAL_MAIN
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
#endif
