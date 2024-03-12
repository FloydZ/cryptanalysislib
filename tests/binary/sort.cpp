#include <bitset>
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>

#include "element.h"
#include "list/list.h"
#include "matrix/fq_matrix.h"
#include "matrix/matrix.h"
#include "sort.h"
#include "tree.h"

constexpr uint32_t n = 50;
using BinaryValue = BinaryContainer<n>;
using BinaryLabel = BinaryContainer<n>;
using BinaryMatrix = FqMatrix<uint64_t, n, n, 2>;
using BinaryElement = Element_T<BinaryValue, BinaryLabel, BinaryMatrix>;
using BinaryList = List_T<BinaryElement>;
using BinaryTree = Tree_T<BinaryList>;

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(Bucket_Sort, get_data) {
	BinaryMatrix A;
	A.identity();

	BinaryList L{1};
	L.generate_base_random(1, A);
	L.set_load(1);

	for (size_t i = 0; i < L.load(); ++i) {
		std::cout << L[i].label;
		std::cout << L[i].label.get_bits(0, 8) << "\n";
		std::cout << std::bitset<64>(L[i].label.get_bits(0, 8)) << "\n";
	}
}


template<const uint32_t l, const uint32_t h>
static uint64_t HashSearch(uint64_t a) {
	constexpr uint64_t mask = (~((uint64_t(1) << l) - 1u)) & ((uint64_t(1) << h) - 1u);
	return (a & mask);
}

template<const uint32_t l, const uint32_t h>
static uint64_t Hash2(uint64_t a) {
	return HashSearch<l, h>(a) >> l;
}


constexpr uint64_t size_bucket = 2000, number_bucket = 15;
constexpr uint32_t b0 = 0, b1 = number_bucket, b2 = number_bucket;
constexpr uint32_t threads = 8;


using BinaryList2 = Parallel_List_T<BinaryElement>;
using LPartType = uint64_t;
using IndexType = uint64_t;
constexpr static ConfigParallelBucketSort chm1{b0, b1, b2, size_bucket, uint64_t(1) << number_bucket,
                                               threads, 1, 20, 10, 0, 0,
                                               true, false, false, true};

constexpr static ConfigParallelBucketSort chm2{b0, b1 - 5, b2, size_bucket, uint64_t(1) << number_bucket,
                                               threads, 1, 20, 10, 0, 0,
                                               true, false, false, true};

#ifdef USE_OMP

constexpr uint64_t LSize = 10553600;
constexpr uint64_t loops = 100;
TEST(ParallelBucketSort, first) {
	using HM1Type = ParallelBucketSort<chm1, BinaryList2, LPartType, IndexType, &Hash2<0, 0 + number_bucket>>;
	auto *hm = new HM1Type();
	using Extractor = WindowExtractor<BinaryLabel, LPartType>;
	auto extractor = [](const BinaryLabel &label1) -> LPartType {
		return Extractor::template extract<0, 20>(label1);
	};

	BinaryList2 L{
	        LSize,
	        threads,
	};
	for (uint64_t i = 0; i < LSize; ++i) {
		L.data_label(i).random();
	}

	// Approach2
	uint64_t t1 = clock();
#pragma omp parallel default(none) shared(L, hm, extractor) num_threads(threads)
	{
		uint32_t tid = omp_get_thread_num();
		const std::size_t s_tid = L.start_pos(tid);
		const std::size_t e_tid = L.end_pos(tid);

		uint64_t data;
		IndexType pos[1];
		for (uint64_t j = 0; j < loops; j++) {
			for (std::size_t i = s_tid; i < e_tid; ++i) {
				data = extractor(L.data_label(i));
				pos[0] = i;
				hm->insert(data, pos, tid);
			}

			if (j != loops - 1)
				hm->reset(tid);
		}

#pragma omp barrier
		hm->sort(tid);
	}

	uint64_t time = clock() - t1;
	std::cout << "Time: " << time << "\n";
	std::cout << "load: " << hm->load() << ", size: " << hm->size() << "\n";

	uint64_t load = 0ul;
	auto poss = hm->find(extractor(L.data_label(30)), load);
	//ASSERT(hm->__buckets[poss].second[0] == 30);
}

TEST(ParallelBucketSort, need2sort) {
	using HM1Type = ParallelBucketSort<chm2, BinaryList2, LPartType, IndexType, &Hash2<0, 0 + number_bucket>>;
	auto *hm = new HM1Type();
	using Extractor = WindowExtractor<BinaryLabel, LPartType>;
	auto extractor = [](const BinaryLabel &label1) -> LPartType {
		return Extractor::template extract<0, 20>(label1);
	};

	BinaryList2 L{LSize, threads};
	for (uint64_t i = 0; i < LSize; ++i) {
		L.data_label(i).random();
	}

	// Approach2
	uint64_t t1 = clock();
#pragma omp parallel default(none) shared(L, hm, extractor) num_threads(threads)
	{
		uint32_t tid = omp_get_thread_num();
		const std::size_t s_tid = L.start_pos(tid);
		const std::size_t e_tid = L.end_pos(tid);

		uint64_t data;
		IndexType pos[1];
		for (uint64_t j = 0; j < loops; j++) {
			for (std::size_t i = s_tid; i < e_tid; ++i) {
				data = extractor(L.data_label(i));
				pos[0] = i;
				hm->insert(data, pos, tid);
			}

			if (j != loops - 1)
				hm->reset(tid);
		}

#pragma omp barrier
		hm->sort(tid);
	}

	uint64_t time = clock() - t1;
	std::cout << "Time: " << time << "\n";
	std::cout << "load: " << hm->load() << ", size: " << hm->size() << "\n";

	uint64_t load = 0ul;
	auto poss = hm->find(extractor(L.data_label(30)), load);
	//ASSERT(hm->__buckets[poss].second[0] == 30);
}

#endif
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
