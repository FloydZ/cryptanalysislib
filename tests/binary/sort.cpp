#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>
#include <bitset>

#include "binary.h"
#include "helper.h"
#include "sort.h"

constexpr uint64_t ListSize = 2;

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// TODO reactivate
//TEST(Bucket_Sort, get_data) {
//	mzd_t *A_ = mzd_init(n, n);
//	Matrix_T<mzd_t *> A((mzd_t *)A_);
//	A.gen_identity(n);
//
//	BinaryList L{1};
//	L.generate_base_random(1, A);
//	L.set_load(1);
//
//	for (int i = 0; i < L.get_load(); ++i) {
//		std::cout << L[i].get_label();
//		std::cout << L[i].get_label().data().get_bits(0, 8) << "\n";
//		std::cout << std::bitset<64>(L[i].get_label().data().get_bits(0, 8)) << "\n";
//
//		std::cout << L[i].get_label().data().get_bits(60, 68) << "\n";
//		std::cout << std::bitset<64>(L[i].get_label().data().get_bits(60, 68)) << "\n";
//	}
//}
//
//TEST(Bucket_Sort, asdasd) {
//	mzd_t *A_ = mzd_init(n, n);
//	Matrix_T<mzd_t *> A((mzd_t *)A_);
//	A.gen_identity(n);
//
//	BinaryList L{0}; L.generate_base_random(ListSize, A);
//	L.set_load(ListSize);
//
//	auto *a = new Bucket_Sort<BinaryList, 200, 2, 4, n>{};
//	a->hash(L);
//
//	for (int i = 0; i < a->n_buckets; ++i) {
//		std::cout << "bucket: " << i << ":\n";
//		for (int j = 0; j < a->buckets_load[i]; ++j) {
//			std::cout << a->buckets[i][j].first << ":\t " << std::bitset<n>(a->buckets[i][j].second) << " \n";
//		}
//		std::cout << "\n";
//	}
//
//	uint64_t bucket, lower, upper;
//	auto t = a->find(L[0].get_label(), &bucket, &lower, &upper);
//
//	std::cout << L[0].get_label();
//	std::cout << bucket << " " << lower << " " << upper << "\n";
//}

template<const uint32_t l, const uint32_t h>
static uint64_t HashSearch(uint64_t a) {
	constexpr uint64_t mask = (~((uint64_t(1) << l) - 1u)) & ((uint64_t (1) << h) - 1u);
	return (a&mask);
}

template<const uint32_t l, const uint32_t h>
static uint64_t Hash(uint64_t a) {
	return HashSearch<l, h>(a) >> l;
}


constexpr uint64_t size_bucket = 2000, number_bucket = 15;
constexpr uint32_t b0 = 0, b1 = number_bucket, b2 = number_bucket;
constexpr uint32_t threads = 8;
constexpr uint64_t LSize = 10553600;
constexpr uint64_t loops = 100;


using BinaryList2 = Parallel_List_T<BinaryElement>;
using LPartType = uint64_t;
using IndexType = uint64_t;
constexpr static ConfigParallelBucketSort chm1{b0, b1, b2, size_bucket, uint64_t(1) << number_bucket,
                                               threads, 1, 20, 10, 0, 0,
                                               true, false, false, true};

constexpr static ConfigParallelBucketSort chm2{b0, b1-5, b2, size_bucket, uint64_t(1) << number_bucket,
                                               threads, 1, 20, 10, 0, 0,
                                               true, false, false, true};
TEST(ParallelLockFreeBucketSort, first) {
	using HM1Type = ParallelLockFreeBucketSort<chm1, BinaryList2, LPartType, IndexType, &Hash<0, 0+number_bucket>, &HashSearch<0, 0+number_bucket>>;
	auto *hm = new HM1Type();
	using Extractor = WindowExtractor<BinaryLabel, LPartType>;
	auto extractor = [](const BinaryLabel &label1) -> LPartType {
		return Extractor::template extract<0, 20>(label1);
	};

	BinaryList2 L{LSize, threads, LSize/threads};
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

			if (j != loops-1)
				hm->reset(tid);
		}
	}

	uint64_t time = clock() - t1;
	std::cout << "Time: " << time << "\n";
	std::cout << "load: " << hm->load() << ", size: " << hm->size() <<  "\n";

	uint64_t load = 0ul;
	auto poss = hm->find(extractor(L.data_label(30)), load);
	//ASSERT(hm->__buckets[poss].second[0] == 30);
}


TEST(ParallelBucketSort, first) {
	using HM1Type = ParallelBucketSort<chm1, BinaryList2, LPartType, IndexType, &Hash<0, 0+number_bucket>>;
	auto *hm = new HM1Type();
	using Extractor = WindowExtractor<BinaryLabel, LPartType>;
	auto extractor = [](const BinaryLabel &label1) -> LPartType {
		return Extractor::template extract<0, 20>(label1);
	};

	BinaryList2 L{LSize, threads, LSize/threads};
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
				hm->insert1(data, pos, tid);
			}

			if (j != loops-1)
				hm->reset(tid);
		}

#pragma omp barrier
		hm->sort(tid);
	}

	uint64_t time = clock() - t1;
	std::cout << "Time: " << time << "\n";
	std::cout << "load: " << hm->load() << ", size: " << hm->size() <<  "\n";

	uint64_t load = 0ul;
	auto poss = hm->find(extractor(L.data_label(30)), load);
	//ASSERT(hm->__buckets[poss].second[0] == 30);
}

TEST(ParallelBucketSort, need2sort) {
	using HM1Type = ParallelBucketSort<chm2, BinaryList2, LPartType, IndexType, &Hash<0, 0+number_bucket>>;
	auto *hm = new HM1Type();
	using Extractor = WindowExtractor<BinaryLabel, LPartType>;
	auto extractor = [](const BinaryLabel &label1) -> LPartType {
		return Extractor::template extract<0, 20>(label1);
	};

	BinaryList2 L{LSize, threads, LSize/threads};
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
				hm->insert1(data, pos, tid);
			}

			if (j != loops-1)
				hm->reset(tid);
		}

#pragma omp barrier
		hm->sort(tid);
	}

	uint64_t time = clock() - t1;
	std::cout << "Time: " << time << "\n";
	std::cout << "load: " << hm->load() << ", size: " << hm->size() <<  "\n";

	uint64_t load = 0ul;
	auto poss = hm->find(extractor(L.data_label(30)), load);
	//ASSERT(hm->__buckets[poss].second[0] == 30);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
