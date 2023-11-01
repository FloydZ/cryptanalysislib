#include <gtest/gtest.h>

#include "combination/chase.h"
#include "container/fq_vector.h"
#include "helper.h"
#include "list/enumeration/enumeration.h"
#include "list/enumeration/fq.h"
#include "list/list.h"
#include "matrix/fq_matrix.h"
#include "random.h"
#include "sort.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr uint32_t n = 20;
constexpr uint32_t l = 10;
constexpr uint32_t q = 4;
constexpr uint32_t w = 1;

constexpr uint32_t qprime = q-1;
constexpr size_t list_size = compute_combinations_fq_chase_list_size<n, q, w>();
constexpr size_t chase_size = bc(n, w);

using T = uint8_t;
using Value = kAryContainer_T<T, n, q>;
using Label = kAryContainer_T<T, n, q>;
using Matrix = FqMatrix<T, n, n, q>;
using Element = Element_T<Value, Label, Matrix>;
using List = List_T<Element>;
constexpr static ConfigParallelBucketSort chm1{0, l, l, l, l, 1, 1, n-l, l, 0};


using LPartType = uint16_t;
using IndexType = uint16_t;

inline static LPartType DummyHash(__uint128_t a) noexcept {
	constexpr __uint128_t mask = (1ull << (l)) - 1ull;
	return a&mask;
}
using HMType = ParallelBucketSort<chm1, List, LPartType, IndexType, &DummyHash>;

/// TODO somehow not full length
TEST(ListEnumerateMultiFullLength, single_list) {
	List L(list_size);
	Matrix HT;
	HT.random();
	ListEnumerateMultiFullLength<List, n, q, w> enumerator{HT};
	enumerator.run<std::nullptr_t, std::nullptr_t, std::nullptr_t>(&L, nullptr);

	for (size_t i = 0; i < list_size; ++i) {
		EXPECT_EQ(L.data_value(i).weight(), w);
	}
}

TEST(ListEnumerateMultiFullLength, single_hashmap) {
	HMType hm;
	List L(list_size);
	auto extractor = [](const Label l){
		return l.ptr()[0];
	};

	Matrix HT;
	HT.random();

	ListEnumerateMultiFullLength<List, n, q, w> enumerator{HT};
	enumerator.run<HMType, decltype(extractor), std::nullptr_t>(&L, nullptr, 0, 0, &hm, &extractor, nullptr);

	for (size_t i = 0; i < list_size; ++i) {
		const auto data = extractor(L.data_label(i));
		IndexType load=0,
		          pos = hm.find(data, load);

		// make sure we found something
		ASSERT_NE(pos, IndexType(-1));
	}
}

TEST(ListEnumerateMultiFullLength, two_lists) {
	constexpr size_t list_size = compute_combinations_fq_chase_list_size<n/2, q, w>();
	List L1(list_size);
	List L2(list_size);

	Matrix HT;
	HT.random();

	ListEnumerateMultiFullLength<List, n/2, q, w> enumerator{HT};
	enumerator.run<std::nullptr_t, std::nullptr_t, std::nullptr_t>(&L1, &L2, n/2);

	for (size_t i = 0; i < list_size; ++i) {
		EXPECT_EQ(L1.data_value(i).weight(), w);
		EXPECT_EQ(L2.data_value(i).weight(), w);

		for(uint32_t j = 0; j < n/2; j++) {
			EXPECT_EQ(L1.data_value(i).get(j + n/2), 0);
			EXPECT_EQ(L2.data_value(i).get(j), 0);
		}

		EXPECT_EQ(L1.data_label(i).is_zero(), false);
		EXPECT_EQ(L2.data_label(i).is_zero(), false);
	}
}


TEST(ListEnumerateSingleFullLength, single_list) {
	List L(chase_size);
	Matrix HT;
	HT.random();
	ListEnumerateSingleFullLength<List, n, q, w> enumerator{qprime, HT};
	enumerator.run<std::nullptr_t, std::nullptr_t, std::nullptr_t>(&L, nullptr);

	for (size_t i = 0; i < chase_size; ++i) {
		std::cout << i << " " << L.data_value(i).weight() << std::endl;
		ASSERT_EQ(L.data_value(i).weight(), w);
	}
}

TEST(ListEnumerateSingleFullLength, single_hashmap) {
	HMType hm;
	List L(chase_size);
	auto extractor = [](const Label l){
	  return l.ptr()[0];
	};

	Matrix HT;
	HT.random();

	ListEnumerateSingleFullLength<List, n, q, w> enumerator{qprime, HT};
	enumerator.run<HMType, decltype(extractor), std::nullptr_t>(&L, nullptr, 0, 0, &hm, &extractor, nullptr);

	for (size_t i = 0; i < chase_size; ++i) {
		const auto data = extractor(L.data_label(i));
		IndexType load=0,
				pos = hm.find(data, load);

		// make sure we found something
		ASSERT_NE(pos, IndexType(-1));
	}
}

TEST(ListEnumerateSingleFullLength, two_lists) {
	constexpr size_t list_size = bc(n/2, w);
	List L1(list_size);
	List L2(list_size);

	Matrix HT;
	HT.random();

	ListEnumerateSingleFullLength<List, n/2, q, w> enumerator{qprime, HT};
	enumerator.run<std::nullptr_t, std::nullptr_t, std::nullptr_t>(&L1, &L2, n/2);

	for (size_t i = 0; i < list_size; ++i) {
		ASSERT_EQ(L1.data_value(i).weight(), w);
		ASSERT_EQ(L2.data_value(i).weight(), w);

		for(uint32_t j = 0; j < n/2; j++) {
			ASSERT_EQ(L1.data_value(i).get(j + n/2), 0);
			ASSERT_EQ(L2.data_value(i).get(j), 0);
		}

		ASSERT_EQ(L1.data_label(i).is_zero(), false);
		ASSERT_EQ(L2.data_label(i).is_zero(), false);
	}
}



TEST(ListEnumerateSinglePartialSingle, simple_nohashmap) {
	constexpr uint32_t mitm_w = 2;
	constexpr uint32_t noreps_w = 1;
	constexpr uint32_t split = n / 2;
	constexpr size_t list_size = ListEnumerateSinglePartialSingle<List, n, q, mitm_w, noreps_w, split>::LIST_SIZE;
	List L1(list_size), L2(list_size), L3(list_size), L4(list_size);
	Matrix HT;
	HT.random();
	ListEnumerateSinglePartialSingle<List, n, q, mitm_w, noreps_w, split> enumerator{q-1, HT};
	enumerator.template run<nullptr_t, nullptr_t>(L1, L2, L3, L4);

	for (size_t i = 0; i < list_size; ++i) {
		ASSERT_EQ(L1.data_value(i).weight(), mitm_w + noreps_w);
		ASSERT_EQ(L2.data_value(i).weight(), mitm_w + noreps_w);
		ASSERT_EQ(L3.data_value(i).weight(), mitm_w + noreps_w);
		ASSERT_EQ(L4.data_value(i).weight(), mitm_w + noreps_w);
	}
}

TEST(ListEnumerateMultiDisjointBlock, simple_nohashmap) {
	constexpr uint32_t mitm_w = 2;
	constexpr uint32_t noreps_w = 1;
	constexpr uint32_t split = n / 2;
	constexpr size_t list_size = ListEnumerateMultiDisjointBlock<List, n, q, mitm_w, noreps_w, split>::LIST_SIZE;
	List L1(list_size), L2(list_size), L3(list_size), L4(list_size);
	Matrix HT;
	HT.random();
	ListEnumerateMultiDisjointBlock<List, n, q, mitm_w, noreps_w, split> enumerator{HT};
	enumerator.template run<nullptr_t, nullptr_t>(L1, L2, L3, L4);

	for (size_t i = 0; i < list_size; ++i) {
		ASSERT_EQ(L1.data_value(i).weight(), mitm_w + noreps_w);
		ASSERT_EQ(L2.data_value(i).weight(), mitm_w + noreps_w);
		ASSERT_EQ(L3.data_value(i).weight(), mitm_w + noreps_w);
		ASSERT_EQ(L4.data_value(i).weight(), mitm_w + noreps_w);
	}
}
int main(int argc, char **argv) {
	random_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
