#include <gtest/gtest.h>

#include "combination/chase.h"
#include "container/fq_vector.h"
#include "container/hashmap.h"
#include "helper.h"
#include "list/enumeration/enumeration.h"
#include "list/enumeration/fq.h"
#include "list/list.h"
#include "matrix/matrix.h"
#include "random.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr uint32_t n = 20;
constexpr uint32_t k = 2;
constexpr uint32_t l = 10;
constexpr uint32_t q = 4;
constexpr uint32_t w = 1;

constexpr uint32_t qprime = q - 1;
constexpr size_t list_size = compute_combinations_fq_chase_list_size<n, q, w>();
constexpr size_t chase_size = bc(n, w);

using T = uint8_t;
using Value = FqPackedVector<n, q, T>;
using Label = FqPackedVector<n - k, q, T>;
using Matrix = FqMatrix<T, n, n - k, q>;// NOTE this is the transposed type
using Element = Element_T<Value, Label, Matrix>;
using List = List_T<Element>;

using K = uint16_t;
using V = size_t[1];

constexpr static SimpleHashMapConfig simple{10, 1u << l, 1};
using HMType = SimpleHashMap<K, V, simple, Hash<uint64_t, 0, l, q>>;
using load_type = HMType::load_type;

TEST(ListEnumerateMultiFullLength, single_list) {
	List L(list_size);
	Matrix HT;
	HT.random();

	Label syndrome;
	syndrome.random();
	ListEnumerateMultiFullLength<List, n, q, w> enumerator{HT, 0, &syndrome};
	//enumerator.run<std::nullptr_t, std::nullptr_t, std::nullptr_t>(&L, nullptr);
	enumerator.run(&L);

	for (size_t i = 0; i < list_size; ++i) {
		EXPECT_EQ(L.data_value(i).popcnt(), w);
		EXPECT_EQ(L.data_label(i).is_zero(), false);
	}
}

TEST(ListEnumerateMultiFullLength, single_hashmap) {
	HMType hm;
	List L(list_size);
	auto extractor = [](const Label label) {
		constexpr uint64_t mask = (1ul << l) - 1ul;
		return (label.ptr()[0]) & mask;
	};

	Matrix HT;
	HT.random();

	Label syndrome;
	syndrome.zero();
	ListEnumerateMultiFullLength<List, n, q, w> enumerator{HT, 0, &syndrome};
	enumerator.run<HMType, decltype(extractor), std::nullptr_t>(&L, nullptr, 0, 0, 0, &hm, &extractor, nullptr);

	for (size_t i = 0; i < list_size; ++i) {
		EXPECT_EQ(L.data_value(i).popcnt(), w);
		EXPECT_EQ(L.data_label(i).is_zero(), false);

		const auto data = extractor(L.data_label(i));
		load_type load = 0;
		const auto pos = hm.find(data, load);

		// make sure we found something
		EXPECT_NE(pos, size_t(-1));
	}
}

/// the labels are updated incrementally, they must be equal to HT*e + s
TEST(ListEnumerateMultiFullLength, labels_with_syndrome) {
	List L(list_size);
	Matrix HT;
	HT.random();

	Label syndrome;
	syndrome.random();
	ListEnumerateMultiFullLength<List, n, q, 2> enumerator{HT, 0, &syndrome};
	constexpr size_t ls = ListEnumerateMultiFullLength<List, n, q, 2>::max_list_size;
	List L2(ls);
	enumerator.run(&L2);

	for (size_t i = 0; i < ls; ++i) {
		Element t = L2[i];
		t.recalculate_label(HT);
		Label::add(t.label, t.label, syndrome);
		EXPECT_TRUE(t.label.is_equal(L2.data_label(i)));
		EXPECT_EQ(L2.data_value(i).popcnt(), 2);
	}
}

/// value mod 3, label mod 2**32 (as in `BCJ`): the label update must not
/// rely on `q * HT[i] == 0`.
TEST(ListEnumerateMultiFullLength, labels_ternary_subsetsum) {
	using T = uint64_t;
	constexpr uint32_t n = 16;
	constexpr uint64_t q = 1ull << 32;
	using Value = FqPackedVector<n, 3, T, true>;
	using Label = kAry_Type_T<q>;
	using Matrix = FqVector<T, n, q>;
	using Element = Element_T<Value, Label, Matrix>;
	using List = List_T<Element>;
	using Enumerator = ListEnumerateMultiFullLength<List, n/2, 3, 2>;

	constexpr size_t ls = Enumerator::max_list_size;
	List L1(ls), L2(ls);
	Matrix A;
	A.random();
	Enumerator en{A};
	en.run(&L1, &L2, n/2);

	for (size_t i = 0; i < ls; ++i) {
		for (const List *L : {&L1, &L2}) {
			Element t = (*L)[i];
			t.recalculate_label(A);
			EXPECT_TRUE(t.label.is_equal(L->data_label(i)));
		}
	}
}

TEST(ListEnumerateMultiFullLength, two_lists) {
	constexpr size_t list_size = compute_combinations_fq_chase_list_size<n / 2, q, w>();
	List L1(list_size);
	List L2(list_size);

	Matrix HT;
	HT.random();

	Label syndrome;
	syndrome.zero();
	ListEnumerateMultiFullLength<List, n / 2, q, w> enumerator{HT, 0, &syndrome};
	//enumerator.run<std::nullptr_t, std::nullptr_t, std::nullptr_t>(&L1, &L2, n / 2);
	enumerator.run(&L1, &L2, n / 2);

	for (size_t i = 0; i < list_size; ++i) {
		EXPECT_EQ(L1.data_value(i).popcnt(), w);
		EXPECT_EQ(L2.data_value(i).popcnt(), w);

		for (uint32_t j = 0; j < n / 2; j++) {
			EXPECT_EQ(L1.data_value(i).get(j + n / 2), 0);
			EXPECT_EQ(L2.data_value(i).get(j), 0);
		}

		EXPECT_EQ(L1.data_label(i).is_zero(), false);
		// NOTE: this can happen in such small matrices: EXPECT_EQ(L2.data_label(i).is_zero(), false);
	}
}


TEST(ListEnumerateSingleFullLength, single_list) {
	List L(chase_size);
	Matrix HT;
	HT.random();
	ListEnumerateSingleFullLength<List, n, q, w> enumerator{qprime, HT};
	// enumerator.run<std::nullptr_t, std::nullptr_t, std::nullptr_t>(&L, nullptr);
	enumerator.run(&L, nullptr);

	for (size_t i = 0; i < chase_size; ++i) {
		std::cout << i << " " << L.data_value(i).popcnt() << std::endl;
		EXPECT_EQ(L.data_value(i).popcnt(), w);
		EXPECT_EQ(L.data_label(i).is_zero(), false);
	}
}

TEST(ListEnumerateSingleFullLength, single_hashmap) {
	HMType hm;
	List L(chase_size);
	auto extractor = [](const Label l) {
		return l.ptr()[0];
	};

	Matrix HT;
	HT.random();

	ListEnumerateSingleFullLength<List, n, q, w> enumerator{qprime, HT};
	enumerator.run<HMType, decltype(extractor), std::nullptr_t>(&L, nullptr, 0, 0, 0, &hm, &extractor, nullptr);

	for (size_t i = 0; i < chase_size; ++i) {
		const auto data = extractor(L.data_label(i));
		load_type load = 0;
		const auto pos = hm.find(data, load);

		// make sure we found something
		EXPECT_NE(pos, size_t(-1));
	}
}

TEST(ListEnumerateSingleFullLength, two_lists) {
	constexpr size_t list_size = bc(n / 2, w);
	List L1(list_size);
	List L2(list_size);

	Matrix HT;
	HT.random();

	ListEnumerateSingleFullLength<List, n / 2, q, w> enumerator{qprime, HT};
	// enumerator.run<std::nullptr_t, std::nullptr_t, std::nullptr_t>(&L1, &L2, n / 2);
	enumerator.run(&L1, &L2, n / 2);

	for (size_t i = 0; i < list_size; ++i) {
		EXPECT_EQ(L1.data_value(i).popcnt(), w);
		EXPECT_EQ(L2.data_value(i).popcnt(), w);

		for (uint32_t j = 0; j < n / 2; j++) {
			EXPECT_EQ(L1.data_value(i).get(j + n / 2), 0);
			EXPECT_EQ(L2.data_value(i).get(j), 0);
		}

		EXPECT_EQ(L1.data_label(i).is_zero(), false);
		EXPECT_EQ(L2.data_label(i).is_zero(), false);
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
	ListEnumerateSinglePartialSingle<List, n, q, mitm_w, noreps_w, split> enumerator{q - 1, HT};
	// enumerator.template run<std::nullptr_t, std::nullptr_t, std::nullptr_t>(L1, L2, L3, L4);
	enumerator.run(L1, L2, L3, L4);

	for (size_t i = 0; i < list_size; ++i) {
		EXPECT_EQ(L1.data_value(i).popcnt(), mitm_w + noreps_w);
		EXPECT_EQ(L2.data_value(i).popcnt(), mitm_w + noreps_w);
		EXPECT_EQ(L3.data_value(i).popcnt(), mitm_w + noreps_w);
		EXPECT_EQ(L4.data_value(i).popcnt(), mitm_w + noreps_w);
	}
}

/// noreps_w = 3: weights and labels (HT*e) after the reset of the noreps
/// part, L1 is hashed into `hm` and L3 into `hm3`.
TEST(ListEnumerateSinglePartialSingle, hashmap_noreps3) {
	constexpr uint32_t n = 32, k = 2, q = 4;
	using Value = FqPackedVector<n, q, T>;
	using Label = FqPackedVector<n - k, q, T>;
	using Matrix = FqMatrix<T, n, n - k, q>;
	using Element = Element_T<Value, Label, Matrix>;
	using List = List_T<Element>;

	constexpr uint32_t mitm_w = 2;
	constexpr uint32_t noreps_w = 3;
	constexpr uint32_t split = n / 2;
	using Enumerator = ListEnumerateSinglePartialSingle<List, n, q, mitm_w, noreps_w, split>;
	constexpr size_t list_size = Enumerator::LIST_SIZE;

	using HK = uint64_t;
	constexpr static SimpleHashMapConfig cfg{32, 1u << l, 1};
	using HM = SimpleHashMap<HK, V, cfg, Hash<HK, 0, l, 2>>;
	auto extractor = [](const Label label) -> HK {
		constexpr uint64_t mask = (1ull << l) - 1ull;
		return label.ptr()[0] & mask;
	};

	List L1(list_size), L2(list_size), L3(list_size), L4(list_size);
	auto *hm = new HM{}, *hm3 = new HM{};
	Matrix HT;
	HT.random();
	Enumerator enumerator{q - 1, HT};
	enumerator.template run<HM, decltype(extractor), std::nullptr_t>
	        (L1, L2, L3, L4, 0, hm, &extractor, nullptr, hm3);

	auto contains = [&](const HM *h, const List &L, const size_t i) {
		typename HM::load_type load = 0;
		const auto pos = h->find(extractor(L.data_label(i)), load);
		for (size_t j = 0; j < load; ++j) {
			if (h->ptr(pos + j)[0] == i) { return true; }
		}
		return false;
	};

	for (const List *L : {&L1, &L2, &L3, &L4}) {
		for (size_t i = 0; i < list_size; ++i) {
			EXPECT_EQ(L->data_value(i).popcnt(), mitm_w + noreps_w);
			Element t = (*L)[i];
			t.recalculate_label(HT);
			EXPECT_TRUE(t.label.is_equal(L->data_label(i)));

			// all elements are distinct
			for (size_t j = 0; j < i; ++j) {
				EXPECT_FALSE(L->data_value(i).is_equal(L->data_value(j)));
			}
		}
	}

	for (size_t i = 0; i < list_size; ++i) {
		EXPECT_TRUE(contains(hm, L1, i));
		EXPECT_TRUE(contains(hm3, L3, i));
	}

	delete hm; delete hm3;
}

TEST(ListEnumerateSinglePartialSingle, simple_nohashmap_subsetsum) {
	using T = uint64_t;

	constexpr uint64_t n = 20;
	constexpr uint64_t q = 1ull<<n;
	using Label = kAry_Type_T<q>;
	using Value = FqPackedVector<n, q, T>;
	using Matrix = FqVector<T, n, q>;
	using Element = Element_T<Value, Label, Matrix>;
	using List = List_T<Element>;

	constexpr uint32_t mitm_w = 2;
	constexpr uint32_t noreps_w = 2;
	constexpr uint32_t split = n / 2;
	using Enumerator = ListEnumerateSinglePartialSingle<List, n, q, mitm_w, noreps_w, split>;

	constexpr size_t list_size = Enumerator::LIST_SIZE;
	List L1(list_size), L2(list_size), L3(list_size), L4(list_size);
	Matrix HT;
	HT.random();
	Enumerator enumerator{3, HT};
	enumerator.run(L1, L2, L3, L4);

	for (size_t i = 0; i < list_size; ++i) {
		EXPECT_EQ(L1.data_value(i).popcnt(), mitm_w + noreps_w);
		EXPECT_EQ(L2.data_value(i).popcnt(), mitm_w + noreps_w);
		EXPECT_EQ(L3.data_value(i).popcnt(), mitm_w + noreps_w);
		EXPECT_EQ(L4.data_value(i).popcnt(), mitm_w + noreps_w);

		// EXPECT_EQ(L1.data_value(i).popcnt(0, split), mitm_w);
		EXPECT_EQ(L2.data_value(i).popcnt(0, split), mitm_w);
		EXPECT_EQ(L3.data_value(i).popcnt(0, split), mitm_w);
		EXPECT_EQ(L4.data_value(i).popcnt(0, split), mitm_w);

		// EXPECT_EQ(L1.data_value(i).popcnt(split, n), noreps_w);
		EXPECT_EQ(L2.data_value(i).popcnt(split, n), noreps_w);
		EXPECT_EQ(L3.data_value(i).popcnt(split, n), noreps_w);
		EXPECT_EQ(L4.data_value(i).popcnt(split, n), noreps_w);
	}

}

TEST(BinarySinglePartialSingleEnumerator, simple_nohashmap_subsetsum) {
	using T = uint64_t;

	constexpr uint64_t n = 20;
	constexpr uint64_t q = 1ull<<n;
	using Label = kAry_Type_T<q>;
	using Value = FqPackedVector<n, q, T>;
	using Matrix = FqVector<T, n, q>;
	using Element = Element_T<Value, Label, Matrix>;
	using List = List_T<Element>;

	constexpr uint32_t mitm_w = 2;
	constexpr uint32_t noreps_w = 2;
	constexpr uint32_t split = n / 2;
	using Enumerator = BinarySinglePartialSingleEnumerator<List, n, mitm_w, noreps_w, split>;
	Enumerator ::info();

	constexpr size_t list_size = Enumerator::LIST_SIZE;
	List L1(list_size), L2(list_size), L3(list_size), L4(list_size);
	Matrix HT;
	HT.random();
	Enumerator enumerator{HT};
	enumerator.run(L1, L2, L3, L4);

	for (size_t i = 0; i < list_size; ++i) {
		EXPECT_EQ(L1.data_value(i).popcnt(), mitm_w + noreps_w);
		EXPECT_EQ(L2.data_value(i).popcnt(), mitm_w + noreps_w);
		EXPECT_EQ(L3.data_value(i).popcnt(), mitm_w + noreps_w);
		EXPECT_EQ(L4.data_value(i).popcnt(), mitm_w + noreps_w);

		/// NOTE: as we enforce a certain overlap, this is not a valid test
		// EXPECT_EQ(L1.data_value(i).popcnt(0, split), mitm_w);
		EXPECT_EQ(L2.data_value(i).popcnt(0, split), mitm_w);
		EXPECT_EQ(L3.data_value(i).popcnt(0, split), mitm_w);
		EXPECT_EQ(L4.data_value(i).popcnt(0, split), mitm_w);

		// EXPECT_EQ(L1.data_value(i).popcnt(split, n), noreps_w);
		EXPECT_EQ(L2.data_value(i).popcnt(split, n), noreps_w);
		EXPECT_EQ(L3.data_value(i).popcnt(split, n), noreps_w);
		EXPECT_EQ(L4.data_value(i).popcnt(split, n), noreps_w);
	}
}

/// L1 is hashed into `hm`, L3 into `hm3`, each with its list index
TEST(BinarySinglePartialSingleEnumerator, hashmap_subsetsum) {
	using T = uint64_t;

	constexpr uint64_t n = 20;
	constexpr uint64_t q = 1ull<<n;
	using Label = kAry_Type_T<q>;
	using Value = FqPackedVector<n, q, T>;
	using Matrix = FqVector<T, n, q>;
	using Element = Element_T<Value, Label, Matrix>;
	using List = List_T<Element>;

	constexpr uint32_t mitm_w = 2;
	constexpr uint32_t noreps_w = 2;
	constexpr uint32_t split = n / 2;
	using Enumerator = BinarySinglePartialSingleEnumerator<List, n, mitm_w, noreps_w, split>;

	using HK = uint64_t;
	constexpr static SimpleHashMapConfig cfg{16, 1u << l, 1};
	using HM = SimpleHashMap<HK, V, cfg, Hash<HK, 0, l, 2>>;
	auto extractor = [](const Label label) -> HK {
		constexpr uint64_t mask = (1ull << l) - 1ull;
		return label.value() & mask;
	};

	constexpr size_t list_size = Enumerator::LIST_SIZE;
	List L1(list_size), L2(list_size), L3(list_size), L4(list_size);
	auto *hm = new HM{}, *hm3 = new HM{};
	Matrix HT;
	HT.random();
	Enumerator enumerator{HT};
	enumerator.template run<HM, decltype(extractor), std::nullptr_t>
	        (L1, L2, L3, L4, 0, hm, &extractor, nullptr, hm3);

	auto contains = [&](const HM *h, const List &L, const size_t i) {
		typename HM::load_type load = 0;
		const auto pos = h->find(extractor(L.data_label(i)), load);
		for (size_t j = 0; j < load; ++j) {
			if (h->ptr(pos + j)[0] == i) { return true; }
		}
		return false;
	};

	for (size_t i = 0; i < list_size; ++i) {
		EXPECT_TRUE(contains(hm, L1, i));
		EXPECT_TRUE(contains(hm3, L3, i));
	}

	delete hm; delete hm3;
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
	enumerator.template run<std::nullptr_t, std::nullptr_t>(L1, L2, L3, L4);

	for (size_t i = 0; i < list_size; ++i) {
		EXPECT_EQ(L1.data_value(i).popcnt(), mitm_w + noreps_w);
		EXPECT_EQ(L2.data_value(i).popcnt(), mitm_w + noreps_w);
		EXPECT_EQ(L3.data_value(i).popcnt(), mitm_w + noreps_w);
		EXPECT_EQ(L4.data_value(i).popcnt(), mitm_w + noreps_w);
	}

	// every coordinate of the noreps part [split, n) is enumerated
	std::vector<bool> covered(n, false);
	for (const List *L : {&L1, &L2, &L3, &L4}) {
		for (size_t i = 0; i < list_size; ++i) {
			for (uint32_t j = split; j < n; ++j) {
				if (L->data_value(i)[j]) { covered[j] = true; }
			}
		}
	}
	for (uint32_t j = split; j < n; ++j) { EXPECT_TRUE(covered[j]) << j; }
}

/// L1 is hashed into `hm`, L3 into `hm3`, each with the label of its list entry
/// (index = position in the list)
TEST(ListEnumerateMultiDisjointBlock, hashmap) {
	constexpr uint32_t mitm_w = 2;
	constexpr uint32_t noreps_w = 1;
	constexpr uint32_t split = n / 2;
	using Enumerator = ListEnumerateMultiDisjointBlock<List, n, q, mitm_w, noreps_w, split>;
	constexpr size_t list_size = Enumerator::LIST_SIZE;

	using HK = uint64_t;
	constexpr static SimpleHashMapConfig cfg{32, 1u << l, 1};
	using HM = SimpleHashMap<HK, V, cfg, Hash<HK, 0, l, 2>>;
	auto extractor = [](const Label label) -> HK {
		constexpr uint64_t mask = (1ull << l) - 1ull;
		return (uint64_t(label.ptr()[0]) | (uint64_t(label.ptr()[1]) << 8u)) & mask;
	};

	List L1(list_size), L2(list_size), L3(list_size), L4(list_size);
	auto *hm = new HM{}, *hm3 = new HM{};
	Matrix HT;
	HT.random();
	Enumerator enumerator{HT};
	enumerator.template run<HM, decltype(extractor)>(L1, L2, L3, L4, 0, hm, &extractor, hm3);

	auto contains = [&](const HM *h, const List &L, const size_t i) {
		typename HM::load_type load = 0;
		const auto pos = h->find(extractor(L.data_label(i)), load);
		for (size_t j = 0; j < load; ++j) {
			if (h->ptr(pos + j)[0] == i) { return true; }
		}
		return false;
	};

	for (size_t i = 0; i < list_size; ++i) {
		EXPECT_TRUE(contains(hm, L1, i));
		EXPECT_TRUE(contains(hm3, L3, i));
	}

	delete hm; delete hm3;
}


int main(int argc, char **argv) {
	rng_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
