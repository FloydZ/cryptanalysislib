#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/random_index.h"
#include "algorithm/subsetsum.h"
#include "container/kAry_type.h"
#include "helper.h"
#include "matrix/matrix.h"
#include "tree.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

constexpr uint32_t n    = 16ul;
constexpr uint32_t q    = (1ul << n);

using T 			= uint64_t;
using Value     	= BinaryVector<n>;
using Label    		= kAry_Type_T<q>;
using Matrix 		= FqVector<T, n, q, true>;
using Element		= Element_T<Value, Label, Matrix>;
using List			= List_T<Element>;
using Tree			= Tree_T<List>;

using D = typename Label::DataType;
using E = std::pair<size_t, size_t>;
using EE = std::pair<E, E>;

// NOTE: the solution has weight n/2 = 4*p, so it is the sum of four
// 	elements with disjoint supports. A random intermediate target on 8 bits
// 	keeps one of its 36 representations with probability ~13%, hence the
// 	retries.
constexpr uint32_t p = n/8;
constexpr size_t baselist_size = sum_bc(n/2, p);
using Enumerator = BinaryLexicographicEnumerator<List, n/2, p>;

/// \return the number of entries in the hashmap
template<typename HM>
static size_t hashmap_load(const HM &hm) {
	size_t ret = 0;
	for (size_t i = 0; i < HM::nrbuckets; ++i) {
		ret += hm.load_without_hash(i);
	}
	return ret;
}

/// \return true if `e` is a solution of the subset sum instance on [0, k_upper)
static bool is_solution(const Matrix &A, const Element &e,
                        const Label &target, const uint32_t k_upper) {
	Label recalc;
	A.mul(recalc, e.value);
	return recalc.is_equal(target, 0, k_upper) &&
	       (e.value.popcnt() == n/2);
}

TEST(SubSetSum, constexpr_twolevel_streamjoin_on_iT_hashmap_v2_hashmap_out) {
	constexpr uint32_t k_lower1=0, k_upper1=5;
	constexpr uint32_t k_lower2=5, k_upper2=10;
	constexpr uint32_t k_lower3=10, k_upper3=16;

	Matrix A; A.random();
	List l1{baselist_size}, l2{baselist_size};
	Enumerator e{A};
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);

	constexpr static SimpleHashMapConfig simpleHashMapConfigL0 {
			100, 1ul<<(k_upper1-k_lower1), 1
	};
	constexpr static SimpleHashMapConfig simpleHashMapConfigL1 {
			100, 1ul<<(k_upper2-k_lower2), 1
	};
	constexpr static SimpleHashMapConfig simpleHashMapConfigL2 {
			100, 1ul<<(k_upper3-k_lower3), 1
	};
	using HML0 = SimpleHashMap<D, size_t, simpleHashMapConfigL0, Hash<D, k_lower1, k_upper1, 2>>;
	using HML1 = SimpleHashMap<D,      E, simpleHashMapConfigL1, Hash<D, k_lower2, k_upper2, 2>>;
	using HML2 = SimpleHashMap<D,     EE, simpleHashMapConfigL2, Hash<D, k_lower3, k_upper3, 2>>;
	HML0 *hml0 = new HML0{};
	HML1 *hml1 = new HML1{};
	HML2 *hml2 = new HML2{};

	Label iT, target, t1;
	iT.random(0, 1ull << k_upper1);
	target.random(0, 1ull << k_upper2);
	Label::sub(t1, target, iT);

	Tree t{1, A, 0};
	t.template join2lists_on_iT_v2
			<k_lower1, k_upper1>
			(*hml1, l1, l2, *hml0, iT, true);

	const size_t ret = t.template twolevel_streamjoin_on_iT_hashmap_v2
			<k_lower1, k_upper1, k_lower2, k_upper2>
			(*hml2, *hml1, l1, l2, *hml0, target, t1);

	// brute force
	size_t num = 0, num_weight = 0;
	Label e1, e2, s;
	Value v1, v2;
	for (size_t k = 0; k < l1.load(); ++k) {
		for (size_t b = 0; b < l2.load(); ++b) {
			Label::add(e1, l1[k].label, l2[b].label);
			if (!e1.is_equal(t1, k_lower1, k_upper1)) { continue; }
			Value::add(v1, l1[k].value, l2[b].value);
			for (size_t a1 = 0; a1 < l1.load(); ++a1) {
				for (size_t a2 = 0; a2 < l2.load(); ++a2) {
					Label::add(e2, l1[a1].label, l2[a2].label);
					if (!e2.is_equal(iT, k_lower1, k_upper1)) { continue; }
					Label::add(s, e1, e2);
					if (!s.is_equal(target, k_lower2, k_upper2)) { continue; }
					num += 1;
					Value::add(v2, l1[a1].value, l2[a2].value);
					Value::add(v2, v2, v1);
					num_weight += v2.popcnt() == n/2;
				}
			}
		}
	}

	EXPECT_GT(num, 0);
	EXPECT_EQ(ret, num);
	EXPECT_EQ(hashmap_load(*hml2), num);

	// filter function: only insert the sums with weight n/2
	hml2->clear();
	const size_t ret2 = t.template twolevel_streamjoin_on_iT_hashmap_v2
			<k_lower1, k_upper1, k_lower2, k_upper2>
			(*hml2, *hml1, l1, l2, *hml0, target, t1,
			 [&l1, &l2](HML2 &out, const Label &e, const EE &indices) {
				 Value v;
				 Value::add(v, l1[indices.first.first].value, l2[indices.first.second].value);
				 Value::add(v, v, l1[indices.second.first].value);
				 Value::add(v, v, l2[indices.second.second].value);
				 if (v.popcnt() == n/2) {
					 out.insert(e.value(), indices);
				 }
				 return false;
			 });
	EXPECT_EQ(ret2, num);
	EXPECT_EQ(hashmap_load(*hml2), num_weight);

	delete hml0;
	delete hml1;
	delete hml2;
}

TEST(SubSetSum, constexpr_join4lists_on_iT_hashmap_v2) {
	constexpr uint32_t k_lower1=0, k_upper1=8;
	constexpr uint32_t k_lower2=8, k_upper2=16;

	Matrix A; A.random();
	List out{1u<<10}, l1{baselist_size}, l2{baselist_size},
	                  l3{baselist_size}, l4{baselist_size};
	Enumerator e{A};
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l3, &l4, n/2);

	// NOTE: different orders of the elements in L3, L4 than in L1, L2
	for (size_t i = 0; i < l3.load()/2; ++i) {
		std::swap(l3[i], l3[l3.load() - 1 - i]);
	}
	for (size_t i = 0; i+1 < l4.load(); i += 2) {
		std::swap(l4[i], l4[i+1]);
	}

	constexpr static SimpleHashMapConfig simpleHashMapConfigL0 {
			100, 1ul<<(k_upper1-k_lower1), 1
	};
	constexpr static SimpleHashMapConfig simpleHashMapConfigL1 {
			100, 1ul<<(k_upper2-k_lower2), 1
	};
	using HML0 = SimpleHashMap<D, size_t, simpleHashMapConfigL0, Hash<D, k_lower1, k_upper1, 2>>;
	using HML1 = SimpleHashMap<D,      E, simpleHashMapConfigL1, Hash<D, k_lower2, k_upper2, 2>>;
	HML0 *hml0 = new HML0{};
	HML1 *hml1 = new HML1{};

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	Tree t{1, A, 0};
	uint32_t right = 0;
	for (uint32_t tries = 0; (tries < 128) && (right == 0); ++tries) {
		out.set_load(0);
		t.template join4lists_on_iT_hashmap_v2
				<k_lower1, k_upper1, k_lower2, k_upper2>
				(out, l1, l2, l3, l4, *hml0, *hml1, target, true);

		for (size_t i = 0; i < out.load(); ++i) {
			// every output is a solution
			EXPECT_TRUE(is_solution(A, out[i], target, n));
			EXPECT_TRUE(out[i].label.is_equal(target, 0, n));
			right += 1;
		}
	}

	EXPECT_GT(right, 0);
	delete hml0;
	delete hml1;
}

TEST(SubSetSum, constexpr_join4lists_on_iT_hashmap_v2_enumerator) {
	// NOTE: the four base lists of the enumerator have different supports:
	// 	weight 1 in the mitm part [0, n/2) and weight 1 in one quarter of
	// 	[n/2, n). So the four elements have the weight n/2 = 8 if they are
	// 	disjoint.
	constexpr uint32_t k_lower1=0, k_upper1=2;
	constexpr uint32_t k_lower2=2, k_upper2=4;
	using Enumerator4 = BinarySinglePartialSingleEnumerator<List, n, 1, 1, n/2>;

	Matrix A; A.random();
	constexpr size_t size = Enumerator4::LIST_SIZE;
	List out{1u<<12}, l1{size}, l2{size}, l3{size}, l4{size};

	Label target;
	target.random();

	Tree t{1, A, 0};
	size_t outputs = 0;
	for (uint32_t tries = 0; tries < 8; ++tries) {
		out.set_load(0);
		t.template join4lists_on_iT_hashmap_v2
				<k_lower1, k_upper1, k_lower2, k_upper2, Enumerator4>
				(out, l1, l2, l3, l4, target);

		for (size_t i = 0; i < out.load(); ++i) {
			EXPECT_TRUE(is_solution(A, out[i], target, k_upper2));
		}
		outputs += out.load();
	}

	EXPECT_GT(outputs, 0);
}

TEST(SubSetSum, constexpr_join4lists_twolists_on_iT_v2_hashmap_f) {
	constexpr uint32_t k_lower1=0, k_upper1=8;
	constexpr uint32_t k_lower2=8, k_upper2=16;

	Matrix A; A.random();
	List out{1u<<10}, l1{baselist_size}, l2{baselist_size};
	Enumerator e{A};
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);

	constexpr static SimpleHashMapConfig simpleHashMapConfigL0 {
			100, 1ul<<(k_upper1-k_lower1), 1
	};
	constexpr static SimpleHashMapConfig simpleHashMapConfigL1 {
			100, 1ul<<(k_upper2-k_lower2), 1
	};
	using HML0 = SimpleHashMap<D, size_t, simpleHashMapConfigL0, Hash<D, k_lower1, k_upper1, 2>>;
	using HML1 = SimpleHashMap<D,      E, simpleHashMapConfigL1, Hash<D, k_lower2, k_upper2, 2>>;
	HML0 *hml0 = new HML0{};
	HML1 *hml1 = new HML1{};

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	Tree t{1, A, 0};
	uint32_t right = 0;
	for (uint32_t tries = 0; (tries < 128) && (right == 0); ++tries) {
		// default functions: all matches are appended
		out.set_load(0);
		t.template join4lists_twolists_on_iT_v2
				<k_lower1, k_upper1, k_lower2, k_upper2>
				(out, l1, l2, *hml0, *hml1, target, true);

		for (size_t i = 0; i < out.load(); ++i) {
			EXPECT_TRUE(out[i].label.is_equal(target, k_lower1, k_upper2));
			right += is_solution(A, out[i], target, n);
		}
	}
	EXPECT_GT(right, 0);

	// user defined `f2`: stop at the first solution
	right = 0;
	for (uint32_t tries = 0; (tries < 128) && (right == 0); ++tries) {
		out.set_load(0);
		t.template join4lists_twolists_on_iT_v2
				<k_lower1, k_upper1, k_lower2, k_upper2>
				(out, l1, l2, *hml0, *hml1, target, true,
				 [](HML1 &out, const List &L1, const List &L2, const size_t i, const size_t j) {
					 Label e;
					 Label::add(e, L1[i].label, L2[j].label);
					 out.insert(e.value(), E{i, j});
					 return false;
				 },
				 [&A, &target](List &out, Element &a1, Element &a2, size_t, size_t, size_t, size_t) {
					 Element s;
					 Element::add(s, a1, a2);
					 if (!is_solution(A, s, target, n)) { return false; }
					 out[0] = s;
					 out.set_load(1);
					 return true;
				 });
		right += out.load();
	}
	EXPECT_EQ(right, 1);

	delete hml0;
	delete hml1;
}

TEST(SubSetSum, constexpr_join4lists_twolists_on_iT_hashmap_v2_iT) {
	constexpr uint32_t k_lower1=0, k_upper1=8;
	constexpr uint32_t k_lower2=8, k_upper2=16;

	Matrix A; A.random();
	List out{1u<<10}, l1{baselist_size}, l2{baselist_size};
	Enumerator e{A};
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);

	constexpr static SimpleHashMapConfig simpleHashMapConfigL0 {
			100, 1ul<<(k_upper1-k_lower1), 1
	};
	constexpr static SimpleHashMapConfig simpleHashMapConfigL1 {
			100, 1ul<<(k_upper2-k_lower2), 1
	};
	using HML0 = SimpleHashMap<D, size_t, simpleHashMapConfigL0, Hash<D, k_lower1, k_upper1, 2>>;
	using HML1 = SimpleHashMap<D,      E, simpleHashMapConfigL1, Hash<D, k_lower2, k_upper2, 2>>;
	HML0 *hml0 = new HML0{};
	HML1 *hml1 = new HML1{};

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	Tree t{1, A, 0};
	uint32_t right = 0;
	for (uint32_t tries = 0; (tries < 128) && (right == 0); ++tries) {
		Label iT;
		iT.random(0, 1ull << k_upper1);
		out.set_load(0);
		const size_t ret = t.template join4lists_twolists_on_iT_hashmap_v2
				<k_lower1, k_upper1, k_lower2, k_upper2>
				(out, l1, l2, *hml0, *hml1, target, iT, true);
		EXPECT_EQ(ret, out.load());

		// only solutions are returned
		for (size_t i = 0; i < out.load(); ++i) {
			EXPECT_TRUE(is_solution(A, out[i], target, n));
			EXPECT_TRUE(out[i].label.is_equal(target, 0, n));
			right += 1;
		}
	}

	EXPECT_GT(right, 0);
	delete hml0;
	delete hml1;
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
