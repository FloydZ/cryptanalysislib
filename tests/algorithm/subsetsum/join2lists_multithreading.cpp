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


constexpr uint32_t n = 30u;
constexpr uint64_t q = (1ul << n);

using T 			= uint64_t;
using Value     	= BinaryVector<n>;
using Label    		= kAry_Type_T<q>;
using Matrix 		= FqVector<T, n, q, true>;
using Element		= Element_T<Value, Label, Matrix>;
using List			= List_T<Element>;
using Tree			= Tree_T<List>;

TEST(SubSetSum, constexpr_twolevel_streamjoin_on_iT_hashmap_v2_multithreaded) {
	constexpr uint32_t n = 16;
	constexpr uint32_t p = 2;
	constexpr uint32_t nthreads = 2;
	using Value  = BinaryVector<n>;
	using Label  = kAry_Type_T<1ull << n>;
	using Matrix = FqVector<T, n, 1ull << n, true>;
	using Element= Element_T<Value, Label, Matrix>;
	using List   = List_T<Element>;
	using Tree   = Tree_T<List>;
	constexpr uint32_t k_lower1=0, k_upper1=4;
	constexpr uint32_t k_lower2=4, k_upper2=10;

	Matrix A; A.random();
	constexpr size_t baselist_size = sum_bc(n/2, p);
	// NOTE: every thread streams its chunk of `l1`
	List out{1u<<10}, l1{baselist_size, nthreads}, l2{baselist_size};
	using Enumerator = BinaryLexicographicEnumerator<List, n/2, p>;
	Enumerator e{A};
	e.run(&l1, &l2, n/2);
	ASSERT_EQ(l1.load(), l1.size());

	using D = typename Label::DataType;
	using E = std::pair<size_t, size_t>;
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

	Label iT, target, t1;
	iT.random(0, 1ull << k_upper1);
	target.random(0, 1ull << k_upper2);
	Label::sub(t1, target, iT);

	Tree t{1, A, 0};
	t.template join2lists_on_iT_v2
			<k_lower1, k_upper1>
			(*hml1, l1, l2, *hml0, iT, true);

	std::atomic<size_t> calls_seq{0}, calls_par{0};
	auto f_seq = [&calls_seq](List &, Element &, Element &, size_t, size_t, size_t, size_t) {
		calls_seq += 1;
		return false;
	};
	auto f_par = [&calls_par](List &, Element &, Element &, size_t, size_t, size_t, size_t) {
		calls_par += 1;
		return false;
	};

	const size_t ret_seq = t.template twolevel_streamjoin_on_iT_hashmap_v2
			<k_lower1, k_upper1, k_lower2, k_upper2, nthreads>
			(par_if(false), out, *hml1, l1, l2, *hml0, target, t1, f_seq);
	const size_t ret_par = t.template twolevel_streamjoin_on_iT_hashmap_v2
			<k_lower1, k_upper1, k_lower2, k_upper2, nthreads>
			(par_if(true), out, *hml1, l1, l2, *hml0, target, t1, f_par);

	EXPECT_GT(ret_seq, 0);
	EXPECT_EQ(ret_seq, calls_seq.load());
	EXPECT_EQ(ret_par, ret_seq);
	EXPECT_EQ(calls_par.load(), ret_seq);

	delete hml0;
	delete hml1;
}

TEST(SubSetSum, constexpr_join2lists_on_iT_hashmap_v2_multithreaded) {
	Matrix A; A.random();
	//NOTE: thats 2**16 elements, dont forget that
	constexpr uint32_t k_lower=0, k_higher=16;
	constexpr uint32_t p = 4;
	constexpr uint32_t nthreads = 2;
	constexpr uint32_t chunks = 4;

	constexpr size_t baselist_size = sum_bc(n/2, p);
	List out{1u<<8, chunks}, l1{baselist_size, chunks}, l2{baselist_size, chunks};

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, 4>;
	Enumerator e{A};
	e.run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	Tree t{1, A, 0};
	const size_t found = t.join2lists_on_iT_v2
	    <k_lower, k_higher, 100, nthreads, chunks>
	    (par_if(true), out, l1, l2, target);


	auto right=true;
	int wrong=0;
	// NOTE: each chunk writes into its own block of `out`
	size_t out_load = 0;
	for (uint32_t tid = 0; tid < chunks; ++tid) {
	out_load += out.load(tid);
	for(uint64_t i = out.start_pos(tid); i < out.start_pos(tid) + out.load(tid); ++i) {
		Label test_recalc1(0), test_recalc2(0), test_recalc3(0);
		A.mul(test_recalc3, out[i].value);
		// NOTE: the full length
		for (uint64_t j = 0; j < n; ++j) {
			if (out[i].value.get(j)) {
				test_recalc1 += A[0][j];
				Label::add(test_recalc2, test_recalc2, A[0][j]);
			}
		}

		// NOTE that we do not recalculate the label
		EXPECT_EQ(true, test_recalc1.is_equal(test_recalc2, k_lower, k_higher));
		EXPECT_EQ(true, test_recalc1.is_equal(test_recalc3, k_lower, k_higher));
		EXPECT_EQ(true, test_recalc1.is_equal(out[i].label, k_lower, k_higher));

		if (!(Label::cmp(out[i].label, target, k_lower, k_higher))) {
			right = false;
			wrong++;
		}

		out[i].recalculate_label(A);
		EXPECT_EQ(true, test_recalc1.is_equal(out[i].label, k_lower, k_higher));
	}
	}


	Label el{};
	uint64_t num = 0;
	for (size_t i = 0; i < l1.load(); ++i) {
		for (size_t j = 0; j < l2.load(); ++j) {
			Label::add(el, l1[i].label, l2[j].label);
			if (el.is_equal(target, k_lower, k_higher)) {
				num += 1;
			}
		}
	}

	EXPECT_EQ(found, out_load);
	EXPECT_GT(out_load, 0);
	EXPECT_EQ(0, wrong);
	EXPECT_EQ(right, true);
	if constexpr (n == 16) {
		EXPECT_GT(out_load, 1u<<3);
		EXPECT_LT(out_load, 1u<<7);
	}
	EXPECT_EQ(out_load, num);
}

TEST(SubSetSum, join4lists_twolists_on_iT_v2_policy) {
	// NOTE: the solution has weight n/2 = 4*p, so it is the sum of four
	// 	elements with disjoint supports, which survives the random
	// 	intermediate target with probability ~13%
	constexpr uint32_t n = 16;
	constexpr uint32_t p = n/8;
	constexpr uint32_t nthreads = 2;
	using Value  = BinaryVector<n>;
	using Label  = kAry_Type_T<1ull << n>;
	using Matrix = FqVector<T, n, 1ull << n, true>;
	using List   = List_T<Element_T<Value, Label, Matrix>>;
	using Tree   = Tree_T<List>;
	constexpr uint32_t k_lower1=0, k_higher1=n/2;
	constexpr uint32_t k_lower2=n/2, k_higher2=n;

	Matrix A; A.random();
	constexpr size_t baselist_size = sum_bc(n/2, p);
	// NOTE: the threads work on the chunks of the lists (and write into
	// 	their chunk of the output), the sequential version needs lists
	// 	with a single chunk
	List out{1u<<10}, l1{baselist_size}, l2{baselist_size},
	     outp{1u<<10, nthreads}, l1p{baselist_size, nthreads}, l2p{baselist_size, nthreads};
	using Enumerator = BinaryLexicographicEnumerator<List, n/2, p>;
	Enumerator e{A};
	e.run(&l1, &l2, n/2);
	e.run(&l1p, &l2p, n/2);
	ASSERT_EQ(l1p.load(), l1p.size());
	ASSERT_EQ(l2p.load(), l2p.size());

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	Tree t{1, A, 0};
	for (const bool par : {false, true}) {
		uint32_t right = 0;
		for (uint32_t tries = 0; (tries < 128) && (right == 0); ++tries) {
			List &o = par ? outp : out;
			t.template join4lists_twolists_on_iT_v2
					<k_lower1, k_higher1, k_lower2, k_higher2, 100, nthreads>
					(par_if(par), o, par ? l1p : l1, par ? l2p : l2, target);

			for (uint32_t tid = 0; tid < o.threads(); ++tid) {
				for (size_t i = o.start_pos(tid); i < o.start_pos(tid) + o.load(tid); ++i) {
					EXPECT_TRUE(o[i].label.is_equal(target, k_lower1, k_higher2)) << par;
					Label recalc;
					A.mul(recalc, o[i].value);
					right += recalc.is_equal(target, 0, n) && (o[i].value.popcnt() == n/2);
				}
			}
		}

		EXPECT_GT(right, 0) << par;
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(nullptr));
	return RUN_ALL_TESTS();
}
