#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/random_index.h"
#include "algorithm/subsetsum.h"

#include "algorithm/subsetsum.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// TODO: simplify the enumerator interface to only need the lists as inputs and not the offset.

// NOTE: random enumeration of the values
// NOTE: only two baselists are used
TEST(SubSetSum, n32_d2_baselists2) {
	// 15.841070074033095 7.999094705692274 6 10 0 0 100.0 2.0 0 120.0 14.06  1.77
	constexpr uint32_t n = 32ul;
	constexpr uint64_t q = (1ul << n); //4294967279

	using T 			= uint64_t;
	using Value     	= BinaryVector<n>;
	using Label    		= kAry_Type_T<q>;
	using Matrix 		= FqVector<T, n, q>;
	using Element		= Element_T<Value, Label, Matrix>;
	using List			= List_T<Element>;
	using Tree			= Tree_T<List>;

	Matrix A; A.random();
	// NOTE: the outputs are filtered on weight n/2, so the four base list
	// 	elements (weight <= p each) need 4p >= n/2. `k_higher2` is chosen
	// 	such that the intermediate hashmap has enough buckets.
	constexpr uint32_t k_lower1=0, k_higher1=10, k_higher2=20;
	constexpr uint32_t p = 4;
	constexpr size_t baselist_size = sum_bc(n/2, p);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size};

	using Enumerator = BinaryListEnumerateMultiFullLength<List, n/2, p>;
	//using Enumerator = BinaryLexicographicEnumerator<List, n/2, p>;

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

    Tree t{1, A, 0};
	t.template join4lists_twolists_on_iT_hashmap_v2
			<k_lower1, k_higher1, k_higher1, k_higher2, Enumerator>
			(out, l1, l2, target);

	EXPECT_GT(out.load(), 0);
}

TEST(SubSetSum, n32_d2_rho) {
	constexpr uint32_t n = 32;
	constexpr uint64_t q = 1ul << n;
	constexpr static SSS instance{.n=n, .q=q, .bp=2, .l1=10, .l2=6, .walk_len=128, .flavour_q=1021};
	using S = sss_d2<instance>;
	using Label  = S::Label;
	using Matrix = S::Matrix;

	Matrix A; A.random();
	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n, true, false);

	S s(A, target);
	s.run();
}

// not correct right now: the runtime is unbounded (0.1s to >500s), a walk of
// 	`walk_len`=8 steps only rarely ends in a golden collision.
//TEST(PCS, RhoSubSetSum) {
//	/// Simple example of how to use the PollardRho class
//	/// to solve a subset sum problem.
//	/// As a function distinguished we use the first bit of a number
//	constexpr static uint32_t n = 16;
//	constexpr static uint32_t p = 1u<<n;
//	constexpr static size_t walk_len = 1u<<3u;
//
//	using T 		= uint64_t;
//	using Label    	= kAry_Type_T<p>;
//	using Value    	= BinaryVector<n>;
//	using Matrix 	= FqVector<T, n, p, true>;
//	using Element 	= Element_T<Value, Label, Matrix>;
//
//	// init values
//	Label target(0);
//	Element a,b;
//	a.random(); b.random();
//	Matrix instance; instance.random();
//
//	std::vector<uint32_t> sol;
//	generate_subsetsum_instance(target, sol, instance, n);
//
//	struct SubSetSumCmp {
//		/// simple comparison struct
//		/// only a2 and b2 are compared for equality
//		/// \param a1 predecessor of a2
//		/// \param a2 value to be
//		/// \param b1 predecessor of b2
//		/// \param b2 value to be compared
//		/// \return true if a2==b2, and a1!=b1;
//		auto operator()(const Element &a1,
//						const Element &a2,
//						const Element &b1,
//						const Element &b2) const noexcept {
//			// get the lowest bit
//			const uint32_t alb = a1.label.value() & 1;
//			const uint32_t blb = b1.label.value() & 1;
//
//			// and make sure, that they are different
//			if (alb == blb) {
//				return false;
//			}
//
//			return a2.label == b2.label;
//		}
//	};
//
//	auto f = [&instance,
//			  &target = std::as_const(target)]
//		(const Element &_in) {
//		Label in = _in.label;
//		const uint32_t n2 = n/2, n4 = n/4;
//		static uint32_t weights[n2];
//		// constexpr uint64_t mask = (1ull << (n/2)) - 1ull;
//		int2weights<Label::LimbType>(weights, in.value(), n, n2, n2);
//		Element ret;
//
//		// super simple distinguish function
//		if (in.value() & 1u) {
//			ret.label = target;
//			for (uint32_t i = 0; i < n4; i++) {
//				ret.label -= instance[0][weights[i + n4]];
//				ret.value[weights[i + n4]] = true;
//			}
//		} else {
//			for (uint32_t i = 0; i < n4; i++) {
//				ret.label += instance[0][weights[i]];
//				ret.value[weights[i]] = true;
//			}
//		}
//
//		assert(ret.value.popcnt() == n4);
//		return ret;
//	};
//
//	while(true) {
//		/// restart every X runs
//		const bool found = PollardRho<SubSetSumCmp, Element>::run
//				(f, a, b, walk_len);
//		if (found) {
//			break;
//		}
//
//		// resample the flavour
//		a.label.random();
//		b.label.random();
//	}
//
//	a = f(a); b = f(b);
//	Label c(0);
//
//	std::cout << "solution: ";
//	for (const auto &s : sol) {
//		std::cout << s << " ";
//	}
//	std::cout << std::endl;
//	std::cout << a << ", a" << std::endl;
//	std::cout << b << ", b" << std::endl;
//
//	// reconstruct the solution
//	for (uint32_t i = 0; i < n; ++i) {
//		if (a.value.get_bit_shifted(i)) {
//			c += instance[0][i];
//			std::cout << i << " ";
//		}
//		if (b.value.get_bit_shifted(i)) {
//			c += instance[0][i];
//			std::cout << i << " ";
//		}
//	}
//	std::cout << std::endl;
//
//	Label c1(0);
//	for (const auto &s : sol) {
//		c1 += instance[0][s];
//	}
//
//
//	Label c2(0); Value v2;
//	Value::add(v2, a.value, b.value);
//	instance.mul(c2, v2);
//	EXPECT_EQ(true, c1.is_equal(c2));
//
//	std::cout << c1 << std::endl;
//	std::cout << c << std::endl;
//	std::cout << target << std::endl;
//	const bool correct = c == target;
//	EXPECT_EQ(correct, true);
//
//}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
