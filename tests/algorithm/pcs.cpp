#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>
#include <array>

#include "../../src/algorithm/int2weight.h"
#include "../../src/algorithm/pcs.h"
#include "../../src/algorithm/random_index.h"
#include "../../src/container/binary_packed_vector.h"
#include "../../src/container/kAry_type.h"
#include "../../src/element.h"
#include "../../src/helper.h"
#include "../../src/list/list.h"
#include "../../src/matrix/matrix.h"
#include "../../src/tree.h"
#include "math/gcd.h"

// needed for the generation of subset sum instances
#include "../../src/algorithm/rsa.h"
#include "../../src/algorithm/subsetsum.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

constexpr uint32_t n = 16;
constexpr uint32_t q = 1u << n;
using T  = kAry_Type_T<q>;
using TT = T::LimbType;



TEST(PCS, RhoFactorise) {
	constexpr static RSA_instance instance{21};
	rsa_pollard_rho<instance> rsa;

	EXPECT_TRUE(rsa.run());
}
TEST(PCS, RhoSubSetSumTree) {
	// example of a PCS computing trees within the function f
	constexpr static uint32_t n = 16;
	constexpr static uint32_t p = 1u << 16;
	using T = kAry_Type_T<p>;

	// first generate a random SubSetSum Instance
	T target, a = T(1734), b = T(319);

	// Needed Types
	using TT 			= uint16_t;
	using Matrix 		= FqMatrix<TT, 1, n, q>;
	using Value     	= FqNonPackedVector<n, 2, TT>;
	using Element		= Element_T<Value, T, Matrix>;
	using List			= List_T<Element>;
	using Tree			= Tree_T<List>;

	// represent the SubSetSum Instance a 1xn row matrix= row vector
	// this makes a lot of things alot easier
	Matrix A; A.random();

	std::vector<uint32_t> sol;
	generate_subsetsum_instance(target, sol, A, n);

	// enumerate weight n//4 on n//2 coordinates.
	// NOTE: elements are randomly chosen, so no chase sequence or whatsoever
	using Enumerator = BinaryRandomEnumerator<List, n/2, n/4>;
	Enumerator en{A};

	constexpr static size_t size = 1ull << (n / 4u);
	List L1{size}, L2{size}, out{size};
	Tree t{1, A, 0};

	// NOTE: the base lists are enumerated once, as the step function `f`
	// 	must be deterministic. The elements will have weight n//4 on each halve.
	en.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			  (&L1, &L2, n/4);

	// full solutions found by the tree within `f`
	size_t tree_solutions = 0;

	struct SubSetSumCmp {
		decltype(auto) operator()(const T &a1,
								  const T &a2,
								  const T &b1,
								  const T &b2) const {
			// get the lowest bit
			const uint32_t alb = a1.value() & 1;
			const uint32_t blb = b1.value() & 1;

			// and make sure, that they are different
			if (alb == blb) {
				return false;
			}

			return a2 == b2;
		}
	};

	// NOTE: Floyd's cycle finding returns the collision (the predecessors of
	// 	the cycle entry), or false if it is not golden (same lowest bits).
	// 	A golden collision is a solution: target + S1 = S2, so it is rare and
	// 	the number of walks is bounded.
	constexpr static PollardRhoConfig rhoConfig{.brent = false};
	const auto identity = [](const T &x) { return x; };
	for (uint32_t tries = 0; tries < 256; ++tries) {
		/// restart every X runs
		T a2, b2;
		const bool val = PollardRho<SubSetSumCmp, T, rhoConfig>::run(
		        [&A, &target, &L1, &L2, &out, &t, &tree_solutions](const T &in) {
			// the intermediate target: `in` remapped to a binary string of
			// weight n//4, plus the target if `in` is odd
			static uint32_t weights[n/4];
			int2weights<T::LimbType>(weights, in.value(), n, n/4, n/4);

			T iT = T(0);
			// super simple distinguish function
			if (in.value() & 1u) {
				iT += target;
			}

			for (uint32_t j = 0; j < n/4; j++) {
				iT += A[0][weights[j]];
			}

			// merge the two base lists on the full length ([0, n)) on the
			// intermediate target, every output with the target as label is
			// a full solution
			out.set_load(0);
			t.join2lists(out, L1, L2, iT, 0, n, false);
			for (size_t i = 0; i < out.load(); ++i) {
				T recalc;
				A.mul(recalc, out[i].value);
				tree_solutions += (recalc == target);
			}

			// the next element of the walk
			return iT;
		}, identity, a, b, a2, b2, 1u << 12);

		if (val) {
			// a golden collision: different lowest bits, same image
			EXPECT_NE(a.value() & 1u, b.value() & 1u);
			EXPECT_TRUE(a2 == b2);
			break;
		}

		// resample the flavour
		a = rng<T::LimbType>() % p;
		b = rng<T::LimbType>() % p;
	}
}

/// collision of f(x) = x^2 + 1 mod p, with both cycle finders of the
/// flavoured `run` (Brent and Floyd)
struct ModSquareCmp {
	bool operator()(const uint32_t &a1, const uint32_t &a2,
	                const uint32_t &b1, const uint32_t &b2) const noexcept {
		return (a2 == b2) && (a1 != b1);
	}
};

template<const PollardRhoConfig &cfg>
static void check_rho_config() {
	constexpr uint32_t p = 65521;
	const auto f = [](const uint32_t &x) { return uint32_t((uint64_t(x) * x + 1u) % p); };
	const auto flavour = [](const uint32_t &x) { return x; };
	for (uint32_t start = 2; start < 50; start += 7) {
		uint32_t x1 = start, y1 = start, x2 = 0, y2 = 0;
		const bool found = PollardRho<ModSquareCmp, uint32_t, cfg>::run(f, flavour, x1, y1, x2, y2);
		EXPECT_TRUE(found) << start;
		EXPECT_EQ(f(x1), f(y1)) << start;
		EXPECT_NE(x1, y1) << start;
	}
}

constexpr static PollardRhoConfig brentConfig{.brent = true};
constexpr static PollardRhoConfig floydConfig{.brent = false};
constexpr static PollardRhoConfig gosperConfig{.gosper = true};
TEST(PCS, RhoConfigBrent) { check_rho_config<brentConfig>(); }
TEST(PCS, RhoConfigFloyd) { check_rho_config<floydConfig>(); }
TEST(PCS, RhoConfigGosper) { check_rho_config<gosperConfig>(); }

/// distinguished: the lowest 4 bits are zero
struct ModSquareDistinguished {
	bool operator()(const uint32_t &a) const noexcept {
		return (a & 0xFu) == 0;
	}
};

/// parallel collision search of f(x) = x^2 + 1 mod p, sequential and parallel
TEST(PCS, ParallelCollisionSearch) {
	constexpr uint32_t p = 1048573;
	const auto f = [](const uint32_t &x) { return uint32_t((uint64_t(x) * x + 1u) % p); };
	using S = PCS<ModSquareCmp, ModSquareDistinguished, uint32_t>;

	for (const bool par : {false, true}) {
		uint32_t x = 0, y = 0;
		const bool found = S::run(par_if(par), f, x, y, 1u << 10u, 1u << 20u);
		EXPECT_TRUE(found) << par;
		EXPECT_NE(x, y) << par;
		EXPECT_EQ(f(x), f(y)) << par;
	}

	// with a flavour: the collision is one of x -> f(x ^ 1)
	const auto flavour = [](const uint32_t &x) { return x ^ 1u; };
	uint32_t x = 0, y = 0;
	const bool found = S::run(par_if(true), f, flavour, x, y, 1u << 10u, 1u << 20u);
	EXPECT_TRUE(found);
	EXPECT_NE(x, y);
	EXPECT_EQ(f(flavour(x)), f(flavour(y)));
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
