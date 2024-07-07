#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>
#include <array>

#include "algorithm/gcd.h"
#include "algorithm/int2weight.h"
#include "algorithm/random_index.h"
#include "container/kAry_type.h"
#include "container/binary_packed_vector.h"
#include "matrix/matrix.h"
#include "element.h"
#include "list/list.h"
#include "tree.h"
#include "helper.h"
#include "pcs.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// max n = 15
constexpr uint32_t q = 8051;
using T  = kAry_Type_T<q>;
using TT = T::LimbType;

struct RSACmp {
	constexpr static T one{1};
	constexpr static T nn{q};
	decltype(auto) operator()(T &a, T &b) const {
		auto t = gcd<TT>((a.value() + q - b.value()) % q, q);
		return t > 1;
	}
};

/// TODO move somewhere useful
/// generates a new random subset sum instance
template<typename T>
void generate_random_subset_sum(std::vector<T> &in,
                                T &target,
                                const uint32_t weight) noexcept {
	if (in.size() == 0) {
		return;
	}

	if (weight > in.size()) {
		return;
	}

   	for (size_t i = 0; i < in.size(); i++) {
		in[i] = T::random();
	}

	std::vector<uint32_t> weights(weight);
	generate_random_indices(weights, in.size());

	target = T(0);
	for (auto &w: weights) {
		target += in[w];
	}
}

/// generate a random row major matrix of dimension 1xn
/// \tparam Matrix
/// \param in
/// \param target
/// \param weight
template<typename Matrix, typename T>
void generate_random_subset_sum(Matrix &in, T &target,
								const uint32_t weight) noexcept {
	ASSERT(in.rows() == 1);
	ASSERT(in.cols() >= weight);

	std::vector<uint32_t> weights(weight);
	generate_random_indices(weights, in.cols());

	target = T(0);
	for (auto &w: weights) {
		target += in[0][w];
	}
}

TEST(PCS, RhoFactorise) {
	T a, b;
	a.set(5,0);
	b.set(26,0);
	PollardRho<RSACmp, T>::run([](const T &in){
		const auto t = in*in + T(1);
		return t;
	}, a, b);

	EXPECT_EQ(a.value(), 26);
	EXPECT_EQ(b.value(), 2839);
}

TEST(PCS, RhoSubSetSum) {
	/// Simple example of how to use the PollardRho class
	/// to solve a subset sum problem.
	/// As a function distinuisher we use the first bit a number
	constexpr static uint32_t n = 16;
	constexpr static uint32_t p = 2999;
	using T = kAry_Type_T<p>;

	// init values
	T target, a = T(1734), b = T(319);
	std::vector<T> instance((size_t) n);
	generate_random_subset_sum(instance, target, n/2);

	struct SubSetSumCmp {
		decltype(auto) operator()(const T &a,
		                          const T &b) const {
			return a == b;
		}
	};

	while(true) {
		/// restart every X runs
		const bool val = PollardRho<SubSetSumCmp, T>::run([&instance, &target](const T &in) {
			static uint32_t weights[n/4];
			int2weights<T::LimbType>(weights, in.value(), n, n/4, n/4);
			T ret = T(0);

			// super simple distinguish function
			if (in.value() & 1u) {
				ret += target;
			}

			for (uint32_t i = 0; i < n/4; i++) {
				ret += instance[weights[i]];
			}

			return ret;
		}, a, b, 1u << 6);

		if (val) {
			break;
		}

		// resample the flavour
		a = fastrandombytes_T<T::LimbType>() % p;
		b = fastrandombytes_T<T::LimbType>() % p;
	}
}

TEST(PCS, RhoSubSetSumTree) {
	// example of a PCS computing trees within the function f
	constexpr static uint32_t n = 16;
	constexpr static uint32_t p = 2999;
	using T = kAry_Type_T<p>;

	// first generate a random SubSetSum Instance
	T target, a = T(1734), b = T(319);

	// Needed Types
	using TT 			= uint16_t;
	using Matrix 		= FqMatrix<TT, 1, n, q>;
	using Value     	= kAryContainer_T<TT, n, 2>;
	using Element		= Element_T<Value, T, Matrix>;
	using List			= List_T<Element>;
	using Tree			= Tree_T<List>;

	// represent the SubSetSum Instance a 1xn row matrix= row vector
	// this makes a lot of things alot easier
	Matrix A;
	A.random();
	generate_random_subset_sum(A, target, n/2);

	// enumerate weight n//4 on n//2 coordinates.
	// NOTE: elements are randomly chosen, so no chase sequence or what so ever
	using Enumerator = RandomEnumerator<List, n/2, n/4>;
	Enumerator en{A};

	constexpr static size_t size = 1ull << (n / 4u);
	List L1{size}, L2{size}, out{size};

	struct SubSetSumCmp {
		decltype(auto) operator()(const T &a,
								  const T &b) const {
			return a == b;
		}
	};

	while(true) {
		/// restart every X runs
		const bool val = PollardRho<SubSetSumCmp, T>::run(
		        [&A, &target, &L1, &L2, &out, &en](const T &in) {
			// reset the lists
			L1.set_load(0); L2.set_load(0), out.set_load(0);

			// write random elements in the lists in a mitm split. The elements will have
			// weight n//4 on each halve.
			en.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
					  (&L1, &L2, n/4);

			// merge the two base lists on the full length ([0, n)) on the target
			Tree::join2lists(out, L1, L2, target, 0, n, false);

			T ret = T(0);
			for (size_t i = 0; i < out.load(); ++i) {
				// TODO do something
				const auto el = out[i];

				// remap each solution back to a binary string of weight n //4
				static uint32_t weights[n/4];
				int2weights<T::LimbType>(weights, in.value(), n, n/4, n/4);

				// super simple distinguish function
				if (in.value() & 1u) {
					ret += target;
				}

				for (uint32_t j = 0; j < n/4; j++) {
					ret += A[0][weights[i]];
				}

			}

			// TODO return something
			return ret;
		}, a, b, 1u << 6);

		if (val) {
			break;
		}

		// resample the flavour
		a = fastrandombytes_T<T::LimbType>() % p;
		b = fastrandombytes_T<T::LimbType>() % p;
	}
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
