#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>
#include <array>

#include "algorithm/gcd.h"
#include "algorithm/int2weight.h"
#include "algorithm/pcs.h"
#include "algorithm/random_index.h"
#include "container/binary_packed_vector.h"
#include "container/kAry_type.h"
#include "element.h"
#include "helper.h"
#include "list/list.h"
#include "matrix/matrix.h"
#include "tree.h"

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
	decltype(auto) operator()(const T,
	                          const T &a2,
	                          const T,
	                          const T &b2) const {
		auto t = gcd<TT>((a2.value() + q - b2.value()) % q, q);
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
		in[i].random();
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
template<typename Matrix,
		typename T>
#if __cplusplus > 201709L
requires MatrixAble<Matrix>
#endif
void generate_random_subset_sum(Matrix &in,
								T &target,
                                std::vector<uint32_t> &sol,
								const uint32_t weight) noexcept {
	ASSERT(in.rows() == 1);
	ASSERT(in.cols() >= weight);

	std::vector<uint32_t> weights(weight);
	generate_random_indices(weights, in.cols());
	sol.insert(sol.end(), weights.begin(), weights.end());

	target = T(0);
	for (auto &w: weights) {
		target += in[0][w];
	}
}
/// generate a random row major matrix of dimension 1xn
/// \tparam Matrix
/// \param in
/// \param target
/// \param weight total weight = sum of both mitm sides
template<typename Matrix,
         typename T>
#if __cplusplus > 201709L
	requires MatrixAble<Matrix>
#endif
void generate_random_subset_sum_mitm_split(const Matrix &in,
                                           T &target,
                                           std::vector<uint32_t> &sol,
										   const uint32_t weight) noexcept {
	ASSERT(in.rows() == 1);
	ASSERT(in.cols() >= weight);
	ASSERT((weight & 1u) == 0);

	const uint32_t weight2 = weight/2u;
	const uint32_t n = in.cols();
	std::vector<uint32_t> weights(weight2);
	generate_random_indices(weights, n/2);
	sol.insert(sol.end(), weights.begin(), weights.end());

	target = T(0);
	for (auto &w: weights) {
		target += in[0][w];
	}

	generate_random_indices(weights, n/2);
	for (uint32_t i = 0; i < weight2; ++i) {
		weights[i] += n/2;
	}
	sol.insert(sol.end(), weights.begin(), weights.end());

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
	/// As a function distinguished we use the first bit of a number
	constexpr static uint32_t n = 16;
	constexpr static uint32_t p = 8209; //16411; //32771;// 1u << (n);
	constexpr static size_t walk_len = 1u<<3u;

	using T 		= uint64_t;
	using Label    	= kAry_Type_T<p>;
	using Value    	= BinaryContainer<n>;
	using Matrix 	= FqVector<T, n, p, true>;
	using Element 	= Element_T<Value, Label, Matrix>;

	// init values
	Label target(0);
	Element a,b;
	a.random(); b.random();
	Matrix instance; instance.random();
	std::cout << instance;

	std::vector<uint32_t> sol;
	generate_random_subset_sum_mitm_split(instance, target, sol, n/2);
	//generate_random_subset_sum(instance, target, sol, n/2);
	std::sort(sol.begin(), sol.end());

	struct SubSetSumCmp {
		///
		/// \param a1
		/// \param a2
		/// \param b1
		/// \param b2
		/// \return
		auto operator()(const Element &a1,
		                          const Element &a2,
		                          const Element &b1,
		                          const Element &b2) const {
			// get the lowest bit
			const uint32_t alb = a1.label.value() & 1;
			const uint32_t blb = b1.label.value() & 1;

			// and make sure, that they are different
			if (alb == blb) {
				return false;
			}

			return a2.label == b2.label;
		}
	};

	auto f = [&instance,
	          &target = std::as_const(target)]
		(const Element &_in) {
		Label in = _in.label;
		const uint32_t n2 = n/2, n4 = n/4;
		static uint32_t weights[n2];
		// constexpr uint64_t mask = (1ull << (n/2)) - 1ull;
		int2weights<Label::LimbType>(weights, in.value(), n, n2, n2);
		Element ret;

		// super simple distinguish function
		if (in.value() & 1u) {
			ret.label = target;
			for (uint32_t i = 0; i < n4; i++) {
				ret.label -= instance[0][weights[i + n4]];
				ret.value[weights[i + n4]] = true;
			}
		} else {
			for (uint32_t i = 0; i < n4; i++) {
				ret.label += instance[0][weights[i]];
				ret.value[weights[i]] = true;
			}
		}

		ASSERT(ret.value.popcnt() == n4);
		return ret;
	};

	while(true) {
		/// restart every X runs
		const bool found = PollardRho<SubSetSumCmp, Element>::run
		        (f, a, b, walk_len);
		if (found) {
			break;
		}

		// resample the flavour
		a.label.random();
		b.label.random();
	}

	a = f(a);
	b = f(b);
	Label c(0);

	for (const auto &s : sol) {
		std::cout << s << " ";
	}
	std::cout << std::endl;
	std::cout << a;
	std::cout << b;

	// reconstruct the solution
	for (uint32_t i = 0; i < n; ++i) {
		if (a.value.get_bit_shifted(i)) {
			//Label::add(c, c, instance[0][i]);
			c += instance[0][i];
			std::cout << i << " ";
		}
		if (b.value.get_bit_shifted(i)) {
			// Label::add(c, c, instance[0][i]);
			c += instance[0][i];
			std::cout << i << " ";
		}
	}
	std::cout << std::endl;

	Label c1(0);
	for (const auto &s : sol) {
		// std::cout << s << std::endl;
		// TODO do not produce the same result: add, +=
		c1 += instance[0][s];
		//Label::add(c1, c1, instance[0][s]);
	}


	Label c2(0); Value v2;
	Value::add(v2, a.value, b.value);
	instance.mul(c2, v2);
	EXPECT_EQ(true, c1.is_equal(c2));

	std::cout << c1 << std::endl;
	std::cout << c << std::endl;
	std::cout << target << std::endl;
	const bool correct = c == target;
	EXPECT_EQ(correct, true);
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
	Matrix A; A.random();

	std::vector<uint32_t> sol;
	generate_random_subset_sum(A, target, sol, n/2);

	// enumerate weight n//4 on n//2 coordinates.
	// NOTE: elements are randomly chosen, so no chase sequence or whatsoever
	using Enumerator = BinaryRandomEnumerator<List, n/2, n/4>;
	Enumerator en{A};

	constexpr static size_t size = 1ull << (n / 4u);
	List L1{size}, L2{size}, out{size};

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
				// const auto el = out[i];

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
