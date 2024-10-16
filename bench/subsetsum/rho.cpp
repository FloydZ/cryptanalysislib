#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "algorithm/subsetsum.h"
#include "container/kAry_type.h"
#include "helper.h"
#include "matrix/matrix.h"
#include "tree.h"

#include "params.h"

int main() {
	constexpr uint32_t n = PARAM_n;
	constexpr uint64_t q = 1ul << n;
	constexpr static SSS instance{.n=n, .q=q, .bp=PARAM_n1_3};
	using S = sss_d2<instance>;

	using Value  = S::Value;
	using Label  = S::Label;
	using Matrix = S::Matrix;
	using Element= S::Element;
	using List   = S::List;
	using Tree   = S::Tree;

	Matrix A; A.random();
	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	S s(A, target);
	s.run();
}
