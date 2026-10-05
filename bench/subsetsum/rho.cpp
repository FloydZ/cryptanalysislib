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
	// constexpr static SSS instance{.n=n, .q=q, .bp=PARAM_n1_3, .l1=PARAM_l1, .l2=PARAM_l2, .walk_len=128, .flavour_q=1021,};
	// TODO currently the python optimizer outputs wrong values
	constexpr static SSS instance{.n=n, .q=q, .bp=2, .l1=10, .l2=6, .walk_len=128, .flavour_q=1021,};
	rng_seed();

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
