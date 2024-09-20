#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/random_index.h"
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
//using Value     	= kAryContainer_T<T, n, 2>;
using Value     	= BinaryContainer<n>;
using Label    		= kAry_Type_T<q>;
using Matrix 		= FqVector<T, n, q, true>;
using Element		= Element_T<Value, Label, Matrix>;
using List			= List_T<Element>;
using Tree			= Tree_T<List>;

// unused ignore
static std::vector<std::vector<uint8_t>> __level_filter_array{{ {{4,0,0}}, {{1,0,0}}, {{1,0,0}}, {{0,0,0}} }};

// NOTE: random enumeration of the values
TEST(SubSetSum, Simple) {
	// even it says matrix. It is a simple row vector
	Matrix A;
	A.fill(0);

	// // log size
	constexpr size_t list_size = 10;
	static std::vector<uint64_t> tbl{{0, n}};
	Tree t{1, A, list_size, tbl, __level_filter_array};

	t[0].random(1u << list_size, A);
	t[1].random(1u << list_size, A);
	t.join_stream(0);


	EXPECT_EQ(1u << 20u, t[2].load());
}

TEST(SubSetSum, JoinForLevelTwoPreparelists) {
	const uint32_t d = 2;
	static std::vector<uint64_t> tbl{{0, 5, 10, n}};
	Matrix A; A.random();

	Label target; target.zero();
	std::vector<uint32_t> weights(n/2);
	generate_random_indices(weights, n);
	for (uint32_t i = 0; i < n/2; ++i) {
		Label::add(target, target, A[0][weights[i]]);
	}

	auto intermediat_level_limit = [](const uint32_t i) {
		return (1ULL << (d - i - 1ull)) - 1ull;
	};

	// generate the intermediate targets for all levels
	std::vector<std::vector<Label>> intermediate_targets(d);
	for (uint32_t i = 0; i < d; ++i) {
		const uint32_t limit = intermediat_level_limit(i);
		// +1 to have enough space.
		intermediate_targets[i].resize(limit + 1);

		// set random intermediate targets
		for (uint32_t j = 0; j < limit; ++j) {
			intermediate_targets[i][j].random();
		}

		// the last intermediate target is always the full target
		intermediate_targets[i][limit] = target;
	}

	// adjust last intermediate target (which was the full target) of each level,
	// such that targets of each level add up the true target.
	for (uint32_t i = 0; i < d - 1; ++i) {
		uint64_t k_lower, k_higher;
		const uint32_t limit = intermediat_level_limit(i);
		for (uint32_t j = 0; j < limit; ++j) {
			translate_level(&k_lower, &k_higher, -1, tbl);

			// intermediate_targets[i][limit] = true target
			Label::sub(intermediate_targets[i][limit],
					   intermediate_targets[i][limit],
					   intermediate_targets[i][j], k_lower, k_higher);
		}
	}

	Tree t{2, A, 10, tbl, __level_filter_array};
	t[0].random(1u << 8u, A);
	t[1].random(1u << 8u, A);

	t.prepare_lists(0, intermediate_targets);
	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);
	t.join_stream(0);

	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);
	t[2].sort_level(1, tbl);
	t.prepare_lists(1, intermediate_targets);
	t.join_stream(1);
	EXPECT_GT(t[3].load(), 0);

	for (size_t i = 0; i < t[3].load(); ++i) {
		t[3][i].recalculate_label(A);
	}
	std::cout << target << std::endl;
	std::cout << t[3];
}

TEST(SubSetSum, JoinForLevelTwo) {
	Matrix A;
	A.fill(0);

	static std::vector<uint64_t> tbl{{0, 5, 10, n}};
	Tree t{2, A, 10, tbl, __level_filter_array};

	t[0].random(1u << 2u, A);
	t[1].random(1u << 2u, A);
	t.join_stream(0);
	t.join_stream(1);
	EXPECT_EQ(1u << 8u, t[3].load());
}

TEST(SubSetSum, JoinForLevelThree) {
	Matrix A;
	A.fill(0);
	static std::vector<uint64_t> tbl{{0, 5, 10, n}};
	Tree t{3, A, 10, tbl, __level_filter_array};

	t[0].random(1u << 2u, A);
	t[1].random(1u << 2u, A);
	t.join_stream(0);
	t.join_stream(1);
	t.join_stream(2);
	EXPECT_EQ(1u << 16u, t[4].load());
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
