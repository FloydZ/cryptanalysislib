#include <gtest/gtest.h>
#include <cstdint>

#define TEST_BASE_LIST_SIZE 100
#define TEST_BASE_LIST_ADDITIONAL_SIZE TEST_BASE_LIST_SIZE/10

#include "helper.h"
#include "list.h"
#include "binary.h"
#include "../test.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;



///
/// \param l
/// \return the number of elements which are duplicate
uint64_t helper_list_is_every_element_value_unique(const List &l){
	if (l.get_load() == 0)
		return 0;

	uint64_t errors = 0;
	const uint64_t vs = l[0].value_size();
	for (int i = 0; i < l.get_load(); ++i) {
		for (int j = 0; j < l.get_load(); ++j) {
			// skip element if their are the same
			if (i == j)
				continue;

			bool equal = true;
			for (int k = 0; k < vs; ++k) {
				if (l[i].get_value().data()[k] != l[j].get_value().data()[k]) {
					equal = false;
					break;
				}
			}

			if (equal){
				errors += 1;
			}

		}
	}

	// return only the half amount of errors, because we count every duplicate twice.
	return errors >> uint64_t(1);
}

uint64_t helper_check_weight_of_value(const List &l, const uint64_t e1, const uint64_t em1) {
	typedef typename Element::ValueDataType T;

	if (l.get_load() == 0)
		return 0;

	uint64_t errors = 0;
	const uint64_t vs = l[0].value_size();
	for (int i = 0; i < l.get_load(); ++i) {
		uint64_t counted_ones  = 0;
		uint64_t counted_mones = 0;

		for (int j = 0; j < vs; ++j) {
			if (l[i].get_value().data()[j] == T(-1))
				counted_mones += 1;
			if (l[i].get_value().data()[j] == 1)
				counted_ones += 1;
		}

		if ((counted_mones != em1) || (counted_ones != e1)) {
			errors += 1;
			// std::cout << "Not correct weight: " << l[i] << " pos:" << i << "\n";
		}
	}
	
	return errors;
}


TEST(SearchBinary, Simple) {
	uint64_t bpos, nbpos;
	BinaryList L{0};
	mzd_t *A_ = mzd_init(n, n);
	Matrix_T<mzd_t *> A((mzd_t *)A_);
	A.gen_identity(n);

	L.generate_base_random(TEST_BASE_LIST_SIZE, A);

	for (uint64_t k_lower = 0; k_lower < n; ++k_lower) {
		for (uint64_t k_upper = k_lower+5; k_upper < k_lower + 6; ++k_upper) {
			if ((k_lower%64) + 6 >= 64)
				continue;

			for (uint64_t pos = 0; pos < 1; ++pos) {
				BinaryElement e; e.random(A);
				L[pos] = e;

				// first sort it
				L.sort_level(k_lower, k_upper);

				// the do different indepentend searches
				bpos  = L.search_level_binary_simple(e, k_lower, k_upper);
				nbpos = L.search_level(e, k_lower, k_upper);
				EXPECT_EQ(bpos, nbpos);
			}
		}
	}
}

TEST(SearchBinary, Complex) {
	uint64_t bpos, nbpos;
	BinaryList L{0};
	mzd_t *A_ = mzd_init(n, n);
	Matrix_T<mzd_t *> A((mzd_t *)A_);
	A.gen_identity(n);

	L.generate_base_random(TEST_BASE_LIST_SIZE, A);

	for (uint64_t k_lower = 0; k_lower < n; ++k_lower) {
		for (uint64_t k_upper = k_lower+1; k_upper < n; ++k_upper) {
			for (uint64_t pos = 0; pos < 1; ++pos) {
				BinaryElement e; e.random(A);
				L[pos] = e;

				// first sort it
				L.sort_level(k_lower, k_upper);

				// the do different indepentend searches
				bpos  = L.search_level_binary(e, k_lower, k_upper);
				nbpos = L.search_level(e, k_lower, k_upper);
				EXPECT_EQ(bpos, nbpos);
			}
		}
	}
}


TEST(SearchBinaryCustom, Simple) {
	uint64_t bpos, nbpos;
	BinaryList L{0};
	mzd_t *A_ = mzd_init(n, n);
	Matrix_T<mzd_t *> A((mzd_t *)A_);
	A.gen_identity(n);

	L.generate_base_random(TEST_BASE_LIST_SIZE, A);

	for (uint64_t k_lower = 0; k_lower < 10; ++k_lower) {
		for (uint64_t k_upper = k_lower+5; k_upper < k_lower+6; ++k_upper) {
			for (uint64_t pos = 0; pos < 1; ++pos) {
				BinaryElement e; e.random(A);
				L[pos] = e;

				// first sort it
				L.sort_level(k_lower, k_upper);

				// the do different indepentend searches
				bpos  = L.search_level_binary_custom_simple(e, k_lower, k_upper);
				nbpos = L.search_level(e, k_lower, k_upper);
				EXPECT_EQ(bpos, nbpos);
			}
		}
	}
}

/* unused and not important
TEST(SearchInterpolated, Simple) {
	uint64_t bpos, nbpos;
	BinaryList L{0};
	mzd_t *A_ = mzd_init(n, n);
	Matrix_T<mzd_t *> A((mzd_t *)A_);
	A.gen_identity(n);

	L.generate_base_random(TEST_BASE_LIST_SIZE, A);

	for (uint64_t k_lower = 0; k_lower < n-10; ++k_lower) {
		for (uint64_t k_upper = k_lower+5; k_upper < k_lower+6; ++k_upper) {
			for (uint64_t pos = 0; pos < 1; ++pos) {
				BinaryElement e; e.random(A);
				L[pos] = e;

				// first sort it
				L.sort_level(k_lower, k_upper);

				// the do different indepentend searches
				bpos  = L.search_level_interpolated_simple(e, k_lower, k_upper);
				nbpos = L.search_level(e, k_lower, k_upper);
				EXPECT_EQ(bpos, nbpos);

				if (nbpos != bpos) {
					for (int i = nbpos; i <= bpos; ++i) {
						std::cout << L[i];
					}
				}
			}
		}
	}
}
*/

#if n > 256
// otherwise is this test not make any sense.
TEST(SortLevelExt, Simple) {
	uint64_t bpos, nbpos;
	BinaryList L{0};
	mzd_t *A_ = mzd_init(n, n);
	Matrix_T<mzd_t *> A((mzd_t *)A_);
	A.gen_identity(n);


	for (uint64_t k_lower = 0; k_lower < n; k_lower += 110) {
		for (uint64_t k_upper = k_lower+128; k_upper < n; k_upper += 90) {
			L.resize(0);
			L.generate_base_random(TEST_BASE_LIST_SIZE, A);


			for (uint64_t pos = 0; pos < 1; ++pos) {
				BinaryElement e; e.random(A);
				L[pos] = e;

				// first sort it
				L.sort_level_ext(k_lower, k_upper);
				EXPECT_EQ(true, L.is_sorted(k_lower, k_upper));
			}
		}
	}
}
#endif

TEST(SortLevelSimple, Simple) {
	BinaryList L{0};
	mzd_t *A_ = mzd_init(n, n);
	Matrix_T<mzd_t *> A((mzd_t *)A_);
	A.gen_identity(n);


	for (uint64_t k_lower = 0; k_lower < n; k_lower += 64) {
		for (uint64_t k_upper = k_lower+1; k_upper < k_lower + 64 && k_upper < n; k_upper += 10) {
			L.resize(0);
			L.generate_base_random(TEST_BASE_LIST_SIZE, A);


			for (uint64_t pos = 0; pos < 1; ++pos) {
				BinaryElement e; e.random(A);
				L[pos] = e;

				// first sort it
				L.sort_level_sim(k_lower, k_upper);
				EXPECT_EQ(true, L.is_sorted(k_lower, k_upper));
			}
		}
	}
}

#ifndef EXTERNAL_MAIN
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
#endif
