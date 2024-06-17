#include <gtest/gtest.h>
#include <cstdint>

#define TEST_BASE_LIST_SIZE (1u << 18u)
#define TEST_BASE_LIST_ADDITIONAL_SIZE TEST_BASE_LIST_SIZE/10

#include "../test.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using Matrix 	= FqMatrix<T, n, k, q>;
using Element 	= Element_T<Value, Label, Matrix>;
using List 		= List_T<Element>;

///
/// \param l
/// \return the number of elements which are duplicate
uint64_t helper_list_is_every_element_value_unique(const List &l){
	if (l.load() == 0)
		return 0;

	uint64_t errors = 0;
	const uint64_t vs = l[0].value_size();
	for (uint64_t i = 0; i < l.load(); ++i) {
		for (uint64_t j = 0; j < l.load(); ++j) {
			// skip element if their are the same
			if (i == j)
				continue;

			bool equal = true;
			for (uint64_t k = 0; k < vs; ++k) {
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
	if (l.load() == 0)
		return 0;

	uint64_t errors = 0;
	const uint64_t vs = l[0].value_size();
	for (size_t i = 0; i < l.load(); ++i) {
		uint64_t counted_ones  = 0;
		uint64_t counted_mones = 0;

		for (size_t j = 0; j < vs; ++j) {
			if (l[i].value.ptr()[j] == T(-1ull))
				counted_mones += 1;
			if (l[i].value.ptr()[j] == T(1ul))
				counted_ones += 1;
		}

		if ((counted_mones != em1) || (counted_ones != e1)) {
			errors += 1;
			// std::cout << "Not correct weight: " << l[i] << " pos:" << i << "\n";
		}
	}
	
	return errors;
}

TEST(ListIntTest, DoesNotLeak) {
    List l {1};
}

TEST(ListIntTest, CreateEmptyBaseLists) {
	List l{0};
	Matrix mm{};
	mm.zero();
	Element zero{};
	zero.zero();

	l.generate_base_random(TEST_BASE_LIST_SIZE, mm);
	EXPECT_EQ(TEST_BASE_LIST_SIZE,  l.size());

	for (uint64_t i = 0; i < TEST_BASE_LIST_SIZE; ++i) {
		// zero means on all coordinates of the Label
		EXPECT_EQ(true,  zero.is_equal(l[i]));
	}
}

TEST(SearchBoundaries, BasicLevel0) {
	// simple checks:
	//      -
	List l{0};
	Matrix mm{};
	mm.random();
	l.generate_base_random(TEST_BASE_LIST_SIZE, mm);

	// debug helper.
	// std::cout << l;

	Element zero{};
	zero.zero();

	// nothing should be found.
	auto r = l.search_boundaries(zero, 0, n);
	EXPECT_EQ(r.second,  r.first);
	EXPECT_EQ(TEST_BASE_LIST_SIZE,  r.second);
}

TEST(SearchBoundaries, EndLevel0) {
	List l{0};
	Matrix mm{};
	mm.random();

	l.generate_base_random(TEST_BASE_LIST_SIZE, mm);
	Element zero{};
	zero.zero();

	l[TEST_BASE_LIST_SIZE - 1] = zero;    // add the zero element

	auto r = l.search_boundaries(zero, 0, n);
	EXPECT_NE(-1,  r.first);    // sanity check so we dont get any seg faults
	EXPECT_NE(-1,  r.second);

	EXPECT_EQ(TEST_BASE_LIST_SIZE-1,  r.first);
	EXPECT_EQ(TEST_BASE_LIST_SIZE,  r.second);
}

TEST(SearchBoundaries, End2Level0) {
	const uint64_t add_size = TEST_BASE_LIST_ADDITIONAL_SIZE;
	List l{0};
	Matrix mm{};
	mm.random();

	l.generate_base_random(TEST_BASE_LIST_SIZE, mm);

	Element zero{};
	zero.zero();

	// some arbitrary amount
	for (uint64_t i = 0; i < add_size; ++i) {
		l[TEST_BASE_LIST_SIZE - 1 - i] = zero;
	}

	auto r = l.search_boundaries(zero, 0, n);
	EXPECT_NE(r.second,  r.first);    // sanity check so we dont get any seg faults

	EXPECT_EQ(TEST_BASE_LIST_SIZE-add_size,  r.first);
	EXPECT_EQ(TEST_BASE_LIST_SIZE,  r.second);
}

TEST(SearchBoundaries, BeginLevel0) {
	const uint64_t add_size = TEST_BASE_LIST_ADDITIONAL_SIZE;
	List l{0};
	Matrix mm{};
	mm.random();

	l.generate_base_random(TEST_BASE_LIST_SIZE, mm);

	Element zero{};
	zero.zero();

	for (size_t i = 0; i < add_size; ++i) {
		l[i] = zero;
	}

	auto r = l.search_boundaries(zero, 0, n);
	EXPECT_NE(-1,  r.first);    // sanity check so we don't get any seg faults
	EXPECT_NE(-1,  r.second);

	EXPECT_EQ(0,  r.first);
	EXPECT_EQ(0+add_size,  r.second);
}

TEST(SearchBoundaries, MiddleLevel0) {
	const uint64_t add_size = TEST_BASE_LIST_ADDITIONAL_SIZE;
	const uint64_t middle_index = TEST_BASE_LIST_SIZE/2;

	List l{0};
	Matrix mm{};
	mm.random();

	l.generate_base_random(TEST_BASE_LIST_SIZE, mm);

	Element zero{};
	zero.zero();

	for (size_t i = 0; i < add_size; ++i) {
		l[middle_index + i] = zero;
	}

	auto r = l.search_boundaries(zero, 0, n);
	EXPECT_NE(-1,  r.first);    // sanity check so we dont get any seg faults
	EXPECT_NE(-1,  r.second);

	EXPECT_EQ(middle_index,  r.first);
	EXPECT_EQ(middle_index+add_size,  r.second);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
    return RUN_ALL_TESTS();
}
