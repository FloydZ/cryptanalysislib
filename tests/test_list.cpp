#include <gtest/gtest.h>
#include <cstdint>

#define TEST_BASE_LIST_SIZE 100
#define TEST_BASE_LIST_ADDITIONAL_SIZE TEST_BASE_LIST_SIZE/10

#include "test.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

#ifdef USE_FPLLL
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
	if (l.get_load() == 0)
		return 0;

	uint64_t errors = 0;
	const uint64_t vs = l[0].value_size();
	for (int i = 0; i < l.get_load(); ++i) {
		uint64_t counted_ones  = 0;
		uint64_t counted_mones = 0;

		for (int j = 0; j < vs; ++j) {
			if (l[i].get_value().data()[j] == -1)
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

TEST(ListIntTest, DoesNotLeak) {
    List l {1};
}


TEST(ListIntTest, CreateEmptyBaseLists) {
	List l{0};
	fplll::ZZ_mat<kAryType> m;
	m.fill(0);
	const Matrix_T<ZZ_mat<kAryType>> mm(m);

	Element zero{};
	zero.zero();

	l.generate_base_random(TEST_BASE_LIST_SIZE, mm);
	EXPECT_EQ(TEST_BASE_LIST_SIZE,  l.size());

	for (uint64_t i = 0; i < TEST_BASE_LIST_SIZE; ++i) {
		// zero means on all coordinates of the Label
		EXPECT_EQ(true,  zero.is_equal(l[i], -1));
	}
}

TEST(GenerateBaseLex, SimpleAndLabelZero) {
	/// only checks
	/// 	- if all 'Labels differ from each other
	///		- the hamming weight is correct for all 'Values'
	///		- all labels are zero

	const uint64_t hw = n/2;
	List l{0};
	fplll::ZZ_mat<kAryType> m;
	m.fill(0);
	const Matrix_T<ZZ_mat<kAryType>> mm(m);

	Element e{}; e.zero();

	l.generate_base_lex(TEST_BASE_LIST_SIZE, hw, mm);

	uint64_t errors = helper_list_is_every_element_value_unique(l);
	EXPECT_EQ(TEST_BASE_LIST_SIZE, l.size());
	EXPECT_EQ(0, errors);


	for (uint64_t i = 0; i < TEST_BASE_LIST_SIZE; ++i) {
		EXPECT_EQ(true,  e.is_equal(l[i], -1));
		EXPECT_EQ(hw, l[i].get_value().data().weight());
	}
}

TEST(SearchBoundaries, BasicLevel0) {
	// simple checks:
	//      -
	List l{0};
	fplll::ZZ_mat<kAryType> m(n,n);
	m.gen_uniform(2);
	const Matrix_T<ZZ_mat<kAryType>> mm(m);
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
	fplll::ZZ_mat<kAryType> m(n,n);
	m.gen_uniform(1);
	const Matrix_T<ZZ_mat<kAryType>> mm(m);

	l.generate_base_random(TEST_BASE_LIST_SIZE, mm);

	Element zero{};
	zero.zero();

	l.append(zero);    // add the zero element

	auto r = l.search_boundaries(zero, 0, n);
	EXPECT_NE(-1,  r.first);    // sanity check so we dont get any seg faults
	EXPECT_NE(-1,  r.second);

	EXPECT_EQ(TEST_BASE_LIST_SIZE,  r.first);
	EXPECT_EQ(TEST_BASE_LIST_SIZE+1,  r.second);
}

TEST(SearchBoundaries, End2Level0) {
	const uint64_t add_size = TEST_BASE_LIST_ADDITIONAL_SIZE;
	List l{0};
	fplll::ZZ_mat<kAryType> m(n,n);
	m.gen_uniform(1);
	const Matrix_T<ZZ_mat<kAryType>> mm(m);

	l.generate_base_random(TEST_BASE_LIST_SIZE, mm);

	Element zero{};
	zero.zero();

	// some arbitrary amount
	for (uint64_t i = 0; i < add_size; ++i) {
		l.append(zero);
	}

	auto r = l.search_boundaries(zero, 0, n);
	EXPECT_NE(r.second,  r.first);    // sanity check so we dont get any seg faults

	EXPECT_EQ(TEST_BASE_LIST_SIZE,  r.first);
	EXPECT_EQ(TEST_BASE_LIST_SIZE+add_size,  r.second);
}

TEST(SearchBoundaries, BeginLevel0) {
	const uint64_t add_size = TEST_BASE_LIST_ADDITIONAL_SIZE;
	List l{0};
	fplll::ZZ_mat<kAryType> m(n,n);
	m.gen_uniform(2);
	const Matrix_T<ZZ_mat<kAryType>> mm(m);

	l.generate_base_random(TEST_BASE_LIST_SIZE, mm);

	Element zero{}, *ret;
	zero.zero();

	for (int i = 0; i < add_size; ++i) {
		l.set_data(zero, i);    // add the zero element
	}

	auto r = l.search_boundaries(zero, 0, n);
	EXPECT_NE(-1,  r.first);    // sanity check so we dont get any seg faults
	EXPECT_NE(-1,  r.second);

	EXPECT_EQ(0,  r.first);
	EXPECT_EQ(0+add_size,  r.second);
}

TEST(SearchBoundaries, MiddleLevel0) {
	const uint64_t add_size = TEST_BASE_LIST_ADDITIONAL_SIZE;
	const uint64_t middle_index = TEST_BASE_LIST_SIZE/2;

	List l{0};
	fplll::ZZ_mat<kAryType> m(n,n);
	m.gen_uniform(2);
	const Matrix_T<ZZ_mat<kAryType>> mm(m);

	l.generate_base_random(TEST_BASE_LIST_SIZE, mm);

	Element zero{};
	zero.zero();

	for (int i = 0; i < add_size; ++i) {
		l.set_data(zero, middle_index+i);    // add the zero element
	}

	auto r = l.search_boundaries(zero, 0, n);
	EXPECT_NE(-1,  r.first);    // sanity check so we dont get any seg faults
	EXPECT_NE(-1,  r.second);

	EXPECT_EQ(middle_index,  r.first);
	EXPECT_EQ(middle_index+add_size,  r.second);
}

TEST(SearchBoundaries, BasicLevel) {
	ASSERT(__level_translation_array[1]>= 5 && "this test doesnt make sense");

	List l{TEST_BASE_LIST_SIZE};
	fplll::ZZ_mat<kAryType> m(n,n);
	m.gen_uniform(4);
	const Matrix_T<ZZ_mat<kAryType>> mm(m);

	l.generate_base_random(TEST_BASE_LIST_SIZE, mm);

	Element zero{};
	zero.zero();

	// std::cout << l;

	for (uint64_t level = 0; level < __level_translation_array.size()-1; ++level) {
		// get the correct upper and lower bound of coordinates to match
		uint64_t k_lower, k_higher;
		translate_level(&k_lower, &k_higher, level, __level_translation_array);

		// nothing should be found. Depending on your luck and the single differences between two elements within
		// 'translation_array' is can be possible that a element ot the list is zero on these coordinates.
		auto r = l.search_boundaries(zero, k_lower, k_higher);
		EXPECT_EQ(r.second,  r.first);
		EXPECT_EQ(l.get_load(),  r.second);
	}
}

TEST(SearchBoundaries, Basiclevel1) {
	const uint64_t pos = 2;
	List l{TEST_BASE_LIST_SIZE};
	fplll::ZZ_mat<kAryType> m(n,n);
	m.gen_uniform(1);
	const Matrix_T<ZZ_mat<kAryType>> mm(m);

	l.generate_base_random(TEST_BASE_LIST_SIZE, mm);

	Element zero{};
	zero.zero();

	l.set_data(zero, pos);

	for (uint64_t level = 0; level < __level_translation_array.size()-1; ++level) {
		// get the correct upper and lower bound of coordinates to match
		uint64_t k_lower, k_higher;
		translate_level(&k_lower, &k_higher, level, __level_translation_array);

		// nothing should be found. Depending on your luck and the single differences between two elements within
		// 'translation_array' is can be possible that a element ot the list is zero on these coordinates.
		auto r = l.search_boundaries(zero, k_lower, k_higher);
		EXPECT_EQ(pos,  r.first);
		EXPECT_EQ(pos+1,  r.second);
	}
}

#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
    return RUN_ALL_TESTS();
}
