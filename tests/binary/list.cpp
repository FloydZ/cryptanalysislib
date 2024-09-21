#include <gtest/gtest.h>

#define TEST_BASE_LIST_SIZE 1000
#define TEST_BASE_LIST_ADDITIONAL_SIZE ((TEST_BASE_LIST_SIZE) / 10)

#include "binary.h"
#include "helper.h"
#include "list/list.h"

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
uint64_t helper_list_is_every_element_value_unique(const BinaryList &L) {
	if (L.load() == 0)
		return 0;

	uint64_t errors = 0;
	const uint64_t vs = L[0].value_size();
	for (size_t i = 0; i < L.load(); ++i) {
		for (size_t j = 0; j < L.load(); ++j) {
			// skip element if there are the same
			if (i == j)
				continue;

			bool equal = true;
			for (size_t k = 0; k < vs; ++k) {
				if (L[i].get_value().data()[k] != L[j].get_value().data()[k]) {
					equal = false;
					break;
				}
			}

			if (equal) {
				errors += 1;
			}
		}
	}

	// return only the half amount of errors, because we count every duplicate twice.
	return errors >> uint64_t(1);
}

uint64_t helper_check_weight_of_value(const BinaryList &l, const uint64_t e1, const uint64_t em1) {
	typedef typename BinaryElement::ValueDataType T;

	if (l.load() == 0) {
		return 0;
	}

	uint64_t errors = 0;
	const uint64_t vs = l[0].value_size();
	for (size_t i = 0; i < l.load(); ++i) {
		uint64_t counted_ones = 0;
		uint64_t counted_mones = 0;

		for (size_t j = 0; j < vs; ++j) {
			if (l[i].get_value().data()[j] == T(-1)) {
				counted_mones += 1;
			}

			if (l[i].get_value().data()[j] == 1) {
				counted_ones += 1;
			}
		}

		if ((counted_mones != em1) || (counted_ones != e1)) {
			errors += 1;
		}
	}

	return errors;
}

#ifndef EXTERNAL_MAIN
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
#endif
