#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "test.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using Label_Type = uint64_t;

TEST(Label, DoesNotLeak) {
    auto*l = new Label;
    delete l;
}

TEST(Label, Zero) {
	Label l;
	l.data()[0] = rng();

	l.zero();
	for (uint32_t i = 0; i < Label::size(); ++i) {
		EXPECT_EQ(0, l[i]);
		EXPECT_EQ(Label_Type(0), l[i]);
	}
}

TEST(Label, Random) {
	Label l;

	l.random();
	uint64_t ctr = 0;
	for (uint32_t i = 0; i < Label::size(); ++i) {
		// super stupid test. But who cares
		if(l[i] != 0) {
			ctr += 1;
		}
	}

	EXPECT_NE(0, ctr);

}


TEST(Add, AddWithLevelAllCoordinates) {
	Label l1, l2, l3;
	l1.zero(); l2.zero(); l3.zero();

	// only a simple test.
	for (uint32_t i = 0; i < Label::size(); ++i) {
		l1.data()[i] = i;
		l2.data()[i] = i;
	}

	uint32_t k_lower, k_higher;
	translate_level(&k_lower, &k_higher, -1, __level_translation_array);
	Label::add(l3, l1, l2, k_lower, k_higher);

	for (uint32_t i = 0; i < Label::size(); ++i) {
		EXPECT_EQ(l3[i],  (l1[i] + l2[i]) % q);
	}
}

TEST(Add, AddWithLevelWithTranslationArray) {
	Label l1, l2, l3;
	uint32_t k_lower, k_higher;

	for (uint32_t r = 0; r < TESTSIZE; ++r) {
		l1.zero();
		l2.zero();
		l3.zero();

		// only a simple test.
		for (uint32_t i = 0; i < Label::size(); ++i) {
			l1[i] = i;
			l2[i] = rng();
		}

		for (uint32_t j = 0; j < __level_translation_array.size() - 1; ++j) {
			l3.zero();

			translate_level(&k_lower, &k_higher, j, __level_translation_array);
			Label::add(l3, l1, l2, k_lower, k_higher);

			EXPECT_NE(__level_translation_array[j], __level_translation_array[j + 1]);
			EXPECT_LT(__level_translation_array[j], n);
			EXPECT_LE(__level_translation_array[j + 1], n);

			for (uint32_t i = __level_translation_array[j]; i < __level_translation_array[j + 1]; ++i) {
				EXPECT_EQ(l3[i], (l1[i] + l2[i]) % q);
			}
			for (uint32_t i = 0; i < __level_translation_array[j]; ++i) {
				EXPECT_EQ(0, l3[i]);
				EXPECT_EQ(Label_Type(0), l3[i]);
			}
			for (uint32_t i = __level_translation_array[j + 1]; i < Label::size(); ++i) {
				EXPECT_EQ(0, l3[i]);
				EXPECT_EQ(Label_Type(0), l3[i]);
			}
		}
	}
}

TEST(Add, AddWithLevel) {
	Label l1, l2, l3;
	uint32_t k_lower, k_higher;

	for (uint32_t r = 0; r < TESTSIZE; ++r) {

		l1.zero();
		l2.zero();
		l3.zero();

		// only a simple test.
		for (uint32_t i = 0; i < Label::size(); ++i) {
			l1[i] = i;
			l2[i] = rng();
		}

		for (uint32_t j = 0; j < __level_translation_array.size() - 1; ++j) {
			l3.zero();

			translate_level(&k_lower, &k_higher, j, __level_translation_array);
			Label::add(l3, l1, l2, k_lower, k_higher);

			EXPECT_NE(__level_translation_array[j], __level_translation_array[j + 1]);
			EXPECT_LT(__level_translation_array[j], n);
			EXPECT_LE(__level_translation_array[j + 1], n);

			for (uint32_t i = __level_translation_array[j]; i < __level_translation_array[j + 1]; ++i) {
				EXPECT_EQ(l3[i], (l1[i] + l2[i]) % q);
			}
			for (uint32_t i = 0; i < __level_translation_array[j]; ++i) {
				EXPECT_EQ(0, l3[i]);
				EXPECT_EQ(Label_Type(0), l3[i]);
			}
			for (uint32_t i = __level_translation_array[j + 1]; i < Label::size(); ++i) {
				EXPECT_EQ(0, l3[i]);
				EXPECT_EQ(Label_Type(0), l3[i]);
			}
		}
	}
}

TEST(Add, AddWithK) {
	Label l1, l2, l3;
	for (uint32_t r = 0; r < TESTSIZE; ++r) {

		l1.zero();
		l2.zero();
		l3.zero();

		// only a simple test.
		for (uint32_t i = 0; i < Label::size(); ++i) {
			l1[i] = i;
			l2[i] = rng();
		}

		for (uint32_t k_lower = 0; k_lower < Label::size(); ++k_lower) {
			for (uint32_t k_higher = k_lower+1; k_higher < Label::size(); ++k_higher) {
				l3.zero();
				Label::add(l3, l1, l2, k_lower, k_higher);

				EXPECT_LE(k_lower, n);
				EXPECT_LE(k_higher, n);

				for (uint32_t i = k_lower; i < k_higher; ++i) {
					EXPECT_EQ(l3[i], (l1[i] + l2[i]) % q);
				}
				for (uint32_t i = 0; i < k_lower; ++i) {
					EXPECT_EQ(0, l3[i]);
					EXPECT_EQ(Label_Type(0), l3[i]);
				}
				for (uint32_t i = k_higher; i < Label::size(); ++i) {
					EXPECT_EQ(0, l3[i]);
					EXPECT_EQ(Label_Type(0), l3[i]);
				}
			}
		}
	}
}


TEST(Sub, SubWithLevelAllCoordinates) {
	Label l1, l2, l3;
	l1.zero(); l2.zero(); l3.zero();

	// only a simple test.
	for (uint32_t i = 0; i < Label::size(); ++i) {
		l1[i] = i;
		l2[i] = i;
	}

	uint32_t k_lower, k_higher;
	translate_level(&k_lower, &k_higher, -1, __level_translation_array);

	Label::sub(l3, l1, l2, k_lower, k_higher);

	for (uint32_t i = 0; i < Label::size(); ++i) {
		EXPECT_EQ(l3[i], (l1[i] - l2[i]) % q);
	}
}


TEST(Neg, NegWithK) {
	Label l1, l2;
	for (uint32_t r = 0; r < TESTSIZE; ++r) {
		l1.zero();
		l2.zero();

		for (uint32_t k_lower = 0; k_lower < Label::size(); ++k_lower) {
			for (uint32_t k_higher = k_lower+1; k_higher < Label::size(); ++k_higher) {
				EXPECT_LE(k_lower, n);
				EXPECT_LE(k_higher, n);


				l1.random();
				l2 = l1;
				l1.neg(k_lower, k_higher);

				for (uint32_t i = k_lower; i < k_higher; ++i) {
					EXPECT_EQ(l1[i], (q - l2[i]) % q);
					if (l1[i] != 0) {
						EXPECT_EQ(l1[i],  q - l2[i]);
					}
				}
				for (uint32_t i = 0; i < k_lower; ++i) {
					EXPECT_EQ(l2[i], l1[i]);
					EXPECT_EQ(l2[i], l1[i]);
				}
				for (uint32_t i = k_higher; i < Label::size(); ++i) {
					EXPECT_EQ(l2[i], l1[i]);
					EXPECT_EQ(l2[i], l1[i]);
				}
			}
		}
	}
}


TEST(Compare_Is_Equal, AllLevelsSimple) {
	Label l1, l2;
	l1.zero(); l2.zero();

	for (uint32_t i = 0; i < Label::size(); ++i) {
		EXPECT_EQ(l1[i], l2[i]);
	}

	uint32_t k_lower, k_higher;
	translate_level(&k_lower, &k_higher, -1, __level_translation_array);
	EXPECT_EQ(true, l1.is_equal(l2, k_lower, k_higher));
}

TEST(Compare_Is_Equal, AllLevelsSimpleWithoutTranslationArray) {
	Label l1, l2;
	l1.zero(); l2.zero();

	for (uint32_t i = 0; i < Label::size(); ++i) {
		EXPECT_EQ(l1[i],  l2[i]);
	}

	uint32_t k_lower, k_higher;
	translate_level(&k_lower, &k_higher, -1, __level_translation_array);
	EXPECT_EQ(true, l1.is_equal(l2, k_lower, k_higher));
}

TEST(Compare_Is_Equal, AllK) {
	ASSERT(q > 2 && "q must be bigger than 2 for this test");
	Label l1, l2;

	for (uint32_t k_lower = 0; k_lower < Label::size(); ++k_lower) {
		for (uint32_t k_higher = k_lower+1; k_higher < Label::size(); ++k_higher) {
			l1.zero(); l2.zero();

			for (uint32_t i = k_lower; i < k_higher; ++i) {
				EXPECT_EQ(l1[i],  l2[i]);

			}

			EXPECT_EQ(true, l1.is_equal(l2, k_lower, k_higher));
		}
	}
}


TEST(Compare_Is_Lower, AllCoordinatesSimple) {
	Label l1, l2;
	l1.zero(); l2.zero();

	uint32_t k_lower, k_higher;
	translate_level(&k_lower, &k_higher, -1, __level_translation_array);
	EXPECT_EQ(false, l1.is_lower(l2, k_lower, k_higher));
}

TEST(Compare_Is_Lower, AllK) {
	ASSERT(q > 2 && "q must be bigger than 2 for this test");

	Label l1, l2;

	for (uint32_t k_lower = 0; k_lower < Label::size(); ++k_lower) {
		for (uint32_t k_higher = k_lower+1; k_higher < Label::size(); ++k_higher) {
			l1.zero(); l2.zero();

			EXPECT_EQ(false, l1.is_lower(l2, k_lower, k_higher));

			// because this is a probabilistic test, we need to make sure that the probability of a false Negative is low
			// 5 is completely arbitrary
			if ((k_higher - k_lower) > 5){
				l1.random(); l2.zero();
				l2.data()[k_higher - 1] = 0;
				l1.data()[k_higher - 1] = 1;
				EXPECT_EQ(true, l2.is_lower(l1, k_lower, k_higher));

				l2.data()[k_lower] = 1;
				EXPECT_EQ(true, l2.is_lower(l1, k_lower, k_higher));

				l1.zero();
				l2.data()[k_higher - 1] = 2;
				l1.data()[k_higher - 1] = 0;
				EXPECT_EQ(false, l2.is_lower(l1, k_lower, k_higher));
			}
		}
	}
}


int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	rng_seed(time(nullptr));
    return RUN_ALL_TESTS();
}
