#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "helper.h"
#include "label.h"
#include "binary.h"
#include "../test.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(Label, DoesNotLeak) {
    auto*l = new BinaryLabel;
    delete l;
}

TEST(Label, Check_References) {
	BinaryLabel l{};
	auto b = l[0];      // this should be ok because we get a const reference
	l[0] = true;        // this should also be ok, if the 'container' class decides if its a reference or not.
	l.data()[0] = true; // this should also be ok, if the 'container' class decides if its a reference or not.
}


TEST(Label, Zero) {
	BinaryLabel l;
	l.data()[0] = fastrandombytes_uint64();

	l.zero();
	for (int i = 0; i < l.size(); ++i) {
		EXPECT_EQ(0, l[i].data());
		EXPECT_EQ(kAryType(0), l[i]);
	}
}

TEST(Label, Random) {
	BinaryLabel l;

	l.random();
	uint64_t ctr = 0;
	for (int i = 0; i < l.size(); ++i) {
		if(l.data()[i] != 0)
			ctr += 1;
	}

	EXPECT_NE(0, ctr);
}


TEST(Add, AddWithLevelAllCoordinates) {
	BinaryLabel l1, l2, l3;
	l1.zero(); l2.zero(); l3.zero();

	// only a simple test.
	for (int i = 0; i < l1.size(); ++i) {
		l1.data()[i] = i;
		l2.data()[i] = i;
	}

	uint64_t k_lower, k_higher;
	translate_level(&k_lower, &k_higher, -1, __level_translation_array);
	BinaryLabel::add(l3, l1, l2, k_lower, k_higher);

	for (int i = 0; i < l1.size(); ++i) {
		EXPECT_EQ(l3[i].data(),  (l1[i].data() + l2[i].data()) % q);
	}
}

TEST(Add, AddWithLevelWithTranslationArray) {
	BinaryLabel l1, l2, l3;
	uint64_t k_lower, k_higher;

	for (int r = 0; r < TESTSIZE; ++r) {
		l1.zero();
		l2.zero();
		l3.zero();

		// only a simple test.
		for (int i = 0; i < l1.size(); ++i) {
			l1.data()[i] = i;
			l2.data()[i] = fastrandombytes_uint64();;
		}

		for (int j = 0; j < __level_translation_array.size() - 1; ++j) {
			l3.zero();

			translate_level(&k_lower, &k_higher, j, __level_translation_array);
			BinaryLabel::add(l3, l1, l2, k_lower, k_higher);

			EXPECT_NE(__level_translation_array[j], __level_translation_array[j + 1]);
			EXPECT_LT(__level_translation_array[j], n);
			EXPECT_LE(__level_translation_array[j + 1], n);

			for (int i = __level_translation_array[j]; i < __level_translation_array[j + 1]; ++i) {
				EXPECT_EQ(l3[i].data(), (l1[i].data() + l2[i].data()) % q);
			}
			for (int i = 0; i < __level_translation_array[j]; ++i) {
				EXPECT_EQ(0, l3[i].data());
				EXPECT_EQ(kAryType(0), l3[i]);
			}
			for (int i = __level_translation_array[j + 1]; i < l3.size(); ++i) {
				EXPECT_EQ(0, l3[i].data());
				EXPECT_EQ(kAryType(0), l3[i]);
			}
		}
	}
}

TEST(Add, AddWithLevel) {
	BinaryLabel l1, l2, l3;
	uint64_t k_lower, k_higher;

	for (int r = 0; r < TESTSIZE; ++r) {

		l1.zero();
		l2.zero();
		l3.zero();

		// only a simple test.
		for (int i = 0; i < l1.size(); ++i) {
			l1.data()[i] = i;
			l2.data()[i] = fastrandombytes_uint64();
		}

		for (int j = 0; j < __level_translation_array.size() - 1; ++j) {
			l3.zero();

			translate_level(&k_lower, &k_higher, j, __level_translation_array);
			BinaryLabel::add(l3, l1, l2, k_lower, k_higher);

			EXPECT_NE(__level_translation_array[j], __level_translation_array[j + 1]);
			EXPECT_LT(__level_translation_array[j], n);
			EXPECT_LE(__level_translation_array[j + 1], n);

			for (int i = __level_translation_array[j]; i < __level_translation_array[j + 1]; ++i) {
				EXPECT_EQ(l3[i].data(), (l1[i].data() + l2[i].data()) % q);
			}
			for (int i = 0; i < __level_translation_array[j]; ++i) {
				EXPECT_EQ(0, l3[i].data());
				EXPECT_EQ(kAryType(0), l3[i]);
			}
			for (int i = __level_translation_array[j + 1]; i < l3.size(); ++i) {
				EXPECT_EQ(0, l3[i].data());
				EXPECT_EQ(kAryType(0), l3[i]);
			}
		}
	}
}

TEST(Add, AddWithK) {
	BinaryLabel l1, l2, l3;
	for (int r = 0; r < TESTSIZE; ++r) {

		l1.zero();
		l2.zero();
		l3.zero();

		// only a simple test.
		for (int i = 0; i < l1.size(); ++i) {
			l1.data()[i] = i;
			l2.data()[i] = fastrandombytes_uint64();
		}

		for (int k_lower = 0; k_lower < l3.size(); ++k_lower) {
			for (int k_higher = k_lower+1; k_higher < l3.size(); ++k_higher) {
				l3.zero();
				BinaryLabel::add(l3, l1, l2, k_lower, k_higher);

				EXPECT_LE(k_lower, n);
				EXPECT_LE(k_higher, n);

				for (int i = k_lower; i < k_higher; ++i) {
					EXPECT_EQ(l3[i].data(), (l1[i].data() + l2[i].data()) % q);
				}
				for (int i = 0; i < k_lower; ++i) {
					EXPECT_EQ(0, l3[i].data());
					EXPECT_EQ(kAryType(0), l3[i]);
				}
				for (int i = k_higher; i < l3.size(); ++i) {
					EXPECT_EQ(0, l3[i].data());
					EXPECT_EQ(kAryType(0), l3[i]);
				}
			}
		}
	}
}


TEST(Sub, SubWithLevelAllCoordinates) {
	BinaryLabel l1, l2, l3;
	l1.zero(); l2.zero(); l3.zero();

	// only a simple test.
	for (int i = 0; i < l1.size(); ++i) {
		l1.data()[i] = i;
		l2.data()[i] = i;
	}

	uint64_t k_lower, k_higher;
	translate_level(&k_lower, &k_higher, -1, __level_translation_array);

	BinaryLabel::sub(l3, l1, l2, k_lower, k_higher);

	for (int i = 0; i < l1.size(); ++i) {
		EXPECT_EQ(l3[i].data(),  (l1[i].data() - l2[i].data()) % q);
	}
}

TEST(Sub, SubWithLevelWithTranslationArray) {
	BinaryLabel l1, l2, l3;
	uint64_t k_lower, k_higher;

	for (int r = 0; r < TESTSIZE; ++r) {

		l1.zero();
		l2.zero();
		l3.zero();

		// only a simple test.
		for (int i = 0; i < l1.size(); ++i) {
			l1.data()[i] = i;
			l2.data()[i] = fastrandombytes_uint64();
		}

		for (int j = 0; j < __level_translation_array.size() - 1; ++j) {
			l3.zero();

			translate_level(&k_lower, &k_higher, j, __level_translation_array);
			BinaryLabel::sub(l3, l1, l2, k_lower, k_higher);

			EXPECT_NE(__level_translation_array[j], __level_translation_array[j + 1]);
			EXPECT_LT(__level_translation_array[j], n);
			EXPECT_LE(__level_translation_array[j + 1], n);

			for (int i = __level_translation_array[j]; i < __level_translation_array[j + 1]; ++i) {
				int64_t tmp = (l1[i].data() - l2[i].data()) % q;

				EXPECT_EQ(l3[i].data(), tmp);
			}
			for (int i = 0; i < __level_translation_array[j]; ++i) {
				EXPECT_EQ(0, l3[i].data());
				EXPECT_EQ(kAryType(0), l3[i]);
			}
			for (int i = __level_translation_array[j + 1]; i < l3.size(); ++i) {
				EXPECT_EQ(0, l3[i].data());
				EXPECT_EQ(kAryType(0), l3[i]);
			}
		}
	}
}

TEST(Sub, SubWithLevel) {
	BinaryLabel l1, l2, l3;
	for (int r = 0; r < TESTSIZE; ++r) {

		l1.zero();
		l2.zero();
		l3.zero();

		// only a simple test.
		for (int i = 0; i < l1.size(); ++i) {
			l1.data()[i] = i;
			l2.data()[i] = fastrandombytes_uint64();
		}

		for (int j = 0; j < __level_translation_array.size() - 1; ++j) {
			l3.zero();

			uint64_t k_lower, k_higher;
			translate_level(&k_lower, &k_higher, j, __level_translation_array);

			BinaryLabel::sub(l3, l1, l2, k_lower, k_higher);

			EXPECT_NE(__level_translation_array[j], __level_translation_array[j + 1]);
			EXPECT_LT(__level_translation_array[j], __level_translation_array[j + 1]);
			EXPECT_LT(__level_translation_array[j], n);
			EXPECT_LE(__level_translation_array[j + 1], n);

			for (int i = __level_translation_array[j]; i < __level_translation_array[j + 1]; ++i) {
				int64_t tmp = (int64_t(l1[i].data() - l2[i].data())) % q;
				if (tmp < 0)
					tmp += q;

				EXPECT_EQ(l3[i].data(), tmp);
			}
			for (int i = 0; i < __level_translation_array[j]; ++i) {
				EXPECT_EQ(0, l3[i].data());
				EXPECT_EQ(kAryType(0), l3[i]);
			}
			for (int i = __level_translation_array[j + 1]; i < l3.size(); ++i) {
				EXPECT_EQ(0, l3[i].data());
				EXPECT_EQ(kAryType(0), l3[i]);
			}
		}
	}
}

TEST(Sub, SubWithK) {
	BinaryLabel l1, l2, l3;
	for (int r = 0; r < TESTSIZE; ++r) {

		l1.zero();
		l2.zero();
		l3.zero();

		// only a simple test.
		for (int i = 0; i < l1.size(); ++i) {
			l1.data()[i] = i;
			l2.data()[i] = fastrandombytes_uint64();
		}

		for (int k_lower = 0; k_lower < l3.size(); ++k_lower) {
			for (int k_higher = k_lower+1; k_higher < l3.size(); ++k_higher) {
				l3.zero();
				BinaryLabel::sub(l3, l1, l2, k_lower, k_higher);

				EXPECT_LE(k_lower, n);
				EXPECT_LE(k_higher, n);

				for (int i = k_lower; i < k_higher; ++i) {
					EXPECT_EQ(l3[i].data(), (l1[i].data() - l2[i].data()) % q);
				}
				for (int i = 0; i < k_lower; ++i) {
					EXPECT_EQ(0, l3[i].data());
					EXPECT_EQ(kAryType(0), l3[i]);
				}
				for (int i = k_higher; i < l3.size(); ++i) {
					EXPECT_EQ(0, l3[i].data());
					EXPECT_EQ(kAryType(0), l3[i]);
				}
			}
		}
	}
}


TEST(Neg, NegWithLevelAllCoordinates) {
	BinaryLabel l1, l2;
	l1.zero(); l2.zero();

	// only a simple test.
	for (int i = 0; i < l1.size(); ++i) {
		l2.data()[i] = l1.data()[i] = i;
	}

	uint64_t k_lower, k_higher;
	translate_level(&k_lower, &k_higher, -1, __level_translation_array);

	l1.neg(k_lower, k_higher);

	for (int i = 0; i < l1.size(); ++i) {
		EXPECT_EQ(l1[i].data(),  (q - l2[i].data()) % q);
		if (l1[i].data() != 0)
			EXPECT_EQ(l1[i].data(),  q - l2[i].data());
	}
}

TEST(Neg, NegWithLevel) {
	BinaryLabel l1, l2;
	for (int r = 0; r < TESTSIZE; ++r) {
		l1.zero(); l2.zero();

		for (int level = 0; level < __level_translation_array.size() - 1; ++level) {
			EXPECT_NE(__level_translation_array[level], __level_translation_array[level + 1]);
			EXPECT_LT(__level_translation_array[level], __level_translation_array[level + 1]);
			EXPECT_LT(__level_translation_array[level], n);
			EXPECT_LE(__level_translation_array[level + 1], n);

			for (int i = 0; i < l1.size(); ++i) {
				l2.data()[i] = l1.data()[i] = fastrandombytes_uint64();
			}

			uint64_t k_lower, k_higher;
			translate_level(&k_lower, &k_higher, level, __level_translation_array);

			l1.neg(k_lower, k_higher);

			for (int i = __level_translation_array[level]; i < __level_translation_array[level+1]; ++i) {
				EXPECT_EQ(l1[i].data(),  (q - l2[i].data()) % q);
				if (l1[i].data() != 0)
					EXPECT_EQ(l1[i].data(),  q - l2[i].data());
			}
			for (int i = 0; i < __level_translation_array[level]; ++i) {
				EXPECT_EQ(l2[i].data(), l1[i].data());
				EXPECT_EQ(l2[i], l1[i]);
			}
			for (int i = __level_translation_array[level + 1]; i < l1.size(); ++i) {
				EXPECT_EQ(l2[i].data(), l1[i].data());
				EXPECT_EQ(l2[i], l1[i]);
			}
		}
	}
}

TEST(Neg, NegWithK) {
	BinaryLabel l1, l2;
	for (int r = 0; r < TESTSIZE; ++r) {
		l1.zero();
		l2.zero();

		for (int k_lower = 0; k_lower < l1.size(); ++k_lower) {
			for (int k_higher = k_lower+1; k_higher < l1.size(); ++k_higher) {
				EXPECT_LE(k_lower, n);
				EXPECT_LE(k_higher, n);

				for (int i = 0; i < l1.size(); ++i) {
					l1.data()[i] = l2.data()[i] = fastrandombytes_uint64();
				}

				l1.neg(k_lower, k_higher);

				for (int i = k_lower; i < k_higher; ++i) {
					EXPECT_EQ(l1[i].data(),  (q - l2[i].data()) % q);
					if (l1[i].data() != 0)
						EXPECT_EQ(l1[i].data(),  q - l2[i].data());
				}
				for (int i = 0; i < k_lower; ++i) {
					EXPECT_EQ(l2[i].data(), l1[i].data());
					EXPECT_EQ(l2[i], l1[i]);
				}
				for (int i = k_higher; i < l1.size(); ++i) {
					EXPECT_EQ(l2[i].data(), l1[i].data());
					EXPECT_EQ(l2[i], l1[i]);
				}
			}
		}
	}
}


TEST(Compare_Is_Equal, AllLevelsSimple) {
	BinaryLabel l1, l2;
	l1.zero(); l2.zero();

	for (int i = 0; i < l1.size(); ++i) {
		EXPECT_EQ(l1[i].data(),  l2[i].data());
	}

	uint64_t k_lower, k_higher;
	translate_level(&k_lower, &k_higher, -1, __level_translation_array);
	EXPECT_EQ(true, l1.is_equal(l2, k_lower, k_higher));
}

TEST(Compare_Is_Equal, AllLevelsSimpleWithoutTranslationArray) {
	BinaryLabel l1, l2;
	l1.zero(); l2.zero();

	for (int i = 0; i < l1.size(); ++i) {
		EXPECT_EQ(l1[i].data(),  l2[i].data());
	}

	uint64_t k_lower, k_higher;
	translate_level(&k_lower, &k_higher, -1, __level_translation_array);
	EXPECT_EQ(true, l1.is_equal(l2, k_lower, k_higher));
}



TEST(Compare_Is_Lower, AllCoordinatesSimple) {
	BinaryLabel l1, l2;
	l1.zero(); l2.zero();

	uint64_t k_lower, k_higher;
	translate_level(&k_lower, &k_higher, -1, __level_translation_array);
	EXPECT_EQ(false, l1.is_lower(l2, k_lower, k_higher));
}



#ifndef EXTERNAL_MAIN
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
#endif
