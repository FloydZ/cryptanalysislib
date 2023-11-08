#include <gtest/gtest.h>
#include <bitset>

#include "helper.h"
#include "binary.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using BinaryContainerTest = BinaryContainer<n>;
using BinaryContainerTest2 = BinaryContainer<10*n>;

// Allow the tests only for smaller bit length
#if defined(NNN) && NNN <= 64
TEST(CmpSimple2, Simple_Everything_False) {
	BinaryContainerTest b1;
	BinaryContainerTest b2;

	b1.zero(); b2.one();

	for (uint32_t k_lower  = 0; k_lower < BinaryContainerTest::size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < BinaryContainerTest::size(); ++k_higher) {
			if ((k_lower/64) < (k_higher/64)) {
				break;
			}

			uint64_t limb = BinaryContainerTest::round_down_to_limb(k_lower);
			uint64_t mask = BinaryContainerTest::higher_mask(k_lower) & BinaryContainerTest::lower_mask(k_higher);
			EXPECT_EQ(false, BinaryContainerTest::cmp_simple2(b1, b2, limb, mask));
		}
	}
}

TEST(CmpSimple2, Simple_Everything_True) {
	BinaryContainerTest b1;
	BinaryContainerTest b2;

	b1.zero(); b2.zero();

	for (uint32_t k_lower  = 0; k_lower < BinaryContainerTest::size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < BinaryContainerTest::size(); ++k_higher) {
			if ((k_lower/64) < (k_higher/64)) {
				break;
			}

			uint64_t limb = BinaryContainerTest::round_down_to_limb(k_lower);
			uint64_t mask = BinaryContainerTest::higher_mask(k_lower) & BinaryContainerTest::lower_mask(k_higher);
			EXPECT_EQ(true, BinaryContainerTest::cmp_simple2(b1, b2, limb, mask));
		}
	}
}

TEST(CmpSimple2, OffByOne_Lower_One) {
	BinaryContainerTest b1;
	BinaryContainerTest b2;

	b1.zero(); b2.zero();

	b1[0] = true;
	uint64_t limb  = 0;
	uint64_t mask  = BinaryContainerTest::higher_mask(0);
	uint64_t mask2 = BinaryContainerTest::lower_mask(BinaryContainerTest::size()-1);
	mask &= mask2;

	EXPECT_EQ(1, b1[0]);
	EXPECT_EQ(false, BinaryContainerTest::cmp_simple2(b1, b2, limb, mask));
	for (uint32_t j = 1; j < BinaryContainerTest::size(); ++j) {
		limb = BinaryContainerTest::round_down_to_limb(0);
		mask = BinaryContainerTest::higher_mask(j) & BinaryContainerTest::lower_mask2(BinaryContainerTest::size());
		EXPECT_EQ(true, BinaryContainerTest::cmp_simple2(b1, b2, limb, mask));
	}
}

TEST(CmpSimple2_is_equal_simple2, Complex_One) {
	BinaryContainerTest b1;
	BinaryContainerTest b2;

	b1.zero();

	for (uint32_t k_lower  = 0; k_lower < BinaryContainerTest::size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < BinaryContainerTest::size(); ++k_higher) {
			if ((k_lower/64) < (k_higher/64)) {
				break;
			}
			uint64_t limb = BinaryContainerTest::round_down_to_limb(k_lower);
			uint64_t mask = BinaryContainerTest::higher_mask(k_lower) & BinaryContainerTest::lower_mask(k_higher);

			b2.zero();
			for (uint32_t i = k_lower; i < k_higher; ++i) {
				b2[i] = true;
			}

			EXPECT_EQ(BinaryContainerTest::cmp(b1, b2, k_lower, k_higher),
			          BinaryContainerTest::cmp_simple2(b1, b2, limb, mask));
			EXPECT_EQ(BinaryContainerTest::cmp(b1, b2, k_lower, k_higher),
			          b1.is_equal_simple2(b2, limb, mask));
		}
	}
}

TEST(is_greater_simple2, Simple_Everything_False) {
	BinaryContainerTest b1;
	BinaryContainerTest b2;

	b1.zero(); b2.one();

	for (uint32_t k_lower  = 0; k_lower < BinaryContainerTest::size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < BinaryContainerTest::size(); ++k_higher) {
			if ((k_lower/64) < (k_higher/64)) {
				break;
			}
			uint64_t limb = 0;
			uint64_t mask = BinaryContainerTest::higher_mask(k_lower) & BinaryContainerTest::lower_mask(k_higher);

			EXPECT_EQ(false, b1.is_greater_simple2(b2, limb, mask));
			EXPECT_EQ(true, b2.is_greater_simple2(b1, limb, mask));
		}
	}
}

TEST(is_greater_simple2, Simple_Everything_True) {
	BinaryContainerTest b1;
	BinaryContainerTest b2;

	b1.one(); b2.zero();

	for (uint32_t k_lower  = 0; k_lower < BinaryContainerTest::size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < BinaryContainerTest::size(); ++k_higher) {
			if ((k_lower/64) < (k_higher/64)) {
				break;
			}
			uint64_t limb = 0;
			uint64_t mask = BinaryContainerTest::higher_mask(k_lower) & BinaryContainerTest::lower_mask(k_higher);

			auto b = b1.is_greater_simple2(b2, limb, mask);
			EXPECT_EQ(true, b);
		}
	}
}

TEST(is_greater_simple2, Complex_One) {
	BinaryContainerTest b1;
	BinaryContainerTest b2;

	uint64_t mask;
	b1.zero();

	for (uint32_t k_lower  = 0; k_lower < BinaryContainerTest::size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < BinaryContainerTest::size()-1; ++k_higher) {
			if ((k_lower/64) < (k_higher/64)) {
				break;
			}
			b2.zero();
			for (uint32_t i = 0; i < k_higher; ++i) {
				b2[i] = true;
			}

			uint64_t limb = 0;
			mask = BinaryContainerTest::higher_mask(k_lower) & BinaryContainerTest::lower_mask2(k_higher);
			EXPECT_EQ(false, b1.is_greater_simple2(b2, limb, mask));

			mask = BinaryContainerTest::higher_mask(k_higher) & BinaryContainerTest::lower_mask2(BinaryContainerTest::size());
			EXPECT_EQ(false, b1.is_greater_simple2(b2, limb, mask));

			mask = BinaryContainerTest::higher_mask(k_lower) & BinaryContainerTest::lower_mask2(k_higher);
			EXPECT_EQ(true, b2.is_greater_simple2(b1, limb, mask));

			if (k_higher < BinaryContainerTest::size() - 1){
				b1[k_higher] = true;
				mask = BinaryContainerTest::higher_mask(k_higher) & BinaryContainerTest::lower_mask2(BinaryContainerTest::size()-1);
				EXPECT_EQ(true, b1.is_greater_simple2(b2, 0, mask));
				b1.zero();
			}


			b2.zero();
			for (uint32_t i = k_higher; i < b2.size(); ++i) {
				b2[i] = true;
			}

			mask = BinaryContainerTest::higher_mask(k_lower) & BinaryContainerTest::lower_mask2(k_higher);
			EXPECT_EQ(false, b1.is_greater_simple2(b2, limb, mask));

			mask = BinaryContainerTest::higher_mask(k_higher) & BinaryContainerTest::lower_mask2(BinaryContainerTest::size());
			EXPECT_EQ(false, b1.is_greater_simple2(b2, limb, mask));

			mask = BinaryContainerTest::higher_mask(k_lower) & BinaryContainerTest::lower_mask2(k_higher);
			EXPECT_EQ(false, b2.is_greater_simple2(b1, limb, mask));

			if (k_higher < BinaryContainerTest::size() - 1){
				mask = BinaryContainerTest::higher_mask(k_lower) & BinaryContainerTest::lower_mask(k_higher+1);
				EXPECT_EQ(true, b2.is_greater_simple2(b1, 0, mask));
				EXPECT_EQ(false, b1.is_greater_simple2(b2, 0, mask));

			}
		}
	}
}

TEST(is_greater_ext2, Complex_One) {
	BinaryContainerTest2 b1;
	BinaryContainerTest2 b2;

	uint64_t lower, upper, lmask, umask;
	b1.zero();

	for (uint32_t k_lower  = 0; k_lower < BinaryContainerTest2::size(); ++k_lower) {
		// + 65 so we always increase the counter the size of a limb
		for (uint32_t k_higher = k_lower + 64; k_higher < BinaryContainerTest2::size()-64; ++k_higher) {
			b2.zero();
			for (uint32_t i = 0; i < k_higher; ++i) {
				b2[i] = true;
			}

			lower = BinaryContainerTest2::round_down_to_limb(k_lower);
			upper = BinaryContainerTest2::round_down_to_limb(k_higher);
			lmask = BinaryContainerTest2::higher_mask(k_lower);
			umask = BinaryContainerTest2::lower_mask(k_higher);
			EXPECT_EQ(false, b1.is_greater_ext2(b2, lower, upper, lmask, umask));
			EXPECT_EQ(true, b2.is_greater_ext2(b1, lower, upper, lmask, umask));

			lower = BinaryContainerTest2::round_down_to_limb(k_higher);
			upper = BinaryContainerTest2::round_down_to_limb(BinaryContainerTest2::size()-1);
			lmask = BinaryContainerTest2::higher_mask(k_higher);
			umask = BinaryContainerTest2::lower_mask2(BinaryContainerTest2::size());
			EXPECT_EQ(false, b1.is_greater_ext2(b2, lower, upper, lmask, umask));

			if (k_higher < BinaryContainerTest2::size() - 1){
				b1[k_higher] = true;

				lower = BinaryContainerTest2::round_down_to_limb(k_higher);
				upper = BinaryContainerTest2::round_down_to_limb(BinaryContainerTest2::size()-1);
				lmask = BinaryContainerTest2::higher_mask(k_higher);
				umask = BinaryContainerTest2::lower_mask2(BinaryContainerTest2::size());
				EXPECT_EQ(true, b1.is_greater_ext2(b2, lower, upper, lmask, umask));
				EXPECT_EQ(false, b2.is_greater_ext2(b1, lower, upper, lmask, umask));

				b1.zero();
			}


			b1.zero();
			b2.zero();
			for (uint32_t i = k_higher; i < b2.size(); ++i) {
				b2[i] = true;
			}

			lower = BinaryContainerTest2::round_down_to_limb(k_lower);
			upper = BinaryContainerTest2::round_down_to_limb(k_higher);
			lmask = BinaryContainerTest2::higher_mask(k_lower);
			umask = BinaryContainerTest2::lower_mask2(k_higher);
			EXPECT_EQ(false, b1.is_greater_ext2(b2, lower, upper, lmask, umask));
		}
	}
}



TEST(is_lower_simple2, Simple_Everything_False) {
	BinaryContainerTest b1;
	BinaryContainerTest b2;

	b1.zero(); b2.one();

	for (uint32_t k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < b1.size(); ++k_higher) {
			uint64_t limb = 0;
			uint64_t mask = BinaryContainerTest::higher_mask(k_lower) & BinaryContainerTest::lower_mask(k_higher);

			EXPECT_EQ(false, b2.is_lower_simple2(b1, limb, mask));
			EXPECT_EQ(true, b1.is_lower_simple2(b2, limb, mask));
		}
	}
}

TEST(is_lower_simple2, Simple_Everything_True) {
	BinaryContainerTest b1;
	BinaryContainerTest b2;

	b1.one(); b2.zero();

	for (uint32_t k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < b1.size(); ++k_higher) {
			uint64_t limb = 0;
			uint64_t mask = BinaryContainerTest::higher_mask(k_lower) & BinaryContainerTest::lower_mask(k_higher);

			EXPECT_EQ(false, b1.is_lower_simple2(b2, limb, mask));
			EXPECT_EQ(true, b2.is_lower_simple2(b1, limb, mask));

		}
	}
}

TEST(is_lower_simple2, Complex_One) {
	BinaryContainerTest b1;
	BinaryContainerTest b2;

	uint64_t mask;
	b1.zero();

	for (uint32_t k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < b1.size()-1; ++k_higher) {
			b2.zero();
			for (uint32_t i = 0; i < k_higher; ++i) {
				b2[i] = true;
			}

			uint64_t limb = 0;
			mask = BinaryContainerTest::higher_mask(k_lower) & BinaryContainerTest::lower_mask2(k_higher);
			EXPECT_EQ(true, b1.is_lower_simple2(b2, limb, mask));
			EXPECT_EQ(false, b2.is_lower_simple2(b1, limb, mask));

			mask = BinaryContainerTest::higher_mask(k_higher) & BinaryContainerTest::lower_mask2(b1.size());
			EXPECT_EQ(false, b1.is_lower_simple2(b2, limb, mask));

			if (k_higher < b1.size() - 1){
				b1[k_higher] = true;

				mask = BinaryContainerTest::higher_mask(k_higher) & BinaryContainerTest::lower_mask(b1.size()-1);
				EXPECT_EQ(false, b1.is_lower_simple2(b2, limb, mask));
				EXPECT_EQ(true, b2.is_lower_simple2(b1, limb, mask));

				b1.zero();
			}

			b1.zero();
			b2.zero();
			for (uint32_t i = k_higher; i < b2.size(); ++i) {
				b2[i] = true;
			}

			mask = BinaryContainerTest::higher_mask(k_lower) & BinaryContainerTest::lower_mask2(k_higher);
			EXPECT_EQ(false, b1.is_lower_simple2(b2, limb, mask));

			mask = BinaryContainerTest::higher_mask(k_higher) & BinaryContainerTest::lower_mask2(b1.size());
			// EXPECT_EQ(true, b1.is_lower_simple2(b2, limb, mask));
			EXPECT_EQ(false, b2.is_lower_simple2(b1, limb, mask));

			if (k_higher < b1.size() - 1){
				mask = BinaryContainerTest::higher_mask(k_lower) & BinaryContainerTest::lower_mask(k_higher+1);
				EXPECT_EQ(false, b2.is_lower_simple2(b1, limb, mask));
				EXPECT_EQ(true, b1.is_lower_simple2(b2, limb, mask));
			}
		}
	}
}



TEST(is_lower_ext2, Simple_Everything_False) {
	BinaryContainerTest2 b1;
	BinaryContainerTest2 b2;

	b1.zero(); b2.one();

	for (uint32_t k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 65; k_higher < b1.size(); ++k_higher) {
			const uint64_t lower = BinaryContainerTest2::round_down_to_limb(k_lower);
			const uint64_t upper = BinaryContainerTest2::round_down_to_limb(k_higher);
			const uint64_t lmask = BinaryContainerTest2::higher_mask(k_lower);
			const uint64_t umask = BinaryContainerTest2::lower_mask2(k_higher);

			EXPECT_EQ(false, b2.is_lower_ext2(b1, lower, upper, lmask, umask));
			EXPECT_EQ(true, b1.is_lower_ext2(b2, lower, upper, lmask, umask));
		}
	}
}

TEST(is_lower_ext2, Simple_Everything_True) {
	BinaryContainerTest2 b1;
	BinaryContainerTest2 b2;

	b1.one(); b2.zero();

	for (uint32_t k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 65; k_higher < b1.size(); ++k_higher) {
			const uint64_t lower = BinaryContainerTest2::round_down_to_limb(k_lower);
			const uint64_t upper = BinaryContainerTest2::round_down_to_limb(k_higher);
			const uint64_t lmask = BinaryContainerTest2::higher_mask(k_lower);
			const uint64_t umask = BinaryContainerTest2::lower_mask2(k_higher);

			EXPECT_EQ(false, b1.is_lower_ext2(b2, lower, upper, lmask, umask));
			EXPECT_EQ(true, b2.is_lower_ext2(b1, lower, upper, lmask, umask));
		}
	}
}

TEST(is_lower_ext2, Complex_One) {
	BinaryContainerTest2 b1;
	BinaryContainerTest2 b2;

	uint64_t lower, upper, lmask, umask;
	b1.zero();

	for (uint32_t k_lower  = 0; k_lower < BinaryContainerTest2::size(); ++k_lower) {
		// + 65 so we always increase the counter the size of a limb
		for (uint32_t k_higher = k_lower + 64; k_higher < BinaryContainerTest2::size()-64; ++k_higher) {
			b2.zero();
			for (uint32_t i = 0; i < k_higher; ++i) {
				b2[i] = true;
			}

			lower = BinaryContainerTest2::round_down_to_limb(k_lower);
			upper = BinaryContainerTest2::round_down_to_limb(k_higher);
			lmask = BinaryContainerTest2::higher_mask(k_lower);
			umask = BinaryContainerTest2::lower_mask2(k_higher);
			EXPECT_EQ(true, b1.is_lower_ext2(b2, lower, upper, lmask, umask));
			EXPECT_EQ(false, b2.is_lower_ext2(b1, lower, upper, lmask, umask));

			lower = BinaryContainerTest2::round_down_to_limb(k_higher);
			upper = BinaryContainerTest2::round_down_to_limb(b1.size()-1);
			lmask = BinaryContainerTest2::higher_mask(k_higher);
			umask = BinaryContainerTest2::lower_mask2(b1.size());
			EXPECT_EQ(false, b1.is_lower_ext2(b2, lower, upper, lmask, umask));

			if (k_higher < b1.size() - 1){
				b1[k_higher] = true;

				lower = BinaryContainerTest2::round_down_to_limb(k_higher);
				upper = BinaryContainerTest2::round_down_to_limb(b1.size()-1);
				lmask = BinaryContainerTest2::higher_mask(k_higher);
				umask = BinaryContainerTest2::lower_mask2(b1.size());
				EXPECT_EQ(false, b1.is_lower_ext2(b2, lower, upper, lmask, umask));
				EXPECT_EQ(true, b2.is_lower_ext2(b1, lower, upper, lmask, umask));

				b1.zero();
			}


			b1.zero();
			b2.zero();
			for (uint32_t i = k_higher; i < b2.size(); ++i) {
				b2[i] = true;
			}

			lower = BinaryContainerTest2::round_down_to_limb(k_lower);
			upper = BinaryContainerTest2::round_down_to_limb(k_higher);
			lmask = BinaryContainerTest2::higher_mask(k_lower);
			umask = BinaryContainerTest2::lower_mask(k_higher);
			EXPECT_EQ(false, b1.is_lower_ext2(b2, lower, upper, lmask, umask));

			/*
			mask = BinaryContainerTest::higher_mask(k_higher) & BinaryContainerTest::lower_mask(b1.size()-1);
			// EXPECT_EQ(true, b1.is_lower_ext2(b2, limb, mask));
			EXPECT_EQ(false, b2.is_lower_ext2(b1, limb, mask));

			if (k_higher < b1.size() - 1){
				mask = BinaryContainerTest::higher_mask(k_lower) & BinaryContainerTest::lower_mask(k_higher+1);
				EXPECT_EQ(false, b2.is_lower_ext2(b1, limb, mask));
				EXPECT_EQ(true, b1.is_lower_ext2(b2, limb, mask));
			}*/
		}
	}
}
#endif

#ifndef EXTERNAL_MAIN
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
#endif