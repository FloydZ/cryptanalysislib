#include <gtest/gtest.h>
#include <cstdint>
#include <bitset>

#include "binary.h"

#ifndef TESTSIZE
#define TESTSIZE 100
#endif

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(Internals, access) {
	BinaryContainer<n> b;
	for (uint32_t i = 0; i < BinaryContainer<n>::size(); ++i) {
		// this is the explicit cast steps to the final result.
		auto bit = b[i];
		bool bbit = bool(bit);
		EXPECT_EQ(0, bbit);
	}
}

TEST(Internals, access_pass_through) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;

	b1.zero();
	b2.random();

	EXPECT_EQ(b1.size(), b2.size());
	for (uint32_t i = 0; i < BinaryContainer<n>::size(); ++i) {
		b2[i] = b1[i];
	}

	for (uint32_t i = 0; i < BinaryContainer<n>::size(); ++i) {
		EXPECT_EQ(b2[i], b1[i]);
	}
}


TEST(Internals, masks){
	BinaryContainer<n> b;

	EXPECT_EQ(b.mask(0), 1);
	EXPECT_EQ(b.mask(1), 2);
	EXPECT_EQ(b.mask(2), 4);
}

TEST(Internals, random_limb_with_limits){
	using TestBinaryContainer = BinaryContainer<64>;

	constexpr uint64_t offset = 20;
	using LimbType = TestBinaryContainer::ContainerLimbType;
	for (uint32_t i = 0; i < 1; ++i) {
		for (uint32_t k_lower = 0; k_lower < 64; ++k_lower) {
			for (uint32_t k_upper = k_lower+offset; k_upper < 64; ++k_upper) {
				LimbType a = TestBinaryContainer::random_limb(k_lower, k_upper);
				LimbType lmask = TestBinaryContainer::lower_mask(k_lower%64);
				LimbType umask = TestBinaryContainer::higher_mask(k_upper%64);

				EXPECT_NE(a, 0);
				EXPECT_EQ(a&lmask, 0);
				EXPECT_EQ(a&umask, 0);
			}
		}
	}
}

TEST(Zero, Simple) {
	BinaryContainer<n> b;
	std::bitset<n> bb;
	b.zero();
	bb.reset();

	for (uint32_t i = 0; i < b.size(); ++i) {
		EXPECT_EQ(0, b[i]);
		EXPECT_EQ(bb[i], b[i]);
	}
}

TEST(Zero, Zero_with_Limits) {
	BinaryContainer<n> b;

	for (uint32_t k_lower = 1; k_lower < b.size(); ++k_lower) {
		for (uint32_t k_upper = k_lower+1; k_upper < b.size(); ++k_upper) {
			b.one();
			b.zero(k_lower, k_upper);

			for (uint32_t i = 0; i < k_lower; ++i) {
				EXPECT_EQ(1, b[i]);
			}

			for (uint32_t i = k_lower; i < k_upper; ++i) {
				EXPECT_EQ(0, b[i]);
			}

			for (uint32_t i = k_upper; i < b.size(); ++i) {
				EXPECT_EQ(1, b[i]);
			}
		}
	}

	for (uint32_t k_lower = 1; k_lower < b.size(); ++k_lower) {
		for (uint32_t k_upper = k_lower+1; k_upper < b.size(); ++k_upper) {
			b.zero();
			b.zero(k_lower, k_upper);

			for (uint32_t i = 0; i < k_lower; ++i) {
				EXPECT_EQ(0, b[i]);
			}

			for (uint32_t i = k_lower; i < k_upper; ++i) {
				EXPECT_EQ(0, b[i]);
			}

			for (uint32_t i = k_upper; i < b.size(); ++i) {
				EXPECT_EQ(0, b[i]);
			}
		}
	}
}

TEST(One, One) {
	BinaryContainer<n> b;
	std::bitset<n> bb;
	b.zero();
	b.one();
	bb.reset();

	for (uint32_t i = 0; i < b.size(); ++i) {
		EXPECT_NE(bb[i], b[i]);
		EXPECT_EQ(1, b[i]);
	}
}

TEST(One, One_with_Limits) {
	BinaryContainer<n> b;

	for (uint32_t k_lower = 0; k_lower < b.size(); ++k_lower) {
		for (uint32_t k_upper = k_lower+1; k_upper < b.size(); ++k_upper) {
			b.zero();
			b.one(k_lower, k_upper);

			for (uint32_t i = 0; i < k_lower; ++i) {
				EXPECT_EQ(0, b[i]);
			}

			for (uint32_t i = k_lower; i < k_upper; ++i) {
				EXPECT_EQ(1, b[i]);
			}

			for (uint32_t i = k_upper; i < b.size(); ++i) {
				EXPECT_EQ(0, b[i]);
			}
		}
	}

	for (uint32_t k_lower = 0; k_lower < b.size(); ++k_lower) {
		for (uint32_t k_upper = k_lower+1; k_upper < b.size(); ++k_upper) {
			b.one();
			b.one(k_lower, k_upper);

			for (uint32_t i = 0; i < k_lower; ++i) {
				EXPECT_EQ(1, b[i]);
			}

			for (uint32_t i = k_lower; i < k_upper; ++i) {
				EXPECT_EQ(1, b[i]);
			}

			for (uint32_t i = k_upper; i < b.size(); ++i) {
				EXPECT_EQ(1, b[i]);
			}
		}
	}
}


TEST(Set, Simple) {
	BinaryContainer<n> b;
	std::bitset<n> bb;

	bb.reset();
	b.zero();
	b[0] = true;
	EXPECT_EQ(b[0], 1);

	bb[0] = true;
	for (uint32_t i = 0; i < b.size(); ++i) {
		// das sind die expliciten casts.
		auto bit = b[i];
		bool bbit = bit;
		EXPECT_EQ(bb[i], bbit);
	}
}

TEST(Set, Random) {
	for (uint32_t i = 0; i < TESTSIZE; ++i) {
		BinaryContainer<n> b;
		std::bitset<n> bb;
		bb.reset();
		b.zero();

		auto pos = fastrandombytes_uint64() % n;
		b[pos] = bool(fastrandombytes_uint64() % 2);
		bb[pos] = b[pos];
		for (uint32_t j = 0; j < b.size(); ++j) {
			EXPECT_EQ(bb[j], b[j]);
		}
	}
}

TEST(Set, Full_Length_Zero) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;

	b1.one(); b2.zero();

	BinaryContainer<n>::set(b1, b2, 0, BinaryContainer<n>::size());

	for (uint32_t j = 0; j < BinaryContainer<n>::size(); ++j) {
		EXPECT_EQ(0, b1[j]);
	}
}

TEST(Set, Full_Length_One) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;

	b1.zero(); b2.zero();

	b2[0] = true;
	BinaryContainer<n>::set(b1, b2, 0, n);
	EXPECT_EQ(1, b2[0]);

	for (uint32_t j = 1; j < b1.size(); ++j) {
		EXPECT_EQ(0, b1[j]);
	}

	// 2. test.
	b1.zero(); b2.one();
	BinaryContainer<n>::set(b1, b2, 0, n);
	for (uint32_t j = 0; j < b1.size(); ++j) {
		EXPECT_EQ(true, b1[j]);
		EXPECT_EQ(1, b1[j]);

	}
}

TEST(Set, OffByOne_Lower_One) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;

	b1.one(); b2.zero();
	BinaryContainer<n>::set(b1, b2, 1, n);
	EXPECT_EQ(1, b1[0]);
	for (uint32_t j = 1; j < b1.size(); ++j) {
		EXPECT_EQ(0, b1[j]);
	}

	// 2. test.
	b1.zero(); b2.one();
	BinaryContainer<n>::set(b1, b2, 1, n);
	EXPECT_EQ(0, b1[0]);
	EXPECT_EQ(false, b1[0]);
	for (uint32_t j = 1; j < b1.size(); ++j) {
		EXPECT_EQ(true, b1[j]);
		EXPECT_EQ(1, b1[j]);
	}
}

TEST(Set, OffByOne_Higher_One) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;

	b1.zero(); b2.zero();

	b1[n-1] = true;   // this should be ignored.
	BinaryContainer<n>::set(b1, b2, 0, n - 1);
	EXPECT_EQ(1, b1[n-1]);

	for (uint32_t j = 0; j < b1.size() - 1; ++j) {
		EXPECT_EQ(0, b1[j]);
	}

	// 2. test.
	b1.zero(); b2.one();
	BinaryContainer<n>::set(b1, b2, 0, n - 1);
	EXPECT_EQ(0, b1[n-1]);
	EXPECT_EQ(false, b1[n-1]);
	for (uint32_t j = 0; j < b1.size() - 1; ++j) {
		EXPECT_EQ(true, b1[j]);
		EXPECT_EQ(1, b1[j]);
	}
}

TEST(Set, Complex_Ones) {
	//  test the following:
	//  [111...111]
	// +[000...000] \forall k_lower k_higher
	// =[0...0 1...1 0...0]
	//    k_lower k_higher
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;

	for (uint32_t k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < b1.size(); ++k_higher) {
			b1.zero(); b2.one();

			BinaryContainer<n>::set(b1, b2, k_lower, k_higher);

			for (uint32_t j = 0; j < k_lower; ++j) {
				EXPECT_EQ(0, b1[j]);
			}
			for (uint32_t j = k_lower; j < k_higher; ++j) {
				EXPECT_EQ(1, b1[j]);
			}
			for (uint32_t j = k_higher; j < b1.size(); ++j) {
				EXPECT_EQ(0, b1[j]);
			}
		}
	}
}

TEST(Set, Complex_Zeros) {
	//  test the following:
	//  [111000...000111]
	// +[111000...000111] \forall k_lower k_higher
	// =[1...1 0...0 1...1]
	//    k_lower k_higher

	BinaryContainer<n> b1;
	BinaryContainer<n> b2;

	for (uint32_t k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < b1.size(); ++k_higher) {
			b1.zero(); b2.zero();

			for (uint32_t j = 0; j < k_lower; ++j) {
				b1[j] = true;
				b2[j] = true;
			}
			for (uint32_t j = k_higher; j < b1.size(); ++j) {
				b1[j] = true;
				b2[j] = true;
			}


			BinaryContainer<n>::set(b1, b2, k_lower, k_higher);

			for (uint32_t j = 0; j < k_lower; ++j) {
				EXPECT_EQ(1, b1[j]);
			}
			for (uint32_t j = k_lower; j < k_higher; ++j) {
				EXPECT_EQ(0, b1[j]);
			}
			for (uint32_t j = k_higher; j < b1.size(); ++j) {
				EXPECT_EQ(1, b1[j]);
			}
		}
	}
}

TEST(Static_Add, Probabilistic){
	std::vector<std::pair<uint64_t, uint64_t>> boundsSet = {std::pair(0, 64),
	                                                        std::pair(0, 10),
	                                                        std::pair(2, 70),
	                                                        std::pair(64, 128),
	                                                        std::pair(0, 65),
	                                                        std::pair(3, 66)};

	for(auto bounds : boundsSet){
		uint64_t k_lower = bounds.first;
		uint64_t k_upper = bounds.second;

		for(uint64_t i = 0; i < 100; i++){
			uint64_t a = fastrandombytes_uint64();
			uint64_t b = fastrandombytes_uint64();
			uint64_t c = fastrandombytes_uint64();
			uint64_t d = fastrandombytes_uint64();
			uint64_t e = fastrandombytes_uint64();
			uint64_t f = fastrandombytes_uint64();

			BinaryContainer<128> b1;
			BinaryContainer<128> b2;
			BinaryContainer<128> b3;

			b1.data()[0] = a; b1.data()[1] = b;
			b2.data()[0] = c; b2.data()[1] = d;
			b3.data()[0] = e; b3.data()[1] = f;


			BinaryContainer<128>::add(b3, b2, b1, k_lower, k_upper);

			for(uint64_t k = 0; k < k_lower; k++){
				if(k < 64){
					ASSERT_EQ(b3.get_bit_shifted(k), (e>>k) & 1);
				}
				else {
					ASSERT_EQ(b3.get_bit_shifted(k), (f>>k) & 1);
				}
			}
			for(uint64_t k = k_lower; k < k_upper; k++){
				if(k < 64){
					ASSERT_EQ(b3.get_bit_shifted(k), ((a^c) >> k) & 1);
				}
				else {
					ASSERT_EQ(b3.get_bit_shifted(k), ((b^d) >> k) & 1);
				}
			}
			for(uint64_t k = k_upper; k < 128; k++){
				if(k < 64){
					ASSERT_EQ(b3.get_bit_shifted(k), (e>>k) & 1);
				}
				else {
					ASSERT_EQ(b3.get_bit_shifted(k), (f>>(k-64)) & 1);
				}
			}
		}
	}

}


TEST(Add, Probabilistic){
	using BinaryContainerTest = BinaryContainer<128>;
	std::vector<std::pair<uint64_t, uint64_t>> boundsSet = {std::pair(0, 64),
	                                                        std::pair(0, 10),
	                                                        std::pair(2, 70),
	                                                        std::pair(64, 128),
	                                                        std::pair(0, 65),
	                                                        std::pair(3, 66)};

	for(auto bounds : boundsSet) {
		uint64_t k_lower = bounds.first;
		uint64_t k_upper = bounds.second;

		for(uint64_t i = 0; i < 100; i++) {
			uint64_t a = fastrandombytes_uint64();
			uint64_t b = fastrandombytes_uint64();
			uint64_t c = fastrandombytes_uint64();
			uint64_t d = fastrandombytes_uint64();

			BinaryContainerTest b1;
			BinaryContainerTest b2;

			b1.data()[0] = a; b1.data()[1] = b;
			b2.data()[0] = c; b2.data()[1] = d;


			b1.add(b2, k_lower, k_upper);

			for(uint64_t k = 0; k < k_lower; k++){
				if(k < 64){
					ASSERT_EQ(b1.get_bit_shifted(k), (a>>k) & 1);
				}
				else {
					ASSERT_EQ(b1.get_bit_shifted(k), (b>>k) & 1);
				}
			}
			for(uint64_t k = k_lower; k < k_upper; k++){
				if(k < 64){
					ASSERT_EQ(b1.get_bit_shifted(k), ((a^c) >> k) & 1);
				}
				else {
					ASSERT_EQ(b1.get_bit_shifted(k), ((b^d) >> k) & 1);
				}
			}
			for(uint64_t k = k_upper; k < 128; k++){
				if(k < 64){
					ASSERT_EQ(b1.get_bit_shifted(k), (a>>k) & 1);
				}
				else {
					ASSERT_EQ(b1.get_bit_shifted(k), (b>>k) & 1);
				}
			}
		}
	}
}

TEST(Add, Norm){
	using BinaryContainerTest = BinaryContainer<128>;

	BinaryContainerTest b1;
	BinaryContainerTest b2;
	BinaryContainerTest b3;

	std::vector<std::pair<uint64_t, uint64_t>> boundsSet = {std::pair(0, 128)};

	for(auto bounds : boundsSet){
		uint64_t k_lower = bounds.first;
		uint64_t k_upper = bounds.second;
		b1.one();
		b2.zero();


		uint64_t norm = (k_lower+k_upper) / 2;

		bool result = BinaryContainerTest::add(b3, b2, b1, k_lower, k_upper, norm);
		ASSERT_EQ(true, result);
	}
}

TEST(Add, Full_Length_Zero) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;
	BinaryContainer<n> b3;

	b1.zero(); b2.zero(); b3.zero();

	BinaryContainer<n>::add(b3, b1, b2, 0, n);
	for (uint32_t j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(0, b3[j]);
	}
}

TEST(Add, Full_Length_One) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;
	BinaryContainer<n> b3;

	b1.zero(); b2.zero(); b3.zero();

	b1[0] = true;
	BinaryContainer<n>::add(b3, b1, b2, 0, n);
	EXPECT_EQ(1, b3[0]);

	for (uint32_t j = 1; j < b3.size(); ++j) {
		EXPECT_EQ(0, b3[j]);
	}

	// 2. test.
	b1.zero(); b2.zero(); b3.zero();
	for (uint32_t i = 0; i < b1.size(); ++i) {
		b1[i] = true;
	}

	BinaryContainer<n>::add(b3, b1, b2, 0, n);
	for (uint32_t j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(true, b3[j]);
		EXPECT_EQ(1, b3[j]);

	}

	//3.test
	b1.zero(); b2.zero(); b3.zero();
	for (uint32_t i = 0; i < b1.size(); ++i) {
		b1[i] = true;
		b2[i] = true;
	}

	BinaryContainer<n>::add(b3, b1, b2, 0, n);
	for (uint32_t j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(false, b3[j]);
		EXPECT_EQ(0, b3[j]);
	}
}

TEST(Add, OffByOne_Lower_One) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;
	BinaryContainer<n> b3;

	b1.zero(); b2.zero(); b3.zero();

	b1[0] = true;   // this should be ignored.
	BinaryContainer<n>::add(b3, b1, b2, 1, n);
	for (uint32_t j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(0, b3[j]);
	}

	// 2. test.
	b1.zero(); b2.zero(); b3.zero();
	for (uint32_t i = 0; i < b1.size(); ++i) {
		b1[i] = true;
	}

	BinaryContainer<n>::add(b3, b1, b2, 1, n);
	EXPECT_EQ(0, b3[0]);
	EXPECT_EQ(false, b3[0]);
	for (uint32_t j = 1; j < b3.size(); ++j) {
		EXPECT_EQ(true, b3[j]);
		EXPECT_EQ(1, b3[j]);
	}

	//3.test
	b1.zero(); b2.zero(); b3.zero();
	for (uint32_t i = 0; i < b1.size(); ++i) {
		b1[i] = true;
		b2[i] = true;
	}

	BinaryContainer<n>::add(b3, b1, b2, 1, n);
	EXPECT_EQ(0, b3[0]);
	EXPECT_EQ(false, b3[0]);

	for (uint32_t j = 1; j < b3.size(); ++j) {
		EXPECT_EQ(false, b3[j]);
		EXPECT_EQ(0, b3[j]);
	}
}

TEST(Add, OffByOne_Higher_One) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;
	BinaryContainer<n> b3;

	b1.zero(); b2.zero(); b3.zero();

	b1[n-1] = true;   // this should be ignored.
	BinaryContainer<n>::add(b3, b1, b2, 0, n - 1);
	for (uint32_t j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(0, b3[j]);
	}

	// 2. test.
	b1.zero(); b2.zero(); b3.zero();
	for (uint32_t i = 0; i < b1.size(); ++i) {
		b1[i] = true;
	}

	BinaryContainer<n>::add(b3, b1, b2, 0, n - 1);
	EXPECT_EQ(0, b3[n-1]);
	EXPECT_EQ(false, b3[n-1]);
	for (uint32_t j = 0; j < b3.size() - 1; ++j) {
		EXPECT_EQ(true, b3[j]);
		EXPECT_EQ(1, b3[j]);
	}

	//3.test
	b1.zero(); b2.zero(); b3.zero();
	for (uint32_t i = 0; i < b1.size(); ++i) {
		b1[i] = true;
		b2[i] = true;
	}

	BinaryContainer<n>::add(b3, b1, b2, 0, n - 1);
	EXPECT_EQ(0, b3[n-1]);
	EXPECT_EQ(false, b3[n-1]);

	for (uint32_t j = 1; j < b3.size() - 1; ++j) {
		EXPECT_EQ(false, b3[j]);
		EXPECT_EQ(0, b3[j]);
	}
}

TEST(Add, Complex_Ones) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;
	BinaryContainer<n> b3;

	for (uint32_t k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < b1.size(); ++k_higher) {
			b1.zero(); b2.zero(); b3.zero();

			for (uint32_t i = 0; i < b1.size(); ++i) {
				b1[i] = true;
			}

			BinaryContainer<n>::add(b3, b1, b2, k_lower, k_higher);

			for (uint32_t j = 0; j < k_lower; ++j) {
				EXPECT_EQ(0, b3[j]);
			}
			for (uint32_t j = k_lower; j < k_higher; ++j) {
				EXPECT_EQ(1, b3[j]);
			}
			for (uint32_t j = k_higher; j < b1.size(); ++j) {
				EXPECT_EQ(0, b3[j]);
			}
		}
	}
}

TEST(Sub, Full_Length_Zero) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;
	BinaryContainer<n> b3;

	b1.zero(); b2.zero(); b3.zero();

	BinaryContainer<n>::sub(b3, b1, b2, 0, n);
	for (uint32_t j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(0, b3[j]);
	}
}

TEST(Sub, Full_Length_One) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;
	BinaryContainer<n> b3;

	b1.zero(); b2.zero(); b3.zero();

	b1[0] = true;
	BinaryContainer<n>::sub(b3, b1, b2, 0, n);
	EXPECT_EQ(1, b3[0]);

	for (uint32_t j = 1; j < b3.size(); ++j) {
		EXPECT_EQ(0, b3[j]);
	}

	// 2. test.
	b1.zero(); b2.zero(); b3.zero();
	for (uint32_t i = 0; i < b1.size(); ++i) {
		b1[i] = true;
	}

	BinaryContainer<n>::sub(b3, b1, b2, 0, n);
	for (uint32_t j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(true, b3[j]);
		EXPECT_EQ(1, b3[j]);

	}

	//3.test
	b1.zero(); b2.zero(); b3.zero();
	for (uint32_t i = 0; i < b1.size(); ++i) {
		b1[i] = true;
		b2[i] = true;
	}

	BinaryContainer<n>::sub(b3, b1, b2, 0, n);
	for (uint32_t j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(false, b3[j]);
		EXPECT_EQ(0, b3[j]);
	}
}

TEST(Sub, OffByOne_Lower_One) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;
	BinaryContainer<n> b3;

	b1.zero(); b2.zero(); b3.zero();

	b1[0] = true;   // this should be ignored.
	BinaryContainer<n>::sub(b3, b1, b2, 1, n);
	for (uint32_t j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(0, b3[j]);
	}

	// 2. test.
	b1.zero(); b2.zero(); b3.zero();
	for (uint32_t i = 0; i < b1.size(); ++i) {
		b1[i] = true;
	}

	BinaryContainer<n>::sub(b3, b1, b2, 1, n);
	EXPECT_EQ(0, b3[0]);
	EXPECT_EQ(false, b3[0]);
	for (uint32_t j = 1; j < b3.size(); ++j) {
		EXPECT_EQ(true, b3[j]);
		EXPECT_EQ(1, b3[j]);
	}

	//3.test
	b1.zero(); b2.zero(); b3.zero();
	for (uint32_t i = 0; i < b1.size(); ++i) {
		b1[i] = true;
		b2[i] = true;
	}

	BinaryContainer<n>::sub(b3, b1, b2, 1, n);
	EXPECT_EQ(0, b3[0]);
	EXPECT_EQ(false, b3[0]);

	for (uint32_t j = 1; j < b3.size(); ++j) {
		EXPECT_EQ(false, b3[j]);
		EXPECT_EQ(0, b3[j]);
	}
}

TEST(Sub, OffByOne_Higher_One) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;
	BinaryContainer<n> b3;

	b1.zero(); b2.zero(); b3.zero();

	b1[n-1] = true;   // this should be ignored.
	BinaryContainer<n>::sub(b3, b1, b2, 0, n - 1);
	for (uint32_t j = 0; j < b3.size(); ++j) {
		EXPECT_EQ(0, b3[j]);
	}

	// 2. test.
	b1.zero(); b2.zero(); b3.zero();
	for (uint32_t i = 0; i < b1.size(); ++i) {
		b1[i] = true;
	}

	BinaryContainer<n>::sub(b3, b1, b2, 0, n - 1);
	EXPECT_EQ(0, b3[n-1]);
	EXPECT_EQ(false, b3[n-1]);
	for (uint32_t j = 0; j < b3.size() - 1; ++j) {
		EXPECT_EQ(true, b3[j]);
		EXPECT_EQ(1, b3[j]);
	}

	//3.test
	b1.zero(); b2.zero(); b3.zero();
	for (uint32_t i = 0; i < b1.size(); ++i) {
		b1[i] = true;
		b2[i] = true;
	}

	BinaryContainer<n>::sub(b3, b1, b2, 0, n - 1);
	EXPECT_EQ(0, b3[n-1]);
	EXPECT_EQ(false, b3[n-1]);

	for (uint32_t j = 1; j < b3.size() - 1; ++j) {
		EXPECT_EQ(false, b3[j]);
		EXPECT_EQ(0, b3[j]);
	}
}

TEST(Sub, Complex_Ones) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;
	BinaryContainer<n> b3;

	for (uint32_t k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < b1.size(); ++k_higher) {
			b1.zero(); b2.zero(); b3.zero();

			for (uint32_t i = 0; i < b1.size(); ++i) {
				b1[i] = true;
			}

			BinaryContainer<n>::sub(b3, b1, b2, k_lower, k_higher);

			for (uint32_t j = 0; j < k_lower; ++j) {
				EXPECT_EQ(0, b3[j]);
			}
			for (uint32_t j = k_lower; j < k_higher; ++j) {
				EXPECT_EQ(1, b3[j]);
			}
			for (uint32_t j = k_higher; j < b1.size(); ++j) {
				EXPECT_EQ(0, b3[j]);
			}
		}
	}
}


TEST(Cmp, Simple_Everything_False) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;

	b1.zero(); b2.one();

	for (uint32_t k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < b1.size(); ++k_higher) {
			EXPECT_EQ(false, BinaryContainer<n>::cmp(b1, b2, k_lower, k_higher));
		}
	}
}

TEST(Cmp, Simple_Everything_True) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;

	b1.zero(); b2.zero();

	for (uint32_t k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < b1.size(); ++k_higher) {
			EXPECT_EQ(true, BinaryContainer<n>::cmp(b1, b2, k_lower, k_higher));
		}
	}
}

TEST(Cmp, OffByOne_Lower_One) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;

	b1.zero(); b2.zero();

	b1[0] = true;
	EXPECT_EQ(1, b1[0]);
	EXPECT_EQ(false, BinaryContainer<n>::cmp(b1, b2, 0, b1.size()));
	for (uint32_t j = 1; j < b1.size(); ++j) {
		EXPECT_EQ(true, BinaryContainer<n>::cmp(b1, b2, j, b1.size()));
	}
}

TEST(Cmp, OffByOne_Higher_One) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;

	b1.zero(); b2.zero();

	b1[n-1] = true;
	EXPECT_EQ(false, BinaryContainer<n>::cmp(b1, b2, 0, n));
	for (uint32_t j = 0; j < b1.size() - 1; ++j) {
		EXPECT_EQ(true, BinaryContainer<n>::cmp(b1, b2, j, n - 1));
	}
}

TEST(Cmp, Complex_One) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;

	b1.zero();

	for (uint32_t k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < b1.size(); ++k_higher) {
			b2.zero();
			for (uint32_t i = k_lower; i < k_higher; ++i) {
				b2[i] = true;
			}

			if (k_lower > 0)
				EXPECT_EQ(true, BinaryContainer<n>::cmp(b1, b2, 0, k_lower));
			EXPECT_EQ(false, BinaryContainer<n>::cmp(b1, b2, k_lower, k_higher));
			EXPECT_EQ(true, BinaryContainer<n>::cmp(b1, b2, k_higher, b1.size()));


			b2.zero();
			EXPECT_EQ(true, BinaryContainer<n>::cmp(b1, b2, k_lower, k_higher));

			for (uint32_t i = k_higher; i < b2.size(); ++i) {
				b2[i] = true;
			}
			if (k_lower > 0)
				EXPECT_EQ(true, BinaryContainer<n>::cmp(b1, b2, 0, k_lower));
			EXPECT_EQ(true, BinaryContainer<n>::cmp(b1, b2, k_lower, k_higher));
			EXPECT_EQ(false, BinaryContainer<n>::cmp(b1, b2, k_higher, b1.size()));
		}
	}
}

TEST(Cmp, Complex_Zero) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;

	b1.zero();
	b2.zero();

	for (uint32_t i = 0; i < b1.size(); ++i) { b1[i] = true; }

	for (uint32_t k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < b1.size(); ++k_higher) {
			b2.zero();
			for (uint32_t i = k_lower; i < k_higher; ++i) {
				b2[i] = true;
			}
			if (k_lower > 0)
				EXPECT_EQ(false, BinaryContainer<n>::cmp(b1, b2, 0, k_lower));

			EXPECT_EQ(true, BinaryContainer<n>::cmp(b1, b2, k_lower, k_higher));
			EXPECT_EQ(false, BinaryContainer<n>::cmp(b1, b2, k_higher, b1.size()));


			b2.zero();
			EXPECT_EQ(false, BinaryContainer<n>::cmp(b1, b2, k_lower, k_higher));
		}
	}
}

TEST(Cmp, Special_OffByOne_Lower_Zero) {
	constexpr uint64_t size = 64;
	using TestBinaryContainer = BinaryContainer<size>;
	TestBinaryContainer b1;
	TestBinaryContainer b2;

	b2.zero();

	for (uint32_t i = 0; i < size-2; ++i) {
		b1.zero();
		b1[i] = true;
		EXPECT_EQ(1, b1[i]);
		EXPECT_EQ(false, TestBinaryContainer::cmp(b1, b2, 0, size));

		if (i > 0)
			EXPECT_EQ(false, TestBinaryContainer::cmp(b1, b2, 0, i+1));

		for (uint32_t j = i+1; j < size; ++j) {
			EXPECT_EQ(true, TestBinaryContainer::cmp(b1, b2, j, size));
		}
	}

}

TEST(Cmp, Special_OffByOne_Lower_One) {
	constexpr uint64_t size = 64;
	using TestBinaryContainer = BinaryContainer<size>;
	TestBinaryContainer b1;
	TestBinaryContainer b2;

	b2.one();

	for (uint32_t i = 0; i < size-2; ++i) {
		b1.zero();
		b1[i] = true;
		EXPECT_EQ(1, b1[i]);
		EXPECT_EQ(false, TestBinaryContainer::cmp(b1, b2, 0, size));

		if (i > 0)
			EXPECT_EQ(false, TestBinaryContainer::cmp(b1, b2, 0, i+1));

		for (uint32_t j = i+1; j < size; ++j) {
			EXPECT_EQ(false, TestBinaryContainer::cmp(b1, b2, j, size));
		}

		b1.one();
		for (uint32_t j = 0; j < size; ++j) {
			EXPECT_EQ(true, TestBinaryContainer::cmp(b1, b2, j, size));
		}
	}

}

TEST(IsGreater, Simple_Everything_False) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;

	b1.zero(); b2.one();

	for (uint32_t k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < b1.size(); ++k_higher) {
			EXPECT_EQ(false, b1.is_greater(b2, k_lower, k_higher));
			EXPECT_EQ(true, b2.is_greater(b1, k_lower, k_higher));
		}
	}
}

TEST(IsGreater, Simple_Everything_True) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;

	b1.one(); b2.zero();

	for (uint32_t k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < b1.size(); ++k_higher) {
			// std ::cout << k_lower << " " << k_higher << "\n";
			auto b = b1.is_greater(b2, k_lower, k_higher);
			EXPECT_EQ(true, b);
		}
	}
}

TEST(IsGreater, Complex_One) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;

	b1.zero();

	for (uint32_t k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < b1.size(); ++k_higher) {
			b2.zero();
			for (uint32_t i = 0; i < k_higher; ++i) {
				b2[i] = 1;
			}

			EXPECT_EQ(false, b1.is_greater(b2, k_lower, k_higher));
			EXPECT_EQ(false, b1.is_greater(b2, k_higher, b1.size()));
			EXPECT_EQ(true, b2.is_greater(b1, k_lower, k_higher));

			if (k_higher < b1.size() - 1){
				b1[k_higher] = 1;
				EXPECT_EQ(true, b1.is_greater(b2, k_higher, b1.size()));
				b1.zero();
			}


			b2.zero();
			for (uint32_t i = k_higher; i < b2.size(); ++i) {
				b2[i] = 1;
			}

			EXPECT_EQ(false, b1.is_greater(b2, k_lower, k_higher));
			EXPECT_EQ(false, b1.is_greater(b2, k_higher, b1.size()));

			EXPECT_EQ(false, b2.is_greater(b1, k_lower, k_higher));
			EXPECT_EQ(true, b2.is_greater(b1, k_higher, b1.size()));

			if (k_higher < b1.size() - 1){
				EXPECT_EQ(true, b2.is_greater(b1, k_lower, k_higher+1));
				EXPECT_EQ(false, b1.is_greater(b2, k_lower, k_higher+1));

			}
		}
	}
}

TEST(IsGreater, Complex) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;

	b1.zero();

	for (uint32_t k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < b1.size(); ++k_higher) {
			b2.zero();
			for (uint32_t i = 0; i < k_higher; ++i) {
				b2[i] = true;
			}

			EXPECT_EQ(false, b1.is_greater(b2, k_lower, k_higher));
			EXPECT_EQ(false, b1.is_greater(b2, k_higher, b1.size()));
			EXPECT_EQ(true, b2.is_greater(b1, k_lower, k_higher));

			if (k_higher < b1.size() - 1){
				b1[k_higher] = 1;
				EXPECT_EQ(true, b1.is_greater(b2, k_higher, b1.size()));
				b1.zero();
			}
		}
	}
}


TEST(IsLower, Simple_Everything_False) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;

	b1.zero(); b2.one();

	for (uint32_t k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < b1.size(); ++k_higher) {
			EXPECT_EQ(false, b2.is_lower(b1, k_lower, k_higher));
			EXPECT_EQ(true, b1.is_lower(b2, k_lower, k_higher));
		}
	}
}

TEST(IsLower, Simple_Everything_True) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;

	b1.one(); b2.zero();

	for (uint32_t k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < b1.size(); ++k_higher) {
			EXPECT_EQ(false, b1.is_lower(b2, k_lower, k_higher));
			EXPECT_EQ(true, b2.is_lower(b1, k_lower, k_higher));

		}
	}
}

TEST(IsLower, Complex_One) {
	BinaryContainer<n> b1;
	BinaryContainer<n> b2;

	b1.zero();

	for (uint32_t k_lower  = 0; k_lower < b1.size(); ++k_lower) {
		for (uint32_t k_higher = k_lower + 1; k_higher < b1.size(); ++k_higher) {
			b2.zero();
			for (uint32_t i = 0; i < k_higher; ++i) {
				b2[i] = 1;
			}

			EXPECT_EQ(true, b1.is_lower(b2, k_lower, k_higher));
			EXPECT_EQ(false, b2.is_lower(b1, k_lower, k_higher));
			EXPECT_EQ(false, b1.is_lower(b2, k_higher, b1.size()));

			if (k_higher < b1.size() - 1){
				b1[k_higher] = 1;
				EXPECT_EQ(false, b1.is_lower(b2, k_higher, b1.size()));
				EXPECT_EQ(true, b2.is_lower(b1, k_higher, b1.size()));

				b1.zero();
			}


			b2.zero();
			for (uint32_t i = k_higher; i < b2.size(); ++i) {
				b2[i] = 1;
			}

			EXPECT_EQ(false, b1.is_lower(b2, k_lower, k_higher));
			EXPECT_EQ(true, b1.is_lower(b2, k_higher, b1.size()));
			EXPECT_EQ(false, b2.is_lower(b1, k_higher, b1.size()));

			if (k_higher < b1.size() - 1){
				EXPECT_EQ(false, b2.is_lower(b1, k_lower, k_higher+1));
				EXPECT_EQ(true, b1.is_lower(b2, k_lower, k_higher+1));
			}
		}
	}
}


TEST(weight, Simple_Everything_True) {
	BinaryContainer<n> b1;
	b1.zero();

	for (uint32_t k_lower  = 1; k_lower < b1.size(); ++k_lower) {
		b1[k_lower-1] = true;
		EXPECT_EQ(k_lower, b1.weight());
		EXPECT_EQ(k_lower, b1.weight(0, k_lower));
		EXPECT_EQ(k_lower, b1.weight(0, b1.size()));
		if (k_lower + 1 < b1.size())
			EXPECT_EQ(0, b1.weight(k_lower +1, b1.size()));
	}
}

TEST(add_weight, Simple) {
	BinaryContainer<n> b1,b2,b3;
	b1.zero(); b2.zero(); b3.zero();
	uint32_t w1, w2;

	// Simple zero test.
	w1 = BinaryContainer<n>::add_weight(b3.ptr(), b1.ptr(), b2.ptr());
	w2 = BinaryContainer<n>::add_weight(b3, b1, b2);
	EXPECT_EQ(w1, w2);

	for(uint32_t i = 0; i < 100; i++) {
		b2.random(); b3.random();
		w1 = BinaryContainer<n>::add_weight(b3.ptr(), b1.ptr(), b2.ptr());
		w2 = BinaryContainer<n>::add_weight(b3, b1, b2);
		EXPECT_EQ(w1, w2);
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
