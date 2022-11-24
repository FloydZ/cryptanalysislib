#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "test.h"
#include "helper.h"
#include "random.h"
#include "kAry_type.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(Label, DoesNotLeak) {
    auto*l = new kAryType;
    delete l;
}

TEST(Set, Simple) {
	kAryType l1;
	unsigned int t1 = fastrandombytes_uint64()% q;
	l1 = t1;
	EXPECT_EQ(l1, t1);
}

TEST(Set, EdgeCases) {
	kAryType l1;
	unsigned int t1 = q;
	l1 = t1;
	EXPECT_EQ(l1, 0);
}

TEST(Add, Zero) {
	kAryType l1;
	kAryType l2;
	kAryType l3;
	l1 = l2 = l3 = 0;
	l3 = l1 + l2;

	EXPECT_EQ(l3, 0);
	EXPECT_EQ(l1, 0);
	EXPECT_EQ(l2, 0);

	l3 = 1;
	l3 = l1 + l2;

	EXPECT_EQ(l3, 0);
	EXPECT_EQ(l1, 0);
	EXPECT_EQ(l2, 0);
}

TEST(Add, one) {
	kAryType l1;
	kAryType l2;
	kAryType l3;
	l1 = l2 = l3 = 1;
	l3 = l1 + l2;

	EXPECT_EQ(l3, 2 % q);

	EXPECT_EQ(l1, 1);
	EXPECT_EQ(l2, 1);
}

TEST(Add, simple) {
	kAryType l1;
	kAryType l2;
	kAryType l3;

	for (uint32_t i = 0; i < TESTSIZE; ++i) {
		uint64_t t1 = fastrandombytes_uint64();
		uint64_t t2 = fastrandombytes_uint64();


		l1 = t1;
		l2 = t2;

		l3 = l1 + l2;
		//EXPECT_EQ(l3, (t1 + t2) % q);
		EXPECT_EQ(l1, t1 % q);
		EXPECT_EQ(l2, t2 % q);
	}
}

TEST(Add, signed_simple) {
	kAryType l1;
	kAryType l2;
	kAryType l3;

	for (int i = 0; i < TESTSIZE; ++i) {
		int64_t t1 = fastrandombytes_uint64()% q;
		int64_t t2 = fastrandombytes_uint64()% q;

		l1 = t1;
		l2 = t2;

		l3 = l1 + l2;
		EXPECT_EQ(l3, uint64_t (t1 + t2 ) % q);
		EXPECT_EQ(l1, uint64_t(t1) % q);
		EXPECT_EQ(l2, uint64_t(t2) % q);
	}
}

TEST(Add, uint64_t) {
	kAryType l1;
	kAryType l2;
	kAryType l3;

	for (int i = 0; i < TESTSIZE; ++i) {
		uint64_t t1 = (fastrandombytes_uint64() * fastrandombytes_uint64());
		uint64_t t2 = (fastrandombytes_uint64() * fastrandombytes_uint64());

		l1 = t1;
		l2 = t2;

		l3 = l1 + l2;
		//EXPECT_EQ(l3, (t1 + t2 ) % q);
		EXPECT_EQ(l1, t1 % q);
		EXPECT_EQ(l2, t2 % q);
	}
}

//TEST(Sub, simple) {
//	kAryType l1;
//	kAryType l2;
//	kAryType l3;
//
//	for (int i = 0; i < TESTSIZE; ++i) {
//		unsigned int t1 = fastrandombytes_uint64();
//		unsigned int t2 = fastrandombytes_uint64();
//
//		l1 = t1;
//		l2 = t2;
//
//		l3 = l1 - l2;
//		EXPECT_EQ(l3, (t1 - t2 ) % q);
//		EXPECT_EQ(l1, t1 % q);
//		EXPECT_EQ(l2, t2 % q);
//	}
//}
//
//TEST(Sub, signed_simple) {
//	kAryType l1;
//	kAryType l2;
//	kAryType l3;
//
//	for (int i = 0; i < TESTSIZE; ++i) {
//		signed int t1 = fastrandombytes_uint64()% q;
//		signed int t2 = fastrandombytes_uint64()% q;
//
//		l1 = t1;
//		l2 = t2;
//
//		l3 = l1 - l2;
//		EXPECT_EQ(l3, (t1 - t2 ) % q);
//		EXPECT_EQ(l1, t1 % q);
//		EXPECT_EQ(l2, t2 % q);
//	}
//}
//
//TEST(Sub, uint64_t) {
//	kAryType l1;
//	kAryType l2;
//	kAryType l3;
//
//	for (int i = 0; i < TESTSIZE; ++i) {
//		uint64_t t1 = (fastrandombytes_uint64() * fastrandombytes_uint64());
//		uint64_t t2 = (fastrandombytes_uint64() * fastrandombytes_uint64());
//
//		l1 = t1;
//		l2 = t2;
//
//		l3 = l1 - l2;
//		EXPECT_EQ(l3, (t1 - t2 ) % q);
//		EXPECT_EQ(l1, t1 % q);
//		EXPECT_EQ(l2, t2 % q);
//	}
//}

//TEST(AddMul, simple) {
//	kAryType l1;
//	kAryType l2;
//	kAryType l3;
//
//	for (int i = 0; i < TESTSIZE; ++i) {
//		unsigned int t1 = fastrandombytes_uint64();
//		unsigned int t2 = fastrandombytes_uint64();
//		unsigned int t3 = fastrandombytes_uint64();
//
//		l1 = t1;
//		l2 = t2;
//		l3 = t3;
//
//		l3.addmul(l1, l2);
//		EXPECT_EQ(l3, (t3+(t1*t2)% q) % q);
//		EXPECT_EQ(l1, t1 % q);
//		EXPECT_EQ(l2, t2 % q);
//	}
//}
//
//TEST(AddMul, signed_simple) {
//	kAryType l1;
//	kAryType l2;
//	kAryType l3;
//
//	for (int i = 0; i < TESTSIZE; ++i) {
//		signed int t1 = fastrandombytes_uint64()% q;
//		signed int t2 = fastrandombytes_uint64()% q;
//		signed int t3 = fastrandombytes_uint64()% q;
//
//		l1 = t1;
//		l2 = t2;
//		l3 = t3;
//
//		l3.addmul(l1, l2);
//		EXPECT_EQ(l3, (t3+(t1*t2)% q) % q);
//		EXPECT_EQ(l1, t1 % q);
//		EXPECT_EQ(l2, t2 % q);
//	}
//}
//
//TEST(AddMul, uint64_t) {
//	kAryType l1;
//	kAryType l2;
//	kAryType l3;
//
//	for (int i = 0; i < TESTSIZE; ++i) {
//		uint64_t t1 = (fastrandombytes_uint64() * fastrandombytes_uint64());
//		uint64_t t2 = (fastrandombytes_uint64() * fastrandombytes_uint64());
//		uint64_t t3 = fastrandombytes_uint64();
//
//		l1 = t1;
//		l2 = t2;
//		l3 = t3;
//
//		l3.addmul(l1, l2);
//		EXPECT_EQ(l3, (t3+(t1*t2)% q) % q);
//		EXPECT_EQ(l1, t1 % q);
//		EXPECT_EQ(l2, t2 % q);
//	}
//}


TEST(Comparison, Simple) {
	kAryType l1, l2;
	unsigned int t1 = fastrandombytes_uint64() % (q-1);
	unsigned int t2 = t1 + 1;

	l1 = t1;
	l2 = t2;

	EXPECT_EQ(true, l1 != l2);
	EXPECT_EQ(false, l1 == l2);
	EXPECT_EQ(true, l1 < l2);
	EXPECT_EQ(true, l1 <= l2);
	EXPECT_EQ(false, l1 > l2);
	EXPECT_EQ(false, l1 >= l2);

	EXPECT_EQ(true, l1 == l1);
	EXPECT_EQ(true, l2 == l2);
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
