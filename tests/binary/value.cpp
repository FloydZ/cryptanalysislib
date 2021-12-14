#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "binary.h"
#include "helper.h"
#include "value.h"
#include "../test.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(ValueTest, DoesNotLeak) {
    auto*l = new BinaryValue;
    delete l;
}

TEST(ValueTest, AddFunctionDoesNotLeak) {
    BinaryValue l1;
    BinaryValue l2;
    BinaryValue l3;
    BinaryValue::add(l3, l1, l2, 0, l1.size());
}

TEST(ValueTest, AddFunctionSimple1) {
    BinaryValue l1;
    BinaryValue l2;
    BinaryValue l3;

    l1.data()[0] = true;
    l2.data()[0] = true;

    BinaryValue::add(l3, l1, l2, 0, l1.size());
    EXPECT_EQ(0, l3[0]);
    EXPECT_EQ(1, l1[0]);
    EXPECT_EQ(1, l2[0]);
}

TEST(ValueTest, AddDoesNotLeak) {
    BinaryValue l1;
    BinaryValue l2;
    BinaryValue l3;

    BinaryValue::add(l3, l1, l2, 0, l1.size());
}

TEST(ValueTest, AddSimple1) {
    BinaryValue l1{};
    BinaryValue l2{};
    BinaryValue l3{};
	l1.zero();
	l2.zero();
	l3.zero();

	l1.data()[0] = true;
    l2.data()[0] = true;

	BinaryValue::add(l3, l1, l2, 0, l1.size());
    EXPECT_EQ(0, l3[0]);
    EXPECT_EQ(1, l1[0]);
    EXPECT_EQ(1, l2[0]);
}

TEST(ValueTest, AddEqualSimple1) {
    BinaryValue l1{};
    BinaryValue l2{};

    l1.data()[0] = false;
    l2.data()[0] = true;

	BinaryValue::add(l1, l1, l2, 0, l1.size());
	EXPECT_EQ(1, l1[0]);
}


TEST(ValueComparisonTest, IsEqual) {
	BinaryValue l1{};
	BinaryValue l2{};
	l1.zero();
	l2.zero();

	/// level 0 = on all bits
	EXPECT_EQ(true, l1.is_equal(l2, 0, n));

	l2.data()[0] = true;
	EXPECT_EQ(false, l1.is_equal(l2, 0, n));
	EXPECT_EQ(true, l1.is_equal(l2, 1, n));

	l2.data()[n-1] = true;
	EXPECT_EQ(false, l1.is_equal(l2, n-1, n));
	EXPECT_EQ(true, l1.is_equal(l2, 1, n-1));
}

TEST(AssignmentTest, Copy) {
	BinaryValue v1{};
	BinaryValue v2{};
	v1.zero();
	v2.data()[0] = true;

	v1 = v2;
	EXPECT_EQ(1, v1[0]);

	for (int i = 1; i < v1.size(); ++i) {
		EXPECT_EQ(0, v1[i]);
	}
}

TEST(AssignmentTest, Copy2) {
	BinaryValue v1{};
	BinaryValue v2{};
	v1.zero();
	for (int i = 0; i < v1.size(); ++i) {
		v2.data()[i] = 1;
	}

	v1 = v2;

	for (int i = 0; i < v1.size(); ++i) {
		EXPECT_EQ(1, v1[i]);
	}

}

TEST(AssignmentTest, Copy3) {
	BinaryValue v1{};
	BinaryValue v2{};
	v1.zero();
	v1.data()[0] = true;
	for (int i = 0; i < v1.size(); ++i) {
		v2.data()[i] = i;
	}

	v2 = v1;
	EXPECT_EQ(1, v2[0]);

	for (int i = 1; i < v2.size(); ++i) {
		EXPECT_EQ(0, v2[i]);
	}
}

TEST(AssignmentTest, Move) {
	BinaryValue v1{};
	BinaryValue v2{};
	v1.zero();

	v2.data()[0] = true;

	v1 = std::move(v2);

	EXPECT_EQ(1, v1[0]);

	for (int i = 1; i < v1.size(); ++i) {
		EXPECT_EQ(0, v1[i]);
	}
}


TEST(AssignmentTest, Move2) {
	BinaryValue v1{};
	BinaryValue v2{};
	v1.zero();
	for (int i = 0; i < v1.size(); ++i) {
		v2.data()[i] = 1;
	}

	v1 = std::move(v2);

	for (int i = 0; i < v1.size(); ++i) {
		EXPECT_EQ(1, v1[i]);
	}

}

TEST(AssignmentTest, Move3) {
	BinaryValue v1{};
	BinaryValue v2{};
	v1.zero();
	v1.data()[0] = true;
	for (int i = 0; i < v1.size(); ++i) {
		v2.data()[i] = i;
	}

	v2 = std::move(v1);

	EXPECT_EQ(1, v2[0]);

	for (int i = 1; i < v2.size(); ++i) {
		EXPECT_EQ(0, v2[i]);
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
