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

TEST(ValueTest, DoesNotLeak) {
    auto*l = new Value;
    delete l;
}

TEST(ValueTest, AddFunctionDoesNotLeak) {
    Value l1;
    Value l2;
    Value l3;
    Value::add(l3, l1, l2, 0, l1.size());
}

TEST(ValueTest, AddFunctionSimple1) {
    Value l1;
    Value l2;
    Value l3;

    l1.data()[0] = 1;
    l2.data()[0] = 1;

    Value::add(l3, l1, l2, 0, l1.size());
    EXPECT_EQ(2, l3[0]);
    EXPECT_EQ(1, l1[0]);
    EXPECT_EQ(1, l2[0]);
}

TEST(ValueTest, AddDoesNotLeak) {
    Value l1;
    Value l2;
    Value l3;

    Value::add(l3, l1, l2, 0, l1.size());
}

TEST(ValueTest, AddSimple1) {
    Value l1{};
    Value l2{};
    Value l3{};
	l1.zero();
	l2.zero();
	l3.zero();

	l1.data()[0] = 1;
    l2.data()[0] = 1;

	Value::add(l3, l1, l2, 0, l1.size());
    EXPECT_EQ(2, l3[0]);
    EXPECT_EQ(1, l1[0]);
    EXPECT_EQ(1, l2[0]);
}

TEST(ValueTest, AddEqualSimple1) {
    Value l1{};
    Value l2{};

    l1.data()[0] = 0;
    l2.data()[0] = 1;

	Value::add(l1, l1, l2, 0, l1.size());
	EXPECT_EQ(1, l1[0]);
}


TEST(ValueComparisonTest, IsEqual) {
	Value l1{};
	Value l2{};
	l1.zero();
	l2.zero();

	/// level 0 = on all bits
	EXPECT_EQ(true, l1.is_equal(l2));
	EXPECT_EQ(true, l1.is_equal(l2, 0, n));

	l2.data()[0] = 1;
	EXPECT_EQ(false, l1.is_equal(l2));
	EXPECT_EQ(false, l1.is_equal(l2, 0, n));
	EXPECT_EQ(true, l1.is_equal(l2, 1, n));

	l2.data()[n-1] = 1;
	EXPECT_EQ(false, l1.is_equal(l2));
	EXPECT_EQ(false, l1.is_equal(l2, n-1, n));
	EXPECT_EQ(true, l1.is_equal(l2, 1, n-1));
}

TEST(AssignmentTest, Copy) {
	Value v1{};
	Value v2{};
	v1.zero();
	v2.data()[0] = 1;

	v1 = v2;
	EXPECT_EQ(1, v1[0]);

	for (int i = 1; i < v1.size(); ++i) {
		EXPECT_EQ(0, v1[i]);
	}
}

TEST(AssignmentTest, Copy2) {
	Value v1{};
	Value v2{};
	v1.zero();
	for (int i = 0; i < v1.size(); ++i) {
		v2.data()[i] = i;
	}

	v1 = v2;

	for (int i = 0; i < v1.size(); ++i) {
		EXPECT_EQ(i, v1[i]);
	}

}

TEST(AssignmentTest, Copy3) {
	Value v1{};
	Value v2{};
	v1.zero();
	v1.data()[0] = 1;
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
	Value v1{};
	Value v2{};
	v1.zero();

	v2.data()[0] = 1;

	v1 = std::move(v2);

	EXPECT_EQ(1, v1[0]);

	for (int i = 1; i < v1.size(); ++i) {
		EXPECT_EQ(0, v1[i]);
	}
}


TEST(AssignmentTest, Move2) {
	Value v1{};
	Value v2{};
	v1.zero();
	for (int i = 0; i < v1.size(); ++i) {
		v2.data()[i] = i;
	}

	v1 = std::move(v2);

	for (int i = 0; i < v1.size(); ++i) {
		EXPECT_EQ(i, v1[i]);
	}

}

TEST(AssignmentTest, Move3) {
	Value v1{};
	Value v2{};
	v1.zero();
	v1.data()[0] = 1;
	for (int i = 0; i < v1.size(); ++i) {
		v2.data()[i] = i;
	}

	v2 = std::move(v1);

	EXPECT_EQ(1, v2[0]);

	for (int i = 1; i < v2.size(); ++i) {
		EXPECT_EQ(0, v2[i]);
	}
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
