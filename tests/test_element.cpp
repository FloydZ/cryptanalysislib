#include <gtest/gtest.h>
#include <iostream>

#include "test.h"
#include "helper.h"
#include "element.h"
#include "matrix/fq_matrix.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using Matrix 	= FqMatrix<T, n, n, q>;
using Element	= Element_T<Value, Label, Matrix>;
TEST(ElementTest, Reference) {
	Element e{};
	e.zero();
}

TEST(ElementTest, DoesNotLeak) {
    Element *l = new Element;
    delete l;
}

TEST(DoesNotLeak, DoesNotLeak) {
    Element l1;
    Element l2;
    Element l3;

    Element::add(l3, l1, l2);
}


TEST(Add, Zero) {
	Element l1;
	Element l2;
	Element l3;
	l1.zero();
	l2.zero();
	l3.zero();

	Element::add(l3, l1, l2);

	EXPECT_EQ(true, l1.is_equal(l2));
	EXPECT_EQ(true, l2.is_equal(l3));
}

TEST(Add, SimpleValue1) {
	Element l1;
	Element l2;
	Element l3;
	l1.zero();
	l2.zero();
	l3.zero();
	l1.get_value().data()[0] = 1;

	// assumes that 'Global_Matrix' = Identity Matrix
	Element::add(l3, l1, l2);
	//std::cout << l1;
	//std::cout << l2;
	//std::cout << l3;

	EXPECT_EQ(true, l3.get_value().is_equal(l1.get_value()));
	EXPECT_EQ(false, l3.get_value().is_equal(l2.get_value()));
}

TEST(Add, SimpleLabel1) {
	Element l1;
	Element l2;
	Element l3;
	l1.zero();
	l2.zero();
	l3.zero();
	l1.get_label().data()[0] = 1;

	// assumes that 'Global_Matrix' = Identity Matrix
	Element::add(l3, l1, l2);
	EXPECT_EQ(false, l3.is_equal(l2));
}


TEST(ElementComparisonTest, IsEqual) {
	Element l1{};
	Element l2{};
	l1.zero();
	l2.zero();

	/// level 0 = on all bits
	EXPECT_EQ(true, l1.is_equal(l2));
	EXPECT_EQ(true, l1.is_equal(l2, 0, n));
}

TEST(ElementComparisonTest, IsGreater) {
	Element l1{};
	Element l2{};
	l1.zero();
	l2.zero();

	/// level 0 = on all bits
	EXPECT_EQ(false, l1.is_greater(l2));
	for (uint64_t i = 0; i < Element::value_size(); ++i) {
		EXPECT_EQ(false, l1.is_greater(l2, i, n));
	}
}


TEST(AssignmentTest, CopyValue) {
	Element v1{};
	Element v2{};
	v1.zero();
	v2.get_value().data()[0] = 1;

	v1 = v2;
	EXPECT_EQ(1, v1.get_value(0));
}

TEST(AssignmentTest, CopyValue2) {
	Element v1{};
	Element v2{};
	v1.zero();
	for (size_t i = 0; i < Element::value_size(); ++i) {
		v2.value[i] = i;
	}

	v1 = v2;

	for (size_t i = 0; i < Element::value_size(); ++i) {
		EXPECT_EQ(i, v1.get_value(i));
	}

}

TEST(AssignmentTest, CopyValue3) {
	Element v1{};
	Element v2{};
	v1.zero();
	v1.get_value().data()[0] = 1;
	for (size_t i = 0; i < Element::value_size(); ++i) {
		v2.value[i] = i;
	}

	v2 = v1;
	EXPECT_EQ(1, v2.get_value(0));

	for (size_t i = 1; i < Element::value_size(); ++i) {
		EXPECT_EQ(0, v2.get_value(i));
	}
}


TEST(AssignmentTest, MoveValue) {
	Element v1{};
	Element v2{};
	v1.zero();

	v2.get_value().data()[0] = 1;

	v1 = std::move(v2);
	EXPECT_EQ(1, v1.get_value(0));
}

TEST(AssignmentTest, MoveValue2) {
	Element v1{};
	Element v2{};
	v1.zero();
	for (size_t i = 0; i < Element::value_size(); ++i) {
		v2.get_value().data()[i] = i;
	}

	v1 = std::move(v2);

	for (size_t i = 0; i < Element::value_size(); ++i) {
		EXPECT_EQ(i, v1.get_value(i));
	}
}

TEST(AssignmentTest, MoveValue3) {
	Element v1{};
	Element v2{};
	v1.zero();
	v1.get_value().data()[0] = 1;
	for (size_t i = 0; i < Element::value_size(); ++i) {
		v2.get_value().data()[i] = i;
	}

	v2 = std::move(v1);

	EXPECT_EQ(1, v2.get_value(0));

	for (size_t i = 1; i < Element::value_size(); ++i) {
		EXPECT_EQ(0, v2.get_value(i));
	}
}


int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}