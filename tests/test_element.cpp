#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "test.h"
#include "helper.h"
#include "element.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(ElementTest, Reference) {
	Element e{};
	e.zero();
	e.get_label(0) = 1; //THIS should actually not compile
	//e.get_label().data()[0] = 1;
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

	EXPECT_EQ(true, l1.is_equal(l2, -1));
	EXPECT_EQ(true, l2.is_equal(l3, -1));
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
	//EXPECT_EQ(true, l3.get_value().is_equal(l1.get_value()));
	EXPECT_EQ(kAryType(1), l3.get_label().data()[0]);
	EXPECT_EQ(false, l3.is_equal(l2, -1));
}


TEST(ElementComparisonTest, IsEqual) {
	Element l1{};
	Element l2{};
	l1.zero();
	l2.zero();

	/// level 0 = on all bits
	EXPECT_EQ(true, l1.is_equal(l2, -1));
	EXPECT_EQ(true, l1.is_equal(l2, 0, n));
}

TEST(ElementComparisonTest, IsGreater) {
	Element l1{};
	Element l2{};
	l1.zero();
	l2.zero();

	/// level 0 = on all bits
	EXPECT_EQ(false, l1.is_greater(l2, -1));
	for (uint64_t i = 0; i < l1.value_size(); ++i) {
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
	for (int i = 0; i < v1.value_size(); ++i) {
		v2.get_value().data()[i] = i;
	}

	v1 = v2;

	for (int i = 0; i < v1.value_size(); ++i) {
		EXPECT_EQ(i, v1.get_value(i));
	}

}

TEST(AssignmentTest, CopyValue3) {
	Element v1{};
	Element v2{};
	v1.zero();
	v1.get_value().data()[0] = 1;
	for (int i = 0; i < v1.value_size(); ++i) {
		v2.get_value().data()[i] = i;
	}

	v2 = v1;
	EXPECT_EQ(1, v2.get_value(0));

	for (int i = 1; i < v2.value_size(); ++i) {
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
	for (int i = 0; i < v1.value_size(); ++i) {
		v2.get_value().data()[i] = i;
	}

	v1 = std::move(v2);

	for (int i = 0; i < v1.value_size(); ++i) {
		EXPECT_EQ(i, v1.get_value(i));
	}

}

TEST(AssignmentTest, MoveValue3) {
	Element v1{};
	Element v2{};
	v1.zero();
	v1.get_value().data()[0] = 1;
	for (int i = 0; i < v1.value_size(); ++i) {
		v2.get_value().data()[i] = i;
	}

	v2 = std::move(v1);

	EXPECT_EQ(1, v2.get_value(0));

	for (int i = 1; i < v2.value_size(); ++i) {
		EXPECT_EQ(0, v2.get_value(i));
	}
}

TEST(AssignmentTest, CopyLabel) {
	Element v1{};
	Element v2{};

	v1.zero();
	v2.get_label().data()[0] = 1;
	v1 = v2;
	EXPECT_EQ(kAryType(1), v1.get_label(0));
}

TEST(AssignmentTest, CopyLabel2) {
	Element v1{};
	Element v2{};

	v1.zero();
	for (int i = 0; i < v1.label_size(); ++i) {
		v2.get_label().data()[i] = i;
	}

	v1 = v2;

	for (int i = 0; i < v1.label_size(); ++i) {
		EXPECT_EQ(kAryType(i), v1.get_label(i));
	}

}

TEST(AssignmentTest, CopyLabel3) {
	Element v1{};
	Element v2{};

	v1.zero();
	v1.get_label().data()[0] = kAryType(1);
	for (int i = 0; i < v1.label_size(); ++i) {
		v2.get_label().data()[i] = kAryType(i);
	}

	v2 = v1;
	EXPECT_EQ(kAryType(1), v2.get_label(0));

	for (int i = 1; i < v2.label_size(); ++i) {
		EXPECT_EQ(kAryType(0), v2.get_label(i));
	}
}


TEST(AssignmentTest, MoveLabel) {
	Element v1{};
	Element v2{};
	v1.zero();
	v2.zero();
	v2.get_label().data()[0] = kAryType(1);

	v1 = std::move(v2);
	EXPECT_EQ(kAryType(1), v1.get_label(0));
}

TEST(AssignmentTest, MoveLabel2) {
	Element v1{};
	Element v2{};
	v1.zero();
	for (int i = 0; i < v1.label_size(); ++i) {
		v2.get_label().data()[i] = i;
	}

	v1 = std::move(v2);

	for (int i = 0; i < v1.label_size(); ++i) {
		EXPECT_EQ(kAryType(i), v1.get_label(i));
	}

}

TEST(AssignmentTest, MoveLabel3) {
	Element v1{};
	Element v2{};

	v1.zero();
	v1.get_label().data()[0] = 1;
	for (int i = 0; i < v1.label_size(); ++i) {
		v2.get_label().data()[i] = i;
	}

	v2 = std::move(v1);

	EXPECT_EQ(kAryType(1), v2.get_label(0));

	for (int i = 1; i < v2.label_size(); ++i) {
		EXPECT_EQ(kAryType(0), v2.get_label(i));
	}
}



TEST(Filter, AddSimple) {
	ASSERT(q >= 2 && "wrong q for this test");
	Element l1, l2, l3;
	uint8_t norm = 2;
	uint64_t k_lower = 0;
	uint64_t k_higher = n;

	fplll::ZZ_mat<kAryType> gm;
	gm.gen_identity(n);
	static Matrix_T<fplll::ZZ_mat<kAryType>> Global_Matrix{gm};

	l1.random(gm);
	l2.random(gm);
	l3.zero();

	// this forces to be at least one coordinate to be >= 2
	l1.get_value().data()[0] = norm + 1;
	l2.get_value().data()[0] = 0;

	EXPECT_EQ(Element::add(l3, l1, l2, 0, n, norm), true);

	l1.zero();
	l2.zero();
	EXPECT_EQ(Element::add(l3, l1, l2, 0, n, norm), false);
}

TEST(Filter, AddBounds) {
	ASSERT(q >= 2 && "wrong q for this test");
	Element l1, l2, l3;
	uint8_t norm = 2;

	for (uint64_t k_lower = 1; k_lower < n; ++k_lower) {
		for (uint64_t k_higher = k_lower + 1; k_higher < n; ++k_higher) {
			// make sure we are higher than norm
			l1.get_value().data()[k_lower-1] = norm + 1;
			l2.zero();
			l3.zero();

			EXPECT_EQ(Element::add(l3, l1, l2, norm, 0, k_lower), true);
			EXPECT_EQ(Element::add(l3, l1, l2, norm, k_lower, k_higher), false);
		}
	}
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
