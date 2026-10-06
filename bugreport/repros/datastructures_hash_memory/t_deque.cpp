#include <gtest/gtest.h>
#include <cstdio>

#include "container/deque.h"

using ::testing::InitGoogleTest;
using ::testing::Test;


template <typename T>
class TestDeque : public testing::Test {};

TYPED_TEST_SUITE_P(TestDeque);

TYPED_TEST_P(TestDeque, simple) {
    constexpr static size_t s = 127;
    Deque<TypeParam> d(s);
    for (TypeParam i = 0; i < s; i++) {
        const size_t ret = d.insert_last(i);
        EXPECT_EQ(ret, i + 1);
    }

    EXPECT_EQ(d.size(), s);
    EXPECT_EQ(d.capacity(), s);

    for (TypeParam i = 0; i < s; i++) {
        TypeParam t;
        const size_t ret = d.extract_first(t);
        EXPECT_EQ(ret, t);
    }
    
    EXPECT_EQ(d.size(), 0);
    
    for (TypeParam i = 0; i < s; i++) {
        const size_t ret = d.insert_last(i);
        EXPECT_EQ(ret, i + 1);
    }

    EXPECT_EQ(d.size(), s);
    EXPECT_EQ(d.capacity(), s);

    for (TypeParam i = 0; i < s; i++) {
        TypeParam t;
        const size_t ret = d.extract_last(t);
        EXPECT_EQ(ret, t);
    }
    
    EXPECT_EQ(d.size(), 0);
}

REGISTER_TYPED_TEST_SUITE_P(TestDeque, simple);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, TestDeque, MyTypes);

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	rng_seed(time(NULL));
    return RUN_ALL_TESTS();
}
