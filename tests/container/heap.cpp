#include <gtest/gtest.h>
#include <cstdio>

#include "container/heap.h"

using ::testing::InitGoogleTest;
using ::testing::Test;


template <typename T>
class TestHeap : public testing::Test {};

TYPED_TEST_SUITE_P(TestHeap);

TYPED_TEST_P(TestHeap, simple) {
    constexpr static size_t s = 127;
    Heap<TypeParam> d(s);
    for (TypeParam i = 0; i < s; i++) {
        const size_t ret = d.push(i);
        EXPECT_EQ(ret, i + 1);
    }
    
    for (TypeParam i = 0; i < s; i++) {
        TypeParam t;
        const size_t ret = d.pop(t);
        EXPECT_EQ(ret, i + 1);
    }

}

REGISTER_TYPED_TEST_SUITE_P(TestHeap, simple);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, TestHeap, MyTypes);

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
