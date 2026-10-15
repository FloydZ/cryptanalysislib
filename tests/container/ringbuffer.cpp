#include <gtest/gtest.h>
#include <cstdio>

#include "container/ringbuffer.h"

using ::testing::InitGoogleTest;
using ::testing::Test;


template <typename T>
class TestRingBuffer : public testing::Test {};

TYPED_TEST_SUITE_P(TestRingBuffer);

TYPED_TEST_P(TestRingBuffer, simple) {
    constexpr static size_t s = 127;
    RingBuffer<TypeParam> d(s);
    EXPECT_EQ(d.capacity(), s);
    for (size_t i = 0; i < s; i++) {
        d.insert(TypeParam(i));
        EXPECT_EQ(d.size(), i + 1);
    }

    for (size_t k = 0; k < s; k++) {
        TypeParam t;
        // read returns k+1 on success
        EXPECT_EQ(d.read(k, t), k + 1);
        EXPECT_EQ(t, TypeParam(k));
    }

    TypeParam t;
    EXPECT_EQ(d.read(s, t), 0);
}

TYPED_TEST_P(TestRingBuffer, overwrite) {
    // inserting into a full buffer overwrites the oldest entry
    constexpr static size_t s = 127, extra = 50;
    RingBuffer<TypeParam> d(s);
    for (size_t i = 0; i < s + extra; i++) {
        d.insert(TypeParam(i));
    }

    EXPECT_EQ(d.size(), s);
    for (size_t k = 0; k < s; k++) {
        TypeParam t;
        EXPECT_EQ(d.read(k, t), k + 1);
        EXPECT_EQ(t, TypeParam(k + extra));
    }
}

REGISTER_TYPED_TEST_SUITE_P(TestRingBuffer, simple, overwrite);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, TestRingBuffer, MyTypes);

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
