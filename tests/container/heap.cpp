#include <gtest/gtest.h>
#include <cstdio>
#include <vector>

#include "container/heap.h"
#include "random.h"

using ::testing::InitGoogleTest;
using ::testing::Test;


template <typename T>
class TestHeap : public testing::Test {};

TYPED_TEST_SUITE_P(TestHeap);

TYPED_TEST_P(TestHeap, simple) {
    constexpr static size_t s = 127;
    Heap<TypeParam> d(s);
    for (size_t i = 0; i < s; i++) {
        // number of elements after the insertion
        EXPECT_EQ(d.push(TypeParam(i)), i + 1);
        EXPECT_EQ(d.test_heap(), 0);
    }

    // full
    EXPECT_EQ(d.push(TypeParam(0)), 0);

    for (size_t i = 0; i < s; i++) {
        TypeParam t;
        // number of elements before the removal; max-heap: largest first
        EXPECT_EQ(d.pop(t), s - i);
        EXPECT_EQ(t, TypeParam(s - 1 - i));
    }

    TypeParam t;
    EXPECT_EQ(d.pop(t), 0);
}

TYPED_TEST_P(TestHeap, random) {
    constexpr static size_t s = 1000;
    Heap<TypeParam> d(s);
    std::vector<TypeParam> ref(s);
    for (size_t i = 0; i < s; i++) {
        ref[i] = cryptanalysislib::rng<TypeParam>();
        d.push(ref[i]);
    }
    EXPECT_EQ(d.test_heap(), 0);

    // reference: sort descending
    for (size_t i = 1; i < s; i++) {
        const TypeParam v = ref[i];
        size_t j = i;
        for (; j > 0 && ref[j-1] < v; j--) {
            ref[j] = ref[j-1];
        }
        ref[j] = v;
    }

    for (size_t i = 0; i < s; i++) {
        TypeParam t;
        EXPECT_EQ(d.pop(t), s - i);
        EXPECT_EQ(t, ref[i]);
    }
}

REGISTER_TYPED_TEST_SUITE_P(TestHeap, simple, random);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, TestHeap, MyTypes);

TEST(Heap2, indexed) {
    constexpr static uint32_t s = 100;
    // insert the keys in a random order
    std::vector<uint32_t> keys(s);
    for (uint32_t i = 0; i < s; i++) { keys[i] = i; }
    for (uint32_t i = s - 1; i > 0; i--) {
        const uint32_t j = cryptanalysislib::rng<uint32_t>(i + 1);
        const uint32_t t = keys[i]; keys[i] = keys[j]; keys[j] = t;
    }

    Heap2<uint32_t> h;
    for (uint32_t i = 0; i < s; i++) {
        h.push(keys[i]);
    }
    EXPECT_EQ(h.size(), s);

    // std::less: min-heap
    for (uint32_t i = 0; i < s; i++) {
        EXPECT_EQ(h.top(), i);
        EXPECT_EQ(h.pop(), i);
    }
    EXPECT_TRUE(h.empty());

    // keys can be inserted again after they were removed
    h.push(5); h.push(3); h.push(9);
    EXPECT_EQ(h.pop(), 3);
    h.push(3);
    EXPECT_EQ(h.pop(), 3);
    EXPECT_EQ(h.pop(), 5);
    EXPECT_EQ(h.pop(), 9);
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
