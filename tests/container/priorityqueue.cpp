#include <gtest/gtest.h>
#include <vector>

#include "container/priorityqueue.h"
#include "random.h"

using ::testing::InitGoogleTest;

// reference: extract the minimum of an unsorted vector
static uint32_t extract_min(std::vector<uint32_t> &v) {
    size_t m = 0;
    for (size_t i = 1; i < v.size(); i++) { if (v[i] < v[m]) { m = i; } }
    const uint32_t r = v[m];
    v[m] = v.back();
    v.pop_back();
    return r;
}

TEST(priority_queue, fixed_size) {
    constexpr static size_t s = 500;
    priority_queue<uint32_t, uint32_t> pq(s);
    std::vector<uint32_t> ref;
    for (size_t i = 0; i < s; i++) {
        const uint32_t t = cryptanalysislib::rng<uint32_t>(1000);
        // the event stores the time, so both can be checked
        EXPECT_TRUE(pq.insert(t, t));
        ref.push_back(t);
    }
    // full and growing disabled
    EXPECT_FALSE(pq.insert(0, 0));
    EXPECT_EQ(pq.num(), s);

    for (size_t i = 0; i < s; i++) {
        uint32_t t, e;
        EXPECT_TRUE(pq.extract_next(t, e));
        EXPECT_EQ(t, extract_min(ref));
        EXPECT_EQ(e, t);
    }
    uint32_t t, e;
    EXPECT_FALSE(pq.extract_next(t, e));
}

TEST(priority_queue, grow) {
    priority_queue<uint32_t, uint32_t> pq(4, 3);
    std::vector<uint32_t> ref;
    for (int op = 0; op < 5000; op++) {
        if (ref.empty() || cryptanalysislib::rng<uint32_t>(3)) {
            const uint32_t t = cryptanalysislib::rng<uint32_t>(1000);
            EXPECT_TRUE(pq.insert(t, t + 1));
            ref.push_back(t);
        } else {
            uint32_t t, e;
            EXPECT_TRUE(pq.get_next_t(t));
            EXPECT_TRUE(pq.extract_next(t, e));
            EXPECT_EQ(t, extract_min(ref));
            EXPECT_EQ(e, t + 1);
        }
        EXPECT_EQ(pq.num(), ref.size());
    }
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
