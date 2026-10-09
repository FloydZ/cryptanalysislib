#include <gtest/gtest.h>
#include <vector>

#include "container/segment_tree.h"
#include "random.h"

using ::testing::InitGoogleTest;

TEST(SegmentTree, update_query) {
    constexpr static size_t n = 256;
    SegmentTree<uint64_t, n> st;
    std::vector<uint64_t> a(n, 0);

    for (int op = 0; op < 5000; op++) {
        const size_t pos = cryptanalysislib::rng<size_t>(n);
        const uint64_t v = cryptanalysislib::rng<uint64_t>(1000);
        st.update(pos, v);
        a[pos] = v;

        size_t l = cryptanalysislib::rng<size_t>(n + 1), r = cryptanalysislib::rng<size_t>(n + 1);
        if (l > r) { const size_t t = l; l = r; r = t; }
        uint64_t s = 0;
        for (size_t k = l; k < r; k++) { s += a[k]; }
        EXPECT_EQ(st.query(l, r), s);
    }
}

TEST(SegmentTree, set_build) {
    constexpr static size_t n = 64;
    SegmentTree<uint32_t, n> st;
    uint32_t s = 0;
    for (size_t i = 0; i < n; i++) {
        st.set(i, uint32_t(i));
        s += uint32_t(i);
    }
    st.build();
    EXPECT_EQ(st.query(0, n), s);
    EXPECT_EQ(st.query(10, 20), 145u);
}

TEST(lazy_segment_tree, range_add_point_set_min) {
    for (size_t n = 1; n <= 64; n++) {
        std::vector<int64_t> a(n);
        for (auto &x : a) { x = (int64_t)cryptanalysislib::rng<uint32_t>(1000) - 500; }
        lazy_segment_tree<int64_t> st(a);

        for (int op = 0; op < 300; op++) {
            int l = (int)cryptanalysislib::rng<size_t>(n), r = (int)cryptanalysislib::rng<size_t>(n);
            if (l > r) { const int t = l; l = r; r = t; }
            const int64_t v = (int64_t)cryptanalysislib::rng<uint32_t>(100) - 50;
            switch (op % 3) {
                case 0:
                    st.range_update(l, r, v);
                    for (int k = l; k <= r; k++) { a[k] += v; }
                    break;
                case 1:
                    st.update(l, v);
                    a[l] = v;
                    break;
                default: {
                    int64_t m = a[l];
                    for (int k = l; k <= r; k++) { if (a[k] < m) { m = a[k]; } }
                    const auto res = st.query(l, r);
                    EXPECT_TRUE(res.valid);
                    EXPECT_EQ(res.x, m);
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
