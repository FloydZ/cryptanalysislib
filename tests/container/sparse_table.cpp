#include <gtest/gtest.h>
#include <vector>

#include "container/sparse_table.h"
#include "random.h"

using ::testing::InitGoogleTest;

TEST(sparse_table, all_ranges) {
    for (size_t n = 1; n <= 70; n++) {
        std::vector<uint32_t> a(n);
        for (auto &x : a) { x = cryptanalysislib::rng<uint32_t>(1000); }
        const sparse_table<uint32_t> st(a);

        for (size_t l = 0; l < n; l++) {
            uint32_t m = a[l];
            for (size_t r = l; r < n; r++) {
                if (a[r] < m) { m = a[r]; }
                EXPECT_EQ(st.query(l, r), m);
            }
        }
    }
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
