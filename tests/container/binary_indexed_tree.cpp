#include <gtest/gtest.h>
#include <vector>

#include "container/binary_indexed_tree.h"
#include "random.h"

using ::testing::InitGoogleTest;

// naive prefix sum over the one-based array a[1..i]
static int64_t prefix(const std::vector<int64_t> &a, const size_t i) {
    int64_t s = 0;
    for (size_t k = 1; k <= i; k++) { s += a[k]; }
    return s;
}

TEST(BIT, point_update_prefix_query) {
    for (int ctor = 0; ctor < 2; ctor++) {
        constexpr size_t n = 200;
        std::vector<int64_t> init(n), a(n + 1, 0);
        for (size_t i = 0; i < n; i++) { init[i] = (int64_t)cryptanalysislib::rng<uint32_t>(100) - 50; }
        if (ctor) { for (size_t i = 0; i < n; i++) { a[i + 1] = init[i]; } }

        BIT<int64_t> b = ctor ? BIT<int64_t>(init) : BIT<int64_t>(n);
        for (int op = 0; op < 2000; op++) {
            const size_t pos = 1 + cryptanalysislib::rng<size_t>(n);
            const int64_t v = (int64_t)cryptanalysislib::rng<uint32_t>(100) - 50;
            b.update(pos, v);
            a[pos] += v;
            const size_t q = cryptanalysislib::rng<size_t>(n + 1);
            EXPECT_EQ(b.query(q), prefix(a, q));
        }
    }
}

TEST(rangeBIT, range_update_prefix_query) {
    for (int ctor = 0; ctor < 2; ctor++) {
        constexpr size_t n = 150;
        std::vector<int64_t> init(n), a(n + 1, 0);
        for (size_t i = 0; i < n; i++) { init[i] = (int64_t)cryptanalysislib::rng<uint32_t>(100) - 50; }
        if (ctor) { for (size_t i = 0; i < n; i++) { a[i + 1] = init[i]; } }

        rangeBIT<int64_t> b = ctor ? rangeBIT<int64_t>(init) : rangeBIT<int64_t>(n);
        for (int op = 0; op < 2000; op++) {
            const int64_t v = (int64_t)cryptanalysislib::rng<uint32_t>(100) - 50;
            if (op & 1) {
                size_t i = 1 + cryptanalysislib::rng<size_t>(n), j = 1 + cryptanalysislib::rng<size_t>(n);
                if (i > j) { const size_t t = i; i = j; j = t; }
                b.rupdate(i, j, v);
                for (size_t k = i; k <= j; k++) { a[k] += v; }
            } else {
                const size_t i = 1 + cryptanalysislib::rng<size_t>(n);
                b.pupdate(i, v);
                a[i] += v;
            }
            const size_t q = cryptanalysislib::rng<size_t>(n + 1);
            EXPECT_EQ(b.query(q), prefix(a, q));
        }
    }
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
