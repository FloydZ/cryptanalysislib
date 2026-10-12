#include <gtest/gtest.h>
#include <vector>

#include "container/kd_tree.h"
#include "random.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// nearest neighbour against brute force, for the euclidean and the
/// manhattan metric, also after inserts
template<const int K, class Metric>
static void check() {
	using T = kd_tree<K, double, Metric>;
	using pt = typename T::pt;
	std::vector<pt> pts(2000);
	for (auto &p : pts) { for (int i = 0; i < K; ++i) { p.coord[i] = double(rng() % 100000) / 100.; } }
	T t(pts);
	for (int r = 0; r < 300; ++r) {
		pt q;
		for (int i = 0; i < K; ++i) { q.coord[i] = double(rng() % 100000) / 100.; }
		const pt nn = t.nearest_neighbour(q);
		double best = INFINITY;
		for (const auto &p : pts) { best = std::min(best, q.dist(p)); }
		EXPECT_NEAR(q.dist(nn), best, 1e-9) << r;
	}
	for (const auto &p : pts) { EXPECT_TRUE(t.contains(p)); }

	pt extra;
	for (int i = 0; i < K; ++i) { extra.coord[i] = -5; }
	EXPECT_FALSE(t.contains(extra));
	t.insert(extra);
	EXPECT_TRUE(t.contains(extra));
	pt q;
	for (int i = 0; i < K; ++i) { q.coord[i] = -6; }
	EXPECT_NEAR(t.nearest_neighbour(q).dist(extra), 0.0, 1e-12);
}

TEST(kd_tree, euclidean) { check<2, kd_tree_euclidean_metric>(); check<5, kd_tree_euclidean_metric>(); }
TEST(kd_tree, manhattan) { check<2, kd_tree_manhattan_metric>(); check<3, kd_tree_manhattan_metric>(); }

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
