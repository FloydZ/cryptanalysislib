#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "graph/digraph.h"

using ::testing::InitGoogleTest;
using G = digraph<uint32_t>;
using P = digraph_paths<uint32_t>;

static uint64_t accept_all(const P &) { return 1; }

/// brute force: number of Hamiltonian paths starting at node 0, and how
/// many of them are cycles
static std::pair<uint64_t, uint64_t> brute_force_paths(const G &g) {
	const uint32_t n = g.num_nodes();
	std::vector<uint32_t> p(n);
	for (uint32_t i = 0; i < n; i++) { p[i] = i; }
	uint64_t paths = 0, cycles = 0;
	do {
		bool ok = true;
		for (uint32_t i = 0; ok && (i + 1 < n); i++) { ok = g.has_edge(p[i], p[i + 1]); }
		if (ok) {
			paths++;
			cycles += g.has_edge(p[n - 1], p[0]);
		}
	} while (std::next_permutation(p.begin() + 1, p.end()));
	return {paths, cycles};
}

/// NOTE: the brute force enumerates all (n-1)! orders, only for small graphs
static void check_paths(G &g) {
	ASSERT_TRUE(g.OK());
	ASSERT_LE(g.num_nodes(), 9u);
	P dp(g);
	const uint64_t found = dp.all_paths(accept_all, 0, 0);
	const auto [paths, cycles] = brute_force_paths(g);
	EXPECT_EQ(found, paths);
	EXPECT_EQ(dp.num_paths(), paths);
	EXPECT_EQ(dp.num_cycles(), cycles);
}

TEST(digraph, complete) {
	G *g = G::make_complete_digraph(5);
	EXPECT_TRUE(g->OK());
	EXPECT_EQ(g->num_nodes(), 5u);
	EXPECT_EQ(g->num_edges(), 20u);
	for (uint32_t i = 0; i < 5; i++) {
		for (uint32_t j = 0; j < 5; j++) { EXPECT_EQ(g->has_edge(i, j), i != j); }
	}
	check_paths(*g);  // 4! paths, all cycles
	delete g;
}

TEST(digraph, debruijn) {
	// binary De Bruijn graph of order 3: 2 De Bruijn cycles
	G *g = G::make_debruijn_digraph(4);
	check_paths(*g);
	P dp(*g);
	dp.all_paths(accept_all, 0, 0);
	EXPECT_EQ(dp.num_cycles(), 2u);
	delete g;

	g = G::make_debruijn_digraph(3, 3);
	check_paths(*g);
	delete g;

	g = G::make_complement_shift_digraph(4);
	check_paths(*g);
	delete g;
}

TEST(digraph, gray) {
	G *g = G::make_gray_digraph(3);
	check_paths(*g);
	EXPECT_EQ(g->max_edges(), 3u);

	// the first free edge gives the binary reflected Gray code
	P dp(*g);
	EXPECT_EQ(dp.try_lucky_path(0, 0), 1u);
	EXPECT_EQ(dp.test_lucky_path(), 0u);
	for (uint32_t k = 0; k < 8; k++) { EXPECT_EQ(dp.path()[k], k ^ (k >> 1u)); }
	delete g;

	g = G::make_gray_digraph(3, true);
	EXPECT_TRUE(g->OK());
	EXPECT_EQ(g->num_edges(), 8u * 3u - 4u);
	delete g;

	g = G::make_gray_digraph(5);
	P dp5(*g);
	EXPECT_EQ(dp5.start_monotonic_gray_path(5), 10u);
	const uint32_t e[10] = {0, 1, 3, 2, 6, 4, 12, 8, 24, 16};
	for (uint32_t k = 0; k < 10; k++) { EXPECT_EQ(dp5.path()[k], e[k]); }
	delete g;
}

TEST(digraph, fibrep_mtl_paren) {
	G *g = G::make_fibrepgray_digraph(8);
	check_paths(*g);
	const uint32_t f[8] = {0, 1, 2, 4, 5, 8, 9, 10};
	for (uint32_t k = 0; k < 8; k++) { EXPECT_EQ(g->node_values()[k], f[k]); }
	delete g;

	g = G::make_mtl_digraph(3);
	EXPECT_TRUE(g->OK());
	EXPECT_EQ(g->num_nodes(), 20u);
	EXPECT_EQ(g->num_edges(), 60u);
	delete g;
	g = G::make_mtl_digraph(3, true);
	EXPECT_TRUE(g->OK());
	EXPECT_EQ(g->num_edges(), 58u);
	delete g;

	for (uint32_t pcd = 0; pcd < 3; pcd++) {
		g = G::make_parengray_digraph(4, pcd);
		EXPECT_TRUE(g->OK());
		EXPECT_EQ(g->num_nodes(), 14u);
		// symmetric, values ascending
		for (uint32_t i = 0; i < 14; i++) {
			for (uint32_t j = 0; j < 14; j++) { EXPECT_EQ(g->has_edge(i, j), g->has_edge(j, i)); }
			if (i) { EXPECT_LT(g->node_values()[i - 1], g->node_values()[i]); }
		}
		delete g;
	}
}

TEST(digraph, perm) {
	for (uint32_t n = 1; n <= 6; n++) {
		uint32_t x[32];
		for (uint64_t k = 0; k < cryptanalysislib::internal::digraph::factorial(n); k++) {
			cryptanalysislib::internal::digraph::num2perm_ffact(k, x, n);
			EXPECT_EQ(cryptanalysislib::internal::digraph::perm2num_ffact(x, n), k);
			cryptanalysislib::internal::digraph::num2perm_rfact(k, x, n);
			EXPECT_EQ(cryptanalysislib::internal::digraph::perm2num_rfact(x, n), k);
		}
	}

	// transpositions and prefix reversals are involutions: symmetric graphs
	for (const bool stq : {true, false}) {
		G *g3 = G::make_perm_gray_digraph(3, stq);
		check_paths(*g3);
		delete g3;

		G *g = G::make_perm_gray_digraph(4, stq);
		EXPECT_TRUE(g->OK());
		for (uint32_t i = 0; i < 24; i++) {
			for (uint32_t j = 0; j < 24; j++) { EXPECT_EQ(g->has_edge(i, j), g->has_edge(j, i)); }
		}
		delete g;
	}

	G *g = G::make_perm_pref_rev_digraph(3);
	check_paths(*g);
	delete g;
	g = G::make_perm_pref_rev_digraph(4);
	EXPECT_TRUE(g->OK());
	for (uint32_t i = 0; i < 24; i++) {
		for (uint32_t j = 0; j < 24; j++) { EXPECT_EQ(g->has_edge(i, j), g->has_edge(j, i)); }
	}
	delete g;

	// left and right prefix rotations are inverse
	G *l = G::make_perm_pref_rot_digraph(4, false);
	G *r = G::make_perm_pref_rot_digraph(4, true);
	EXPECT_TRUE(l->OK());
	EXPECT_TRUE(r->OK());
	for (uint32_t i = 0; i < 24; i++) {
		for (uint32_t j = 0; j < 24; j++) { EXPECT_EQ(l->has_edge(i, j), r->has_edge(j, i)); }
	}
	delete l;
	delete r;
}

TEST(digraph, sort_reverse_randomize) {
	G *g = G::make_complete_digraph(6);
	g->randomize_edge_order();
	EXPECT_TRUE(g->OK());
	g->sort_edges(1);
	EXPECT_TRUE(g->is_edge_sorted(1));
	g->reverse_edge_order();
	EXPECT_TRUE(g->is_edge_sorted(0));
	g->sort_edges(0);
	EXPECT_TRUE(g->is_edge_sorted(0));
	EXPECT_EQ(g->num_edges(), 30u);
	delete g;
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
