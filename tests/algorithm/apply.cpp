#include <atomic>
#include <cstdint>
#include <tuple>
#include <vector>
#include <gtest/gtest.h>

#include "algorithm/apply.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

/// `parallel_apply` calls `op` once per argument tuple
TEST(apply, parallel_apply) {
	std::atomic<uint64_t> sum{0}, calls{0};
	std::vector<std::tuple<uint64_t, uint64_t>> args;
	for (uint64_t i = 0; i < 100; ++i) { args.emplace_back(i, 2u * i); }

	auto futures = cryptanalysislib::parallel_apply(cryptanalysislib::par_if(true),
	        [&sum, &calls](const uint64_t a, const uint64_t b) { sum += a + b; calls += 1; }, args);
	EXPECT_EQ(futures.size(), args.size());
	for (auto &f : futures) { f.get(); }
	EXPECT_EQ(calls.load(), 100u);
	EXPECT_EQ(sum.load(), 3u * 99u * 100u / 2u);
}

/// `should_par`: no threads for sequential policies or small problems,
/// at most the size of the pool otherwise
TEST(algorithm, should_par) {
	// `should_par` needs a config with `min_size_per_thread`
	constexpr cryptanalysislib::AlgorithmApplyConfig c;
	EXPECT_EQ(should_par(cryptanalysislib::par_if(false), c, 1ull << 30), 0u);
	EXPECT_EQ(should_par(cryptanalysislib::par_if(true), c, 0), 0u);
	EXPECT_EQ(should_par(cryptanalysislib::par_if(true), c, c.min_size_per_thread - 1u), 0u);
	const uint32_t n1 = should_par(cryptanalysislib::par_if(true), c, c.min_size_per_thread);
	EXPECT_EQ(n1, 1u);
	const uint32_t nt = cryptanalysislib::par_if(true).pool()->get_num_threads();
	const uint32_t n2 = should_par(cryptanalysislib::par_if(true), c, c.min_size_per_thread * 1000ull);
	EXPECT_GE(n2, 1u);
	EXPECT_LE(n2, nt);
	EXPECT_EQ(n2, std::min<uint32_t>(1000u, nt));
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
