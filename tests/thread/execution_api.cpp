#include <algorithm>
#include <atomic>
#include <cstdint>
#include <future>
#include <numeric>
#include <vector>
#include <gtest/gtest.h>

#include "algorithm/histogram.h"
#include "thread/execution.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace cryptanalysislib;

TEST(execution, policies) {
	auto pool = internal::get_default_pool();
	ASSERT_NE(pool, nullptr);
	EXPECT_EQ(pool, internal::get_default_pool());
	EXPECT_GT(pool->get_num_threads(), 0u);

	const auto p1 = par_if(true, *pool), p2 = par_if(false, *pool);
	EXPECT_FALSE(is_seq(p1));
	EXPECT_TRUE(is_seq(p2));
	EXPECT_EQ(p1.pool(), pool.get());
	EXPECT_TRUE(is_seq(seq));
	EXPECT_FALSE(is_seq(par_if(true)));
}

TEST(execution, chunk_helpers) {
	EXPECT_EQ(internal::get_chunk_size(10, 3), 4u);
	EXPECT_EQ(internal::get_chunk_size(9, 3), 3u);
	EXPECT_EQ(internal::get_chunk_size(0, 3), 0u);
	std::vector<int> v(10);
	EXPECT_EQ(internal::get_chunk_size(v.begin(), v.end(), 4), 3);
	EXPECT_EQ(internal::get_iter_chunk_size(v.begin() + 8, v.end(), 3), 2);
	EXPECT_EQ(internal::get_iter_chunk_size(v.begin(), v.end(), 3), 3);
	EXPECT_EQ(internal::advanced(v.begin(), 7), v.begin() + 7);
}

/// all chunk loops cover [first, last) exactly once
TEST(execution, parallel_chunk_for) {
	for (const size_t n : {size_t(0), size_t(1), size_t(7), size_t(1000), size_t(100003)}) {
		std::vector<uint64_t> v(n), w(n);
		for (size_t i = 0; i < n; ++i) { v[i] = i; w[i] = 2 * i; }
		using It = std::vector<uint64_t>::iterator;
		const uint64_t e = n ? n * (n - 1) / 2 : 0;
		auto sum = [](It b, It e2) { uint64_t s = 0; for (; b != e2; ++b) { s += *b; } return s; };

		auto f1 = internal::parallel_chunk_for_1(par_if(true), v.begin(), v.end(), sum, (uint64_t *)nullptr, 2, 0);
		uint64_t s1 = 0;
		for (auto &f : f1) { s1 += f.get(); }
		EXPECT_EQ(s1, e) << n;

		auto f2 = internal::parallel_chunk_for_2(par_if(true), v.begin(), v.end(), w.begin(),
		        [](It b, It e2, It c) { uint64_t s = 0; for (; b != e2; ++b, ++c) { s += *c - *b; } return s; },
		        (uint64_t *)nullptr, 0);
		uint64_t s2 = 0;
		for (auto &f : f2) { s2 += f.get(); }
		EXPECT_EQ(s2, e) << n;

		auto fg = internal::parallel_chunk_for_gen(par_if(true), v.begin(), v.end(), sum);
		uint64_t sg = 0;
		for (auto &f : fg) { sg += f.get(); }
		EXPECT_EQ(sg, e) << n;

		std::atomic<uint64_t> sw{0};
		internal::parallel_chunk_for_1_wait(par_if(true), v.begin(), v.end(),
		        [&sw, sum](It b, It e2) { sw += sum(b, e2); }, (void *)nullptr, 1, 4);
		EXPECT_EQ(sw.load(), e) << n;

		auto f3 = internal::parallel_chunk_for_1(par_if(true), v.begin(), v.end(), sum, (uint64_t *)nullptr, 1, 0);
		internal::get_futures(f3);
	}
}

/// the parallel algorithms called with a sequential policy must not touch
/// the (non existing) pool
TEST(execution, sequential_policy) {
	std::vector<uint8_t> d(10000);
	for (size_t i = 0; i < d.size(); ++i) { d[i] = uint8_t(i); }
	std::vector<uint32_t> c(256, 7);
	cryptanalysislib::algorithm::histogram(par_if(false), c.data(), d.data(), d.size());
	for (const auto x : c) { EXPECT_TRUE(x == 39u || x == 40u); }

	// `seq` has no pool at all (`seq.pool()` is null)
	std::fill(c.begin(), c.end(), 7u);
	cryptanalysislib::algorithm::histogram(seq, c.data(), d.data(), d.size());
	for (const auto x : c) { EXPECT_TRUE(x == 39u || x == 40u); }
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
