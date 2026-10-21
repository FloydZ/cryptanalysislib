#include <algorithm>
#include <vector>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "random.h"
#include "algorithm/partition.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

constexpr size_t N = 1u << 20u; // above `min_size_per_thread`

TEST(partition, is_partitioned_parallel) {
	const auto p = [](const uint32_t x) { return x < 1000u; };
	for (const size_t split : {size_t(0), size_t(1), N / 3, N - 1, N}) {
		std::vector<uint32_t> v(N);
		for (size_t i = 0; i < N; ++i) { v[i] = i < split ? uint32_t(rng() % 1000u) : 1000u + uint32_t(rng() % 1000u); }
		EXPECT_TRUE(cryptanalysislib::is_partitioned(par_if(true), v.begin(), v.end(), p)) << split;
		if (split + 7 < N) {
			v[split + 7] = 5; // a `true` element after the first `false` one
			EXPECT_FALSE(cryptanalysislib::is_partitioned(par_if(true), v.begin(), v.end(), p)) << split;
		}
	}
}

TEST(partition, partition_copy_parallel) {
	std::vector<uint32_t> v(N);
	for (auto &x : v) { x = rng<uint32_t>(); }
	const auto p = [](const uint32_t x) { return (x & 3u) == 1u; };
	std::vector<uint32_t> t(N), f(N), et(N), ef(N);
	const auto r = cryptanalysislib::partition_copy(par_if(true), v.begin(), v.end(), t.begin(), f.begin(), p);
	const auto e = std::partition_copy(v.begin(), v.end(), et.begin(), ef.begin(), p);
	EXPECT_EQ(r.first - t.begin(), e.first - et.begin());
	EXPECT_EQ(r.second - f.begin(), e.second - ef.begin());
	EXPECT_EQ(t, et);
	EXPECT_EQ(f, ef);
}

TEST(partition, stable_partition_parallel) {
	// (key, original index): stability is visible in the index order
	std::vector<std::pair<uint32_t, uint32_t>> v(N);
	for (size_t i = 0; i < N; ++i) { v[i] = {uint32_t(rng() % 10u), uint32_t(i)}; }
	auto e = v;
	const auto p = [](const auto &x) { return x.first < 3u; };
	const auto r = cryptanalysislib::stable_partition(par_if(true), v.begin(), v.end(), p);
	const auto re = std::stable_partition(e.begin(), e.end(), p);
	EXPECT_EQ(r - v.begin(), re - e.begin());
	EXPECT_EQ(v, e);
}

/// sequential partition functions against std::
TEST(partition, sequential) {
	const auto p = [](const uint32_t x) { return x % 3u == 0; };
	for (const size_t s : {size_t(0), size_t(1), size_t(2), size_t(17), size_t(1000)}) {
		std::vector<uint32_t> v(s);
		for (size_t i = 0; i < s; ++i) { v[i] = (uint32_t)(i * 2654435761u) % 101u; }
		EXPECT_EQ(cryptanalysislib::is_partitioned(v.begin(), v.end(), p), std::is_partitioned(v.begin(), v.end(), p));

		// partition: a valid partition point and the same multiset
		std::vector<uint32_t> w = v;
		const auto r = cryptanalysislib::partition(w.begin(), w.end(), p);
		EXPECT_EQ(r - w.begin(), std::count_if(v.begin(), v.end(), p));
		EXPECT_TRUE(std::is_partitioned(w.begin(), w.end(), p));
		EXPECT_TRUE(std::is_permutation(v.begin(), v.end(), w.begin()));
		EXPECT_TRUE(cryptanalysislib::is_partitioned(w.begin(), w.end(), p));
		EXPECT_EQ(cryptanalysislib::partition_point(w.begin(), w.end(), p), r);

		// stable_partition: same as std::stable_partition
		std::vector<uint32_t> a = v, b = v;
		const auto ra = cryptanalysislib::stable_partition(a.begin(), a.end(), p);
		const auto rb = std::stable_partition(b.begin(), b.end(), p);
		EXPECT_EQ(ra - a.begin(), rb - b.begin());
		EXPECT_EQ(a, b);

		std::vector<uint32_t> t(s), f(s), te(s), fe(s);
		const auto rc = cryptanalysislib::partition_copy(v.begin(), v.end(), t.begin(), f.begin(), p);
		const auto re = std::partition_copy(v.begin(), v.end(), te.begin(), fe.begin(), p);
		EXPECT_EQ(rc.first - t.begin(), re.first - te.begin());
		EXPECT_EQ(rc.second - f.begin(), re.second - fe.begin());
		EXPECT_EQ(t, te);
		EXPECT_EQ(f, fe);
	}
}

/// SIMD `partition_copy_less` (uint32_t) and the generic fallback against std::
TEST(partition, partition_copy_less) {
	for (const size_t s : {size_t(0), size_t(1), size_t(7), size_t(8), size_t(15), size_t(16),
	                       size_t(17), size_t(31), size_t(100), size_t(1000)}) {
		for (const uint32_t pivot : {0u, 1u, 1u << 31u, 0xffffffffu, rng<uint32_t>()}) {
			std::vector<uint32_t> v(s);
			for (auto &x : v) { x = rng<uint32_t>(); }
			if (s > 3) { v[1] = pivot; v[2] = 0; v[3] = 0xffffffffu; }

			const auto p = [pivot](const uint32_t x) { return x < pivot; };
			std::vector<uint32_t> t(s), f(s), te(s), fe(s);
			const auto r = cryptanalysislib::partition_copy_less(v.begin(), v.end(), t.begin(), f.begin(), pivot);
			const auto e = std::partition_copy(v.begin(), v.end(), te.begin(), fe.begin(), p);
			EXPECT_EQ(r.first - t.begin(), e.first - te.begin()) << s << " " << pivot;
			EXPECT_EQ(r.second - f.begin(), e.second - fe.begin()) << s << " " << pivot;
			EXPECT_EQ(t, te) << s << " " << pivot;
			EXPECT_EQ(f, fe) << s << " " << pivot;

			// generic (non SIMD) version
			std::vector<uint64_t> v64(v.begin(), v.end()), t64(s), f64(s);
			const auto r64 = cryptanalysislib::partition_copy_less(v64.begin(), v64.end(), t64.begin(), f64.begin(), uint64_t(pivot));
			EXPECT_EQ(r64.first - t64.begin(), e.first - te.begin());
			for (size_t i = 0; i < size_t(r64.first - t64.begin()); ++i) { EXPECT_EQ(t64[i], te[i]); }
		}
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
