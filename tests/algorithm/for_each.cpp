#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
// NOTE: no `<execution>`: if `libtbb-dev` is installed, libstdc++ selects its
// TBB pstl backend, which needs `-ltbb` (undefined `tbb::detail::r1::*` at -O0).

#include "algorithm/for_each.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using namespace cryptanalysislib;


// the execution policies satisfy the concept, other types do not
static_assert(cryptanalysislib::ExecutionPolicy<decltype(cryptanalysislib::seq)>);
static_assert(cryptanalysislib::ExecutionPolicy<decltype(cryptanalysislib::par)>);
static_assert(cryptanalysislib::ExecutionPolicy<cryptanalysislib::pure_threads_policy &>);
static_assert(!cryptanalysislib::ExecutionPolicy<int>);

TEST(for_each, single) {
	using T = uint64_t;
	constexpr size_t size = 10;
	std::vector<T> test;
    test.resize(size);
	for (size_t i = 0; i < size; ++i) { test[i] = i; }

	// 	- number of threads
	//  - dynamic threading on/off
	//  see:https://github.com/alugowski/poolSTL/blob/26d95b90aea7c36732a2df50df1c6fa26c96f93e/tests/poolstl_test.cpp#L223
	cryptanalysislib::for_each(par_if(true), test.begin(), test.end(), [](auto &in){
		const auto t = in * in;
		in = t;
	});

	for (size_t i = 0; i < size; ++i) {
		std::cout << test[i] << std::endl;
		EXPECT_EQ(test[i], i*i);
	}
}

/// every element is visited exactly once, the chunk data is created per chunk
TEST(for_each, chunk) {
	constexpr size_t size = 1u << 18u; // above `min_size_per_thread`
	for (const bool par : {false, true}) {
		std::vector<uint64_t> v(size, 0);
		std::atomic<uint32_t> constructed{0};
		cryptanalysislib::for_each_chunk(par_if(par), v.begin(), v.end(),
			[&constructed]() { constructed += 1; return uint64_t(1); },
			[](uint64_t &x, uint64_t &chunk_data) { x += 1; chunk_data += 1; });

		for (size_t i = 0; i < size; ++i) { EXPECT_EQ(v[i], 1u) << i; }
		if (par) { EXPECT_GE(constructed.load(), 1u); }
		else     { EXPECT_EQ(constructed.load(), 1u); }
	}

	// sequential overload without policy
	std::vector<uint64_t> w(100, 0);
	cryptanalysislib::for_each_chunk(w.begin(), w.end(), []() { return uint64_t(0); },
		[](uint64_t &x, uint64_t &c) { x = c++; });
	for (size_t i = 0; i < w.size(); ++i) { EXPECT_EQ(w[i], i); }
}

/// sequential `for_each` returns the functor, parallel `for_each_n` visits exactly n elements
TEST(for_each, sequential_and_n) {
	std::vector<uint64_t> v(1000);
	for (size_t i = 0; i < v.size(); ++i) { v[i] = i; }
	struct Sum {
		uint64_t s = 0;
		void operator()(uint64_t &x) { s += x; x += 1; }
	};
	const Sum r = cryptanalysislib::for_each(v.begin(), v.end(), Sum{});
	EXPECT_EQ(r.s, 999u * 1000u / 2u);
	for (size_t i = 0; i < v.size(); ++i) { ASSERT_EQ(v[i], i + 1u); }

	for (const size_t n : {size_t(0), size_t(1), size_t(500), size_t(1u << 20)}) {
		std::vector<uint32_t> w(n + 4, 0u);
		auto it = cryptanalysislib::for_each_n(par_if(true), w.begin(), n, [](uint32_t &x) { x += 3u; });
		EXPECT_EQ(it, w.begin() + n);
		for (size_t i = 0; i < w.size(); ++i) { ASSERT_EQ(w[i], i < n ? 3u : 0u) << n << " " << i; }
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
