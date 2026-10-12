#include <gtest/gtest.h>

#include "random.h"
#include "search/search.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;

using T = uint32_t;
constexpr static uint64_t SIZE = 1<<10;

// Where to search
constexpr static uint32_t k_lower = 0;
constexpr static uint32_t k_higher = 22;
constexpr static T MASK = ((T(1) << k_higher) - 1) ^ ((T(1) << k_lower) -1);
constexpr static size_t NR_SOLS = 1;

TEST(upper_bound_linear_search, simple) {
	std::vector<T> data;
	size_t solution_index;
	T search = cryptanalysislib::random_data(data, solution_index, SIZE, NR_SOLS, MASK);

	auto a = cryptanalysislib::upper_bound_linear_search(data.begin(), data.end(), search,
		[](const T &e1, const T &e2) -> bool {
			 return e1 < e2;
		}
	);

	// first element greater than `search`
	EXPECT_EQ(solution_index + NR_SOLS, distance(data.begin(), a));
}

TEST(lower_bound_linear_search, simple) {
	std::vector<T> data;
	size_t solution_index;
	T search = cryptanalysislib::random_data(data, solution_index, SIZE, NR_SOLS, MASK);

	auto a = cryptanalysislib::lower_bound_linear_search(data.begin(), data.end(), search,
		[](const T &e1, const T &e2) -> bool {
			 return e1 < e2;
		}
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(upper_bound_breaking_linear_search, simple) {
	std::vector<T> data;
	size_t solution_index;
	T search = cryptanalysislib::random_data(data, solution_index, SIZE, NR_SOLS, MASK);

	auto a = cryptanalysislib::upper_bound_breaking_linear_search(data.begin(), data.end(), search,
		[](const T &e1) -> T {
		  return e1;
		}
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(lower_bound_breaking_linear_search, simple) {
	std::vector<T> data;
	size_t solution_index;
	T search = cryptanalysislib::random_data(data, solution_index, SIZE, NR_SOLS, MASK);

	auto a = cryptanalysislib::lower_bound_breaking_linear_search(data.begin(), data.end(), search,
		[](const T &e1) -> T {
		  return e1;
		}
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(linear_search, simd) {
	for (size_t n = 0; n < 150; ++n) {
		std::vector<uint16_t> data(n);
		for (size_t i = 0; i < n; ++i) { data[i] = uint16_t(i * 3u + 1u); }
		for (size_t p = 0; p < n; ++p) {
			const uint16_t key = data[p];
			EXPECT_EQ(cryptanalysislib::internal::linear_search_simd(data.data(), n, key), p);
			EXPECT_EQ(cryptanalysislib::internal::linear_search_dispatch(data.data(), n, key), p);
			EXPECT_EQ(cryptanalysislib::linear_search(data.begin(), data.end(), key) - data.begin(), (long)p);
		}

		// not found
		EXPECT_EQ(cryptanalysislib::internal::linear_search_simd(data.data(), n, uint16_t(0)), n);
		EXPECT_EQ(cryptanalysislib::linear_search(data.begin(), data.end(), uint16_t(0)), data.end());
	}

	// non unsigned/contiguous fallback
	std::vector<int> d{5, -3, 7};
	EXPECT_EQ(cryptanalysislib::linear_search(d.begin(), d.end(), -3) - d.begin(), 1);
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
