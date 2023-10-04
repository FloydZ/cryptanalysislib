#include <gtest/gtest.h>
#include "search/search.h"
#include "common.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;

using T = uint32_t;
constexpr static uint64_t SIZE = 1<<10;

// Where to search
constexpr static uint32_t k_lower = 0;
constexpr static uint32_t k_higher = 22;
constexpr static T MASK = ((T(1) << k_higher) - 1) ^ ((T(1) << k_lower) -1);

TEST(lower_bound_linear_search, simple) {
	std::vector<T> data;
	size_t solution_index;
	T search = random_data(data, solution_index, SIZE, MASK);

	auto a = lower_bound_linear_search(data.begin(), data.end(), search,
									   [](const T &e1, const T &e2) -> bool {
										 return e1 == e2;
									   }
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(lower_bound_breaking_linear_search, simple) {
	std::vector<T> data;
	size_t solution_index;
	T search = random_data(data, solution_index, SIZE, MASK);

	auto a = lower_bound_breaking_linear_search(data.begin(), data.end(), search,
												[](const T &e1) -> T {
												  return e1;
												}
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
