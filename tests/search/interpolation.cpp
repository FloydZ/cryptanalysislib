#include <cstdint>
#include <gtest/gtest.h>
#include "search/search.h"
#include "common.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;

/// TODO
constexpr static uint64_t SIZE = 1<<10;
using T = uint64_t;
using T = uint64_t;
constexpr static uint32_t k_lower = 0;
constexpr static uint32_t k_higher = 22;
constexpr static T MASK = ((T(1) << k_higher) - 1) ^ ((T(1) << k_lower) -1);


// source: https://medium.com/@vgasparyan1995/interpolation-search-a-generic-implementation-in-c-part-2-164d2c9f55fa
TEST(lower_bound_interpolation_search2, simple) {
	std::vector<T> data;
	size_t solution_index;
	T search = random_data(data, solution_index, SIZE, MASK);
	auto a = lower_bound_interpolation_search2(data.begin(), data.end(), search,
											   [](const T &e1) -> T {
												 return e1;
											   }
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}



TEST(InterpolationSearch, simple) {
	std::vector<T> data;
	size_t solution_index;
	T search = random_data(data, solution_index, SIZE, MASK);

	auto a = LowerBoundInterpolationSearch<T> (
			data.data(), search, 0, data.size(),
			[](const T &bla)  {
			  return bla;
			}
	);

	//for(auto &a : data)
	//	std::cout << a << " ";
	//std::cout << "\n" << search << "\n" << a << "\n";

	EXPECT_EQ(solution_index, a);
}

TEST(InterpolationSearch, iterator) {
	std::vector<T> data;
	size_t solution_index;
	T search = random_data(data, solution_index, SIZE, MASK);

	auto a = LowerBoundInterpolationSearch(
		    data.begin(), data.end(), search,
		    [](const T &bla) {
			    return bla;
		    });

	//for(auto &a : data)
	//	std::cout << a << " ";
	//std::cout << "\n" << search << "\n" << std::distance(data.begin(), a) << "\n";

	EXPECT_EQ(solution_index, std::distance(data.begin(), a));
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
