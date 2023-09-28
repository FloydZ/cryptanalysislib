#include <gtest/gtest.h>
#include "search/search.h"
#include "common.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;


TEST(lower_bound_linear_search, simple) {
	std::vector<ContainerT> data;
	size_t solution_index;
	T search = random_data(data, solution_index, SIZE, MASK);

	auto a = lower_bound_linear_search(data.begin(), data.end(), search,
									   [](const ContainerT &e1, const ContainerT &e2) -> bool {
										 return e1 == e2;
									   }
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(lower_bound_breaking_linear_search, simple) {
	std::vector<ContainerT> data;
	size_t solution_index;
	T search = random_data(data, solution_index, SIZE, MASK);

	auto a = lower_bound_breaking_linear_search(data.begin(), data.end(), search,
												[](const ContainerT &e1) -> ContainerT {
												  return e1;
												}
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
