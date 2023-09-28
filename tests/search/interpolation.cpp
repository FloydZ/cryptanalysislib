#include <gtest/gtest.h>
#include "search/search.h"
#include "common.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;



// source: https://medium.com/@vgasparyan1995/interpolation-search-a-generic-implementation-in-c-part-2-164d2c9f55fa
TEST(lower_bound_interpolation_search2, simple) {
	std::vector<ContainerT> data;
	ContainerT search = random_data(data);
	auto a = lower_bound_interpolation_search2(data.begin(), data.end(), search,
											   [](const ContainerT &e1) -> ContainerT {
												 return e1;
											   }
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}



TEST(InterpolationSearch, simple) {
	std::vector<ContainerT> data;
	ContainerT search = random_data(data);

	auto a = LowerBoundInterpolationSearch<ContainerT> (
			data.data(), search, 0, data.size(),
			[](const ContainerT &bla)  {
			  return bla;
			}
	);

	//for(auto &a : data)
	//	std::cout << a << " ";
	//std::cout << "\n" << search << "\n" << a << "\n";

	EXPECT_EQ(solution_index, a);
}

TEST(InterpolationSearch, iterator) {
	std::vector<ContainerT> data;
	ContainerT search = random_data(data);

	auto a = LowerBoundInterpolationSearch(
		    data.begin(), data.end(), search,
		    [](const ContainerT &bla) {
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
