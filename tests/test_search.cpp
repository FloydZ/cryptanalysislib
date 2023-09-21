#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "test.h"
#include "search.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using ContainerT = uint32_t;

// Size of the list to search
constexpr static uint64_t SIZE = 1<<10;
constexpr static uint64_t ssize = sizeof(ContainerT)*8; // size of the type in bits
// Where to search
constexpr static uint32_t k_lower = 0;
constexpr static uint32_t k_higher = 22;
constexpr static ContainerT mask = ((ContainerT(1) << k_higher) - 1) ^ ((ContainerT(1) << k_lower) -1);

static uint64_t solution_index = rand()%SIZE;

ContainerT random_data(std::vector<ContainerT> &data) {
	data.resize(SIZE);
	for (uint64_t i = 0; i < SIZE; ++i) {
		data[i] = fastrandombytes_uint64() & mask;
	}

	std::sort(data.begin(), data.end(),
	          [](const auto &e1, const auto &e2) {
		          return e1 < e2;
	          }
	);

	assert(std::is_sorted(data.begin(), data.end()));
	return data[solution_index];//fastrandombytes_uint64() % SIZE;
}

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

TEST(lower_bound_linear_search, simple) {
	std::vector<ContainerT> data;
	ContainerT search = random_data(data);

	auto a = lower_bound_linear_search(data.begin(), data.end(), search,
       [](const ContainerT &e1, const ContainerT &e2) -> bool {
           return e1 == e2;
       }
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(lower_bound_breaking_linear_search, simple) {
	std::vector<ContainerT> data;
	ContainerT search = random_data(data);

	auto a = lower_bound_breaking_linear_search(data.begin(), data.end(), search,
	                                            [](const ContainerT &e1) -> ContainerT {
		                                            return e1;
	                                            }
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(lower_bound_standard_binary_search, simple) {
	std::vector<ContainerT> data;
	ContainerT search = random_data(data);

	auto a = lower_bound_standard_binary_search(data.begin(), data.end(), search,
	                                            [](const ContainerT &e1) -> ContainerT {
		                                            return e1;
	                                            }
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(lower_bound_monobound_binary_search, simple) {
	std::vector<ContainerT> data;
	ContainerT search = random_data(data);


	const auto b = monobound_binary_search(data.data(), data.size(), search);
	auto a = lower_bound_monobound_binary_search(data.begin(), data.end(), search,
	     [](const ContainerT &e1) -> ContainerT {
		     return e1;
	     }
	);

	EXPECT_EQ(solution_index, b);
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

	auto a = LowerBoundInterpolationSearch (
	        data.begin(), data.end(), search,
			[](const ContainerT &bla)  {
				return bla;
			}
	);

	//for(auto &a : data)
	//	std::cout << a << " ";
	//std::cout << "\n" << search << "\n" << std::distance(data.begin(), a) << "\n";

	EXPECT_EQ(solution_index, std::distance(data.begin(), a));
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	srand(time(NULL));
	random_seed(time(NULL));
    return RUN_ALL_TESTS();
}
