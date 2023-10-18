#include <gtest/gtest.h>
#include "search/binary.h"
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

/// TODO tests mit meherer loesung, und dann schauen ob oberes und unteres limit erreich werden


TEST(upper_bound_standard_binary_search, simple) {
	std::vector<T> data;
	size_t solution_index;
	const T search = random_data(data, solution_index, SIZE, MASK);

	auto a = upper_bound_standard_binary_search(data.begin(), data.end(), search,
												[](const T &e1) -> T {
												  return e1 & MASK;
												}
	);

	EXPECT_EQ(solution_index, std::distance(data.begin(), a));
}

TEST(lower_bound_standard_binary_search, simple) {
	std::vector<T> data;
	size_t solution_index;
	const T search = random_data(data, solution_index, SIZE, MASK);

	auto a = lower_bound_standard_binary_search(data.begin(), data.end(), search,
												[](const T &e1) -> T {
												  return e1 & MASK;
												}
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(upper_bound_monobound_binary_search, simple) {
	std::vector<T> data;
	size_t solution_index;
	const T search = random_data(data, solution_index, SIZE, MASK);

	const auto b = monobound_binary_search(data.data(), data.size(), search);
	auto a = lower_bound_monobound_binary_search(data.begin(), data.end(), search,
												 [](const T &e1) -> T {
												   return e1 & MASK;
												 }
	);

	EXPECT_EQ(solution_index, b);
	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

TEST(lower_bound_monobound_binary_search, simple) {
	std::vector<T> data;
	size_t solution_index;
	T search = random_data(data, solution_index, SIZE, MASK);

	const auto b = monobound_binary_search(data.data(), data.size(), search);
	auto a = lower_bound_monobound_binary_search(data.begin(), data.end(), search,
	     [](const T &e1) -> T {
		     return e1 & MASK;
	     }
	);

	EXPECT_EQ(solution_index, b);
	EXPECT_EQ(solution_index, distance(data.begin(), a));
}


TEST(tripletapped_binary_search, simple) {
	std::vector<T> data;
	size_t solution_index;
	T search = random_data(data, solution_index, SIZE);

	/// NOTE MASK not working
	size_t a = tripletapped_binary_search(data.data(), SIZE, search);
	EXPECT_EQ(solution_index, a);
}

TEST(monobound_quaternary_search, simple) {
	std::vector<T> data;
	size_t solution_index;
	T search = random_data(data, solution_index, SIZE);

	/// note mask not working
	size_t a = monobound_quaternary_search(data.data(), SIZE, search);
	EXPECT_EQ(solution_index, a);
}

TEST(branchless_lower_bound, simple) {
	std::vector<T> data;
	size_t solution_index;
	T search = random_data(data, solution_index, SIZE, MASK);

	auto a = branchless_lower_bound(data.begin(), data.end(), search,
												 [](const T &e1, const T &e2) -> T {
												   return (e1&MASK) < (e2&MASK);
												 }
	);

	EXPECT_EQ(solution_index, distance(data.begin(), a));
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);

	srand(time(NULL));
	xorshf96_random_seed(rand());
    return RUN_ALL_TESTS();
}
