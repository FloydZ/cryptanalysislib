#include <gtest/gtest.h>
#include <cstdint>

#include "random.h"
#include "helper.h"
#include "sort/sort.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;

constexpr size_t listsize = 128;

/// generate some random data
template<typename T>
void generate_list(std::vector<T> &data,
				   const size_t len) noexcept {
	data.resize(len);
	for (size_t i = 0; i < len; ++i) {
		data[i] = rng();
	}
}

// TODO make tests generic over T
using namespace cryptanalysislib;

TEST(pluggable_sort, std_sort_u8) {
	constexpr size_t s = 10000;
	using T = uint8_t;
	std::vector<T> data;
	generate_list(data, s);

    pluggable_sort(par_if(true), data.begin(), data.end(), std::sort);
	for (size_t i = 0; i < listsize-1; ++i) {
		EXPECT_LE(data[i], data[i+1]);
	}
}

TEST(pluggable_mergesort, std_sort_u8) {
	constexpr size_t s = 10000;
	using T = uint8_t;
	std::vector<T> data;
	generate_list(data, s);

    pluggable_mergesort(par_if(true), data.begin(), data.end(), std::sort);
	for (size_t i = 0; i < listsize-1; ++i) {
		EXPECT_LE(data[i], data[i+1]);
	}
}

TEST(pluggable_quicksort, std_sort_u8) {
	constexpr size_t s = 10000;
	using T = uint8_t;
	std::vector<T> data;
	generate_list(data, s);

    pluggable_quicksort(par_if(true), data.begin(), data.end(), std::sort);
	for (size_t i = 0; i < listsize-1; ++i) {
		EXPECT_LE(data[i], data[i+1]);
	}
}

TEST(ips4o, kek) {
	constexpr size_t s = 10000;
	using T = uint32_t;
	std::vector<T> data;
	generate_list(data, s);
    ips4o::sort(data.begin(), data.end());

	for (size_t i = 0; i < listsize-1; ++i) {
		EXPECT_LE(data[i], data[i+1]);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}




