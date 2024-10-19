#include <gtest/gtest.h>
#include <cstdint>

#include "random.h"
#include "helper.h"
#include "sort/sort.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;

constexpr size_t listsize = 128;

template<typename T>
void generate_list(std::vector<T> &data,
				   const size_t len) noexcept {
	data.resize(len);
	for (size_t i = 0; i < len; ++i) {
		data[i] = rng();
	}
}

TEST(CountingSort, u8) {
	constexpr size_t s = 10000;
	using T = uint8_t;
	std::vector<T> data;
	generate_list(data, s);

	

	for (size_t i = 0; i < listsize-1; ++i) {
		ASSERT_LE(data[i], data[i+1]);
	}
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}




