#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "compression/compression.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using Label_Type = uint64_t;

TEST(huffman, simple) {
	constexpr size_t in_size = sizeof("This is my string of data\0");
	auto data = "This is my string of data\0"_huffman;
	constexpr size_t out_size = sizeof(data);
	printf("Compressed length (%lu): %.02f%%\n", out_size, (float)out_size/in_size*100);

	for (char c : data)
		std::cout << c;
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
