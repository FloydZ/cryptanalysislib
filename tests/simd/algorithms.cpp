#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>

#include "helper.h"
#include "random.h"
#include "simd/simd.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(transpose, simple) {
	constexpr size_t s = 32;
	uint8_t in[s] = {0}, out[s] = {0};
	in[0] = 1;
	in[1] = 4;
	in[2] = 1;
	in[3] = 128;
	bshuf_trans_byte_elem_SSE_16(out, in, 16);
	std::cout << "lek";
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
