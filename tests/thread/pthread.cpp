#include <gtest/gtest.h>

#include "thread/pthread.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace cryptanalysislib;


const static uint32_t c = 0;

TEST(pthread, Simple) {
	const uint32_t a = 32;
	const uint32_t b = 10;
	auto t = cryptanalysislib::pthread([&](const uint32_t a,
	                                          const uint32_t b) -> int {
		std::cout << "kekw: " << a << " " << b << " " << c << std::endl;
		return 0;
	}, a, b);

	// TODO: if you dont pass the arguments its failing
	//std::vector<cryptanalysislib::pthread> v;
	//v.emplace_back([&](const uint32_t a,
	//                   const uint32_t b) -> int {
	//  std::cout << "kekw: " << a << " " << b << " " << c << std::endl;
	//  return 0;
	//});


	t.join();
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
