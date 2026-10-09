#include <gtest/gtest.h>

#include "log.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

using namespace cryptanalysislib;

TEST(log, simple) {
    // NOTE: qualified, `log` alone is ambiguous with ::log from <cmath>
    cryptanalysislib::log << "test1";
    cryptanalysislib::log << logging::debug << "test2";
    cryptanalysislib::log(logging::debug) << "test3";
    logging::debug << "tes4"; 
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
