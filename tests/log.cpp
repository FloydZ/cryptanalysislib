#include <gtest/gtest.h>

#include "log.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

using namespace cryptanalysislib;

TEST(log, simple) {
    log << "test1";
    log << logging::debug << "test2";
    log(logging::debug) << "test3";
    logging::debug << "tes4"; 
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
