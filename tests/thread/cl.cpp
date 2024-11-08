#include <gtest/gtest.h>
#include "thread/execution.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;

#ifdef USE_OPENCL
TEST(OpenCL, Simple) {
    cryptanalysislib::opencl_policy cl{};
}
#endif

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
