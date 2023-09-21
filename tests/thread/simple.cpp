#include <gtest/gtest.h>


using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;


TEST(Thread, Simple) {
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
