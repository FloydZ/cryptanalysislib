#include <gtest/gtest.h>

#include "thread/scheduler.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace cryptanalysislib;


TEST(Heartbeat, Simple) {
	HeartbeatScheduler s;

}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
