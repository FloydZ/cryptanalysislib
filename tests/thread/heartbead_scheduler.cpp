#include <gtest/gtest.h>
#include "thread/heartbeat_scheduler.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace cryptanalysislib;

using Task = HeartbeatScheduler<>::Task;

int test(int a) {
	// (void )t;
	std::cout << "kekw:" << a <<std::endl;
	return a;
}

TEST(Heartbeat, Simple) {
	HeartbeatScheduler s;
	s.call<int>(test, 10);
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
