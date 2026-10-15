#include <gtest/gtest.h>
#include <atomic>
#include <thread>
#include <vector>

#include "atomic/atomic.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// every `signal` wakes exactly one waiter
TEST(semaphore, wait_signal) {
	semaphore_wrapper s;
	std::atomic<uint32_t> woken{0};
	std::vector<std::thread> threads;
	for (uint32_t i = 0; i < 4; ++i) {
		threads.emplace_back([&]() { s.wait(); woken.fetch_add(1); });
	}

	s.signal(2);
	while (woken.load() < 2) { std::this_thread::yield(); }
	std::this_thread::sleep_for(std::chrono::milliseconds(20));
	EXPECT_EQ(woken.load(), 2u);

	s.signal(2);
	for (auto &t : threads) { t.join(); }
	EXPECT_EQ(woken.load(), 4u);
}

/// signals before the wait are not lost
TEST(semaphore, signal_first) {
	semaphore_wrapper s;
	s.signal(3);
	s.wait(); s.wait(); s.wait();
	SUCCEED();
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
