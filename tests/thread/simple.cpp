#include <gtest/gtest.h>
#include <chrono>
#include <future>
#include <thread>

#include "thread/thread.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;


TEST(Thread, Simple) {
}

using namespace std::chrono_literals;

TEST(SimpleScheduler, clear_tasks_wakes_waiter) {
	cryptanalysislib::SimpleScheduler pool(1);
	pool.pause();
	for (int i = 0; i < 3; i++) { pool.submit_detach([] {}); }
	auto w = std::async(std::launch::async, [&] { pool.wait_for_queued_tasks(); });
	std::this_thread::sleep_for(20ms);
	pool.clear_tasks();
	ASSERT_EQ(w.wait_for(2s), std::future_status::ready);
}

TEST(SimpleScheduler, two_waiters) {
	cryptanalysislib::SimpleScheduler pool(1);
	pool.submit_detach([] { std::this_thread::sleep_for(200ms); });
	std::this_thread::sleep_for(20ms);
	// waits for the running task
	auto b = std::async(std::launch::async, [&] { pool.wait_for_tasks(); });
	std::this_thread::sleep_for(20ms);
	// the queue is empty, returns immediately
	pool.wait_for_queued_tasks();
	ASSERT_EQ(b.wait_for(2s), std::future_status::ready);
}

TEST(SimpleScheduler, child_task_at_destruction) {
	std::atomic<bool> child_ran{false};
	{
		cryptanalysislib::SimpleScheduler pool(1);
		pool.submit_detach([&] {
			std::this_thread::sleep_for(50ms);
			pool.submit_detach([&] { child_ran = true; });
		});
		std::this_thread::sleep_for(10ms);
	}
	EXPECT_TRUE(child_ran.load());
}

/// more tasks than the queue holds: the others run in the submitting thread
TEST(SimpleScheduler, full_queue) {
	using S = cryptanalysislib::SimpleScheduler<>;
	S pool(1);
	pool.pause();
	std::atomic<size_t> ctr{0};
	const size_t n = S::queue_size + 100;
	for (size_t i = 0; i < n; i++) { pool.submit_detach([&] { ctr++; }); }
	EXPECT_EQ(pool.get_num_queued_tasks(), S::queue_size);
	EXPECT_EQ(ctr.load(), 100u);

	pool.unpause();
	pool.wait_for_tasks();
	EXPECT_EQ(ctr.load(), n);

	// futures of tasks executed in the calling thread are ready
	pool.pause();
	for (size_t i = 0; i < S::queue_size; i++) { pool.submit_detach([] {}); }
	auto f = pool.submit([] { return 42; });
	EXPECT_EQ(f.wait_for(0s), std::future_status::ready);
	EXPECT_EQ(f.get(), 42);
	pool.clear_tasks();
	EXPECT_EQ(pool.get_num_queued_tasks(), 0u);
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
