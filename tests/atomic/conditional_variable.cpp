#include <gtest/gtest.h>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <vector>

#include "atomic/conditional_variable.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;
using namespace std::chrono_literals;

/// producers push numbers into a queue, consumers pop them; every number
/// must be consumed exactly once
TEST(condition_variable, producer_consumer) {
	constexpr uint32_t producers = 4, consumers = 4, per_producer = 20000;
	std::mutex m;
	condition_variable cv;
	std::deque<uint32_t> q;
	uint32_t finished = 0;
	std::vector<uint32_t> seen(producers * per_producer, 0);

	std::vector<std::thread> threads;
	for (uint32_t p = 0; p < producers; ++p) {
		threads.emplace_back([&, p]() {
			for (uint32_t i = 0; i < per_producer; ++i) {
				{
					std::lock_guard g(m);
					q.push_back(p * per_producer + i);
				}
				cv.notify_one();
			}
			{
				std::lock_guard g(m);
				finished += 1;
			}
			cv.notify_all();
		});
	}
	for (uint32_t c = 0; c < consumers; ++c) {
		threads.emplace_back([&]() {
			std::unique_lock l(m);
			while (true) {
				cv.wait(l, [&]() { return !q.empty() || finished == producers; });
				if (q.empty()) { return; }
				seen[q.front()] += 1;
				q.pop_front();
			}
		});
	}
	for (auto &t : threads) { t.join(); }

	for (size_t i = 0; i < seen.size(); ++i) {
		ASSERT_EQ(seen[i], 1u) << i;
	}
}

/// two threads alternate via a shared turn variable
TEST(condition_variable, ping_pong) {
	constexpr uint32_t rounds = 20000;
	std::mutex m;
	condition_variable cv;
	uint32_t turn = 0;
	auto f = [&](const uint32_t me) {
		for (uint32_t r = 0; r < rounds; ++r) {
			std::unique_lock l(m);
			cv.wait(l, [&]() { return (turn & 1u) == me; });
			turn += 1;
			l.unlock();
			cv.notify_one();
		}
	};
	std::thread a(f, 0u), b(f, 1u);
	a.join();
	b.join();
	EXPECT_EQ(turn, 2 * rounds);
}

TEST(condition_variable, notify_all) {
	constexpr uint32_t n = 8;
	std::mutex m;
	condition_variable cv;
	bool go = false;
	uint32_t woken = 0;
	std::vector<std::thread> threads;
	for (uint32_t t = 0; t < n; ++t) {
		threads.emplace_back([&]() {
			std::unique_lock l(m);
			cv.wait(l, [&]() { return go; });
			woken += 1;
		});
	}
	std::this_thread::sleep_for(10ms);
	{
		std::lock_guard g(m);
		go = true;
	}
	cv.notify_all();
	for (auto &t : threads) { t.join(); }
	EXPECT_EQ(woken, n);
}

TEST(condition_variable, timeout) {
	std::mutex m;
	condition_variable cv;
	std::unique_lock l(m);

	// predicate is never true
	const auto t0 = std::chrono::steady_clock::now();
	EXPECT_FALSE(cv.wait_for(l, 20ms, []() { return false; }));
	const auto dt = std::chrono::steady_clock::now() - t0;
	EXPECT_GE(dt, 20ms);
	EXPECT_LT(dt, 2s);
	EXPECT_TRUE(l.owns_lock());

	// predicate is already true
	EXPECT_TRUE(cv.wait_until(l, std::chrono::system_clock::now() - 1s, []() { return true; }));

	// notified before the timeout
	bool ready = false;
	std::thread t([&]() {
		std::this_thread::sleep_for(10ms);
		{
			std::lock_guard g(m);
			ready = true;
		}
		cv.notify_one();
	});
	EXPECT_TRUE(cv.wait_for(l, 10s, [&]() { return ready; }));
	l.unlock();
	t.join();
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
