#include <gtest/gtest.h>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "atomic/barrier.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

constexpr static uint32_t nthreads = 8;
constexpr static uint32_t rounds = 1000;

/// every thread increments the counter of the round, after the barrier all
/// threads must have arrived; the completion function runs once per round
/// and sees all arrivals
TEST(barrier, arrive_and_wait) {
	std::vector<std::atomic<uint32_t>> counter(rounds);
	std::atomic<uint32_t> completions{0};
	std::atomic<uint32_t> errors{0};
	auto f = [&]() noexcept {
		const uint32_t r = completions.load(std::memory_order_relaxed);
		if (counter[r].load(std::memory_order_relaxed) != nthreads) { errors++; }
		completions.store(r + 1, std::memory_order_relaxed);
	};
	barrier b(nthreads, f);

	std::vector<std::thread> threads;
	for (uint32_t t = 0; t < nthreads; ++t) {
		threads.emplace_back([&]() {
			for (uint32_t r = 0; r < rounds; ++r) {
				counter[r].fetch_add(1, std::memory_order_relaxed);
				b.arrive_and_wait();
				if (counter[r].load(std::memory_order_relaxed) != nthreads) { errors++; }
				if (completions.load(std::memory_order_relaxed) != r + 1) { errors++; }
			}
		});
	}
	for (auto &t : threads) { t.join(); }
	EXPECT_EQ(errors.load(), 0u);
	EXPECT_EQ(completions.load(), rounds);
}

/// `arrive` and `wait` separately, with work in between
TEST(barrier, arrive_wait) {
	std::vector<std::atomic<uint32_t>> counter(rounds);
	std::atomic<uint32_t> errors{0};
	barrier b(nthreads);

	std::vector<std::thread> threads;
	for (uint32_t t = 0; t < nthreads; ++t) {
		threads.emplace_back([&]() {
			for (uint32_t r = 0; r < rounds; ++r) {
				counter[r].fetch_add(1, std::memory_order_relaxed);
				auto token = b.arrive();
				volatile uint32_t x = 0;
				for (uint32_t i = 0; i < 100; ++i) { x = x + i; }
				b.wait(std::move(token));
				if (counter[r].load(std::memory_order_relaxed) != nthreads) { errors++; }
			}
		});
	}
	for (auto &t : threads) { t.join(); }
	EXPECT_EQ(errors.load(), 0u);
}

/// thread `t` leaves after round `t`: the following phases need one
/// arrival less
TEST(barrier, arrive_and_drop) {
	std::vector<std::atomic<uint32_t>> counter(nthreads + 2);
	std::atomic<uint32_t> errors{0};
	barrier b(nthreads + 1);

	std::vector<std::thread> threads;
	for (uint32_t t = 0; t < nthreads; ++t) {
		threads.emplace_back([&, t]() {
			for (uint32_t r = 0; r <= t; ++r) {
				counter[r].fetch_add(1, std::memory_order_relaxed);
				if (r == t) {
					b.arrive_and_drop();
				} else {
					b.arrive_and_wait();
				}
			}
		});
	}

	// the main thread takes part in all rounds: in round r, the threads
	// r..nthreads-1 and the main thread arrive
	for (uint32_t r = 0; r < nthreads + 2; ++r) {
		counter[r].fetch_add(1, std::memory_order_relaxed);
		b.arrive_and_wait();
		const uint32_t e = (r < nthreads) ? (nthreads - r + 1) : 1;
		if (counter[r].load(std::memory_order_relaxed) != e) { errors++; }
	}
	for (auto &t : threads) { t.join(); }
	EXPECT_EQ(errors.load(), 0u);
}

TEST(barrier, single_thread) {
	uint32_t c = 0;
	barrier b(1, [&c]() noexcept { c += 1; });
	for (uint32_t i = 0; i < 10; ++i) {
		b.arrive_and_wait();
	}
	EXPECT_EQ(c, 10u);
	static_assert(barrier<>::max() > 0);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
