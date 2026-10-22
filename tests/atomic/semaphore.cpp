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

/// coroutine, which is started by the caller and destroys itself at the end
/// NOTE: the coroutine is not started eagerly (`suspend_never`), because the
/// 	ramp function of gcc 15 writes into the coroutine frame after the first
/// 	suspension, which races with a `release` in another thread (TSan).
struct detached {
	struct promise_type {
		detached get_return_object() noexcept {
			return {std::coroutine_handle<promise_type>::from_promise(*this)};
		}
		std::suspend_always initial_suspend() noexcept { return {}; }
		std::suspend_never final_suspend() noexcept { return {}; }
		void return_void() noexcept {}
		void unhandled_exception() noexcept { std::terminate(); }
	};

	std::coroutine_handle<promise_type> h;

	/// runs the coroutine until its first suspension
	void start() noexcept { h.resume(); }
};

/// atomic which counts its read-modify-write operations
static std::atomic<uint32_t> nr_rmw{0};
template<class T>
struct counting_atomic : public std::atomic<T> {
	using std::atomic<T>::atomic;

	T fetch_add(const T v, const std::memory_order o) noexcept {
		nr_rmw.fetch_add(1);
		return std::atomic<T>::fetch_add(v, o);
	}

	T fetch_sub(const T v, const std::memory_order o) noexcept {
		nr_rmw.fetch_add(1);
		return std::atomic<T>::fetch_sub(v, o);
	}
};

template<class S>
static detached acquire_and_hold(S &s, typename S::acquire_result &res, bool &done) {
	res = co_await s.acquire();
	done = true;
}

TEST(coro_semaphore, acquire_release) {
	using S = atomic::semaphore<>;
	using R = S::acquire_result;
	S s{2};
	EXPECT_EQ(s.max(), 2);
	EXPECT_EQ(s.value(), 2);

	R r1{}, r2{}, r3{};
	bool d1 = false, d2 = false, d3 = false;
	acquire_and_hold(s, r1, d1).start();
	acquire_and_hold(s, r2, d2).start();
	EXPECT_TRUE(d1 && d2);
	EXPECT_EQ(r1, R::acquired);
	EXPECT_EQ(r2, R::acquired);
	EXPECT_EQ(s.value(), 0);
	EXPECT_FALSE(s.try_acquire());

	// no resource left: suspended until `release`, which resumes it
	acquire_and_hold(s, r3, d3).start();
	EXPECT_FALSE(d3);
	s.release();
	EXPECT_TRUE(d3);
	EXPECT_EQ(r3, R::acquired);
	EXPECT_EQ(s.value(), 0);

	s.release();
	EXPECT_EQ(s.value(), 1);
	EXPECT_TRUE(s.try_acquire());
	EXPECT_EQ(s.value(), 0);
	s.release();
	s.release();
	EXPECT_EQ(s.value(), 2);
}

TEST(coro_semaphore, notify_waiters) {
	using S = atomic::semaphore<>;
	using R = S::acquire_result;
	S s{1, 0};
	EXPECT_EQ(s.value(), 0);

	constexpr uint32_t n = 4;
	R res[n];
	bool done[n] = {false};
	for (uint32_t i = 0; i < n; ++i) {
		acquire_and_hold(s, res[i], done[i]).start();
		EXPECT_FALSE(done[i]);
	}

	s.notify_waiters();
	for (uint32_t i = 0; i < n; ++i) {
		EXPECT_TRUE(done[i]);
		EXPECT_EQ(res[i], R::semaphore_stopped);
	}

	// stopped: does not suspend anymore
	R r{};
	bool d = false;
	acquire_and_hold(s, r, d).start();
	EXPECT_TRUE(d);
	EXPECT_EQ(r, R::semaphore_stopped);
	EXPECT_STREQ(S::to_string(r), "semaphore_stopped");
}

/// the waiters are resumed by the `release` of other threads, at most
/// `max` coroutines hold a resource at the same time
TEST(coro_semaphore, threads) {
	using S = atomic::semaphore<counting_atomic>;
	constexpr std::ptrdiff_t max = 3;
	constexpr uint32_t nr_threads = 8, nr_coros = 1000;
	S s{max};
	std::atomic<int64_t> in_use{0}, max_in_use{0};
	std::atomic<uint32_t> finished{0};

	auto worker = [&](S &sem) -> detached {
		const auto r = co_await sem.acquire();
		EXPECT_EQ(r, S::acquire_result::acquired);
		const int64_t u = in_use.fetch_add(1) + 1;
		int64_t m = max_in_use.load();
		while (u > m && !max_in_use.compare_exchange_weak(m, u)) {}
		in_use.fetch_sub(1);
		sem.release();
		finished.fetch_add(1);
	};

	nr_rmw = 0;
	std::vector<std::thread> threads;
	for (uint32_t t = 0; t < nr_threads; ++t) {
		threads.emplace_back([&]() {
			for (uint32_t i = 0; i < nr_coros; ++i) { worker(s).start(); }
		});
	}
	for (auto &t : threads) { t.join(); }

	EXPECT_EQ(finished.load(), nr_threads * nr_coros);
	EXPECT_LE(max_in_use.load(), max);
	EXPECT_GE(max_in_use.load(), 1);
	EXPECT_EQ(s.value(), max);
	EXPECT_GT(nr_rmw.load(), 0u);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
