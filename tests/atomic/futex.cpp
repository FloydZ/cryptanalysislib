#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "atomic/atomic.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

#ifndef __APPLE__
TEST(futex, simple) {
	cryptanalysislib::atomic::futex f{1};
	EXPECT_EQ(f.down(), 0);
	EXPECT_EQ(f.up(), 0);
}

#ifdef FUTEX2_SIZE_U8
namespace futex2 = cryptanalysislib::atomic::futex2;

/// \return now + `ms` milliseconds on CLOCK_MONOTONIC
static struct timespec in_ms(const long ms) {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	t.tv_nsec += ms * 1000000l;
	t.tv_sec += t.tv_nsec / 1000000000l;
	t.tv_nsec %= 1000000000l;
	return t;
}

/// a thread sleeps on `x` until the main thread changes and wakes it
template<typename T>
static void wait_wake() {
	alignas(8) T x = 0;
	if ((futex2::wake(&x) < 0) && (errno == EINVAL)) {
		// futex word size not implemented by the kernel (only 32 bits so far)
		EXPECT_NE(sizeof(T), 4u);
		return;
	}

	std::atomic<bool> started{false};
	std::thread t([&]() {
		started = true;
		while (__atomic_load_n(&x, __ATOMIC_ACQUIRE) == 0) {
			const int r = futex2::wait(&x, T(0));
			EXPECT_TRUE(r == 0 || errno == EAGAIN || errno == EINTR);
		}
	});
	while (!started) {}
	std::this_thread::sleep_for(std::chrono::milliseconds(10));
	__atomic_store_n(&x, T(1), __ATOMIC_RELEASE);
	EXPECT_GE(futex2::wake(&x), 0);
	t.join();
}

TEST(futex2, wait_wake) {
	if (!futex2::available()) { GTEST_SKIP() << "no futex2 (Linux 6.7+)"; }
	wait_wake<uint8_t>();
	wait_wake<uint16_t>();
	wait_wake<uint32_t>();
	wait_wake<uint64_t>();
}

TEST(futex2, errors) {
	if (!futex2::available()) { GTEST_SKIP() << "no futex2 (Linux 6.7+)"; }
	uint32_t x = 5;
	// value differs
	EXPECT_EQ(futex2::wait(&x, 4u), -1);
	EXPECT_EQ(errno, EAGAIN);
	// timeout
	const struct timespec t = in_ms(5);
	EXPECT_EQ(futex2::wait(&x, 5u, ~0ull, &t), -1);
	EXPECT_EQ(errno, ETIMEDOUT);
	// nobody waits
	EXPECT_EQ(futex2::wake(&x), 0);
}

/// a waiter with mask 0b01 is not woken by a wake with mask 0b10
TEST(futex2, mask) {
	if (!futex2::available()) { GTEST_SKIP() << "no futex2 (Linux 6.7+)"; }
	uint32_t x = 0;
	std::atomic<int> result{1};
	std::thread t([&]() {
		const struct timespec to = in_ms(200);
		result = futex2::wait(&x, 0u, 0b01, &to) == 0 ? 0 : errno;
	});
	std::this_thread::sleep_for(std::chrono::milliseconds(20));
	EXPECT_EQ(futex2::wake(&x, INT32_MAX, 0b10), 0);
	t.join();
	EXPECT_EQ(result.load(), ETIMEDOUT);
}

/// waiting on two futexes, the second one is woken
TEST(futex2, waitv) {
	if (!futex2::available()) { GTEST_SKIP() << "no futex2 (Linux 6.7+)"; }
	uint32_t a = 0, b = 0;
	std::atomic<int> index{-2};
	std::thread t([&]() {
		struct futex_waitv w[2] = {futex2::waiter(&a, 0u), futex2::waiter(&b, 0u)};
		int r;
		while ((r = futex2::waitv(w, 2)) < 0 && errno == EINTR) {}
		index = r < 0 ? -errno : r;
	});
	// wake until the thread is woken (it may not sleep yet)
	while (index == -2) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
		futex2::wake(&b);
	}
	t.join();
	EXPECT_EQ(index.load(), 1);

	// a value differs: no sleep
	struct futex_waitv w[2] = {futex2::waiter(&a, 1u), futex2::waiter(&b, 0u)};
	EXPECT_EQ(futex2::waitv(w, 2), -1);
	EXPECT_EQ(errno, EAGAIN);
}

/// two waiters on `a`: one is woken, the other one moved to `b`
TEST(futex2, requeue) {
	if (!futex2::available()) { GTEST_SKIP() << "no futex2 (Linux 6.7+)"; }
	uint32_t a = 0, b = 0;
	std::atomic<uint32_t> done{0};
	std::vector<std::thread> threads;
	for (uint32_t i = 0; i < 2; ++i) {
		threads.emplace_back([&]() {
			futex2::wait(&a, 0u);
			done++;
		});
	}
	std::this_thread::sleep_for(std::chrono::milliseconds(20));
	const int r = futex2::requeue(&a, 0u, &b, 1, 1);
	EXPECT_GE(r, 0);
	EXPECT_LE(r, 2);
	if (r == 2) {
		// both were sleeping: one woken, one moved to `b`
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
		EXPECT_EQ(done.load(), 1u);
		EXPECT_EQ(futex2::wake(&b), 1);
	}

	// release the threads, which did not sleep yet
	__atomic_store_n(&a, 1u, __ATOMIC_RELEASE);
	futex2::wake(&a);
	futex2::wake(&b);
	for (auto &t : threads) { t.join(); }
	EXPECT_EQ(done.load(), 2u);
}
#endif
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
