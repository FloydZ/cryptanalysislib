#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <thread>
#include <vector>

#include "atomic/atomic_primitives.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

struct likely_padded {
    std::uint8_t c;
    std::uint16_t st;
    std::uint32_t i;
};

TEST(primitives, cas) {
	uint32_t val = 0;
	CAS(&val, &val, 1);
	EXPECT_EQ(val, 1);
}



TEST(__maybe_has_padding, simple) {
    constexpr bool t1 = __atomic_impl::__maybe_has_padding<uint32_t>();
    EXPECT_EQ(t1, false);
    constexpr bool t2 = __atomic_impl::__maybe_has_padding<uint8_t>();
    EXPECT_EQ(t2, false);
    constexpr bool t3 = __atomic_impl::__maybe_has_padding<double>();
    EXPECT_EQ(t3, false);
    constexpr bool t4 = __atomic_impl::__maybe_has_padding<likely_padded>();
    EXPECT_EQ(t4, true);
}


TEST(__clear_padding, simple) {
    likely_padded a;
    likely_padded *p = &a;
    
    auto *t = __atomic_impl::__clear_padding(p);
    EXPECT_EQ(*t, p);
}

TEST(primitives, cmpexch_failure_order) {
	static_assert(__atomic_impl::__is_valid_cmpexch_failure_order(std::memory_order_relaxed));
	static_assert(__atomic_impl::__is_valid_cmpexch_failure_order(std::memory_order_acquire));
	static_assert(__atomic_impl::__is_valid_cmpexch_failure_order(std::memory_order_seq_cst));
	static_assert(!__atomic_impl::__is_valid_cmpexch_failure_order(std::memory_order_release));
	static_assert(!__atomic_impl::__is_valid_cmpexch_failure_order(std::memory_order_acq_rel));

	uint64_t v = 5, e = 5;
	EXPECT_TRUE(__atomic_impl::compare_exchange_strong(&v, e, 7ull, std::memory_order_acq_rel, std::memory_order_acquire));
	EXPECT_EQ(v, 7u);
	e = 5;
	EXPECT_FALSE(__atomic_impl::compare_exchange_strong(&v, e, 9ull, std::memory_order_seq_cst, std::memory_order_relaxed));
	EXPECT_EQ(e, 7u);
}

/// the free functions in `cryptanalysislib::atomic` against `std::atomic`
TEST(atomic, fetch_ops) {
	using namespace cryptanalysislib;
	uint32_t x = 0b1100;
	std::atomic<uint32_t> e{0b1100};
	EXPECT_EQ(atomic::fetch_add(&x, 5u), e.fetch_add(5u));
	EXPECT_EQ(atomic::fetch_sub(&x, 2u, std::memory_order_relaxed), e.fetch_sub(2u));
	EXPECT_EQ(atomic::fetch_and(&x, 0b1010u), e.fetch_and(0b1010u));
	EXPECT_EQ(atomic::fetch_or(&x, 0b0101u), e.fetch_or(0b0101u));
	EXPECT_EQ(atomic::fetch_xor(&x, 0b1111u), e.fetch_xor(0b1111u));
	EXPECT_EQ(atomic::add_fetch(&x, 7u), e += 7u);
	EXPECT_EQ(atomic::sub_fetch(&x, 3u), e -= 3u);
	EXPECT_EQ(atomic::and_fetch(&x, 0xfu), e &= 0xfu);
	EXPECT_EQ(atomic::or_fetch(&x, 0x30u), e |= 0x30u);
	EXPECT_EQ(atomic::xor_fetch(&x, 0x11u), e ^= 0x11u);
	EXPECT_EQ(atomic::load(&x), e.load());

	atomic::store(&x, 3u);
	EXPECT_EQ(atomic::exchange(&x, 4u), 3u);
	uint32_t c = 4u;
	EXPECT_TRUE(atomic::compare_exchange_strong(&x, c, 5u));
	c = 4u;
	EXPECT_FALSE(atomic::compare_exchange_weak(&x, c, 6u));
	EXPECT_EQ(c, 5u);
}

TEST(atomic, fetch_ops_pointer_and_float) {
	using namespace cryptanalysislib;
	uint64_t a[8] = {0};
	uint64_t *p = a;
	// scaled by the size of the pointee, as for `std::atomic<T *>`
	EXPECT_EQ(atomic::fetch_add(&p, 3), a);
	EXPECT_EQ(p, a + 3);
	EXPECT_EQ(atomic::sub_fetch(&p, 2), a + 1);
	EXPECT_EQ(atomic::add_fetch(&p, 1), a + 2);
	EXPECT_EQ(atomic::fetch_sub(&p, 2), a + 2);
	EXPECT_EQ(p, a);

	double d = 1.5;
	EXPECT_EQ(atomic::fetch_add(&d, 2.0), 1.5);
	EXPECT_EQ(atomic::add_fetch(&d, 0.5), 4.0);
	EXPECT_EQ(atomic::fetch_sub(&d, 1.0), 4.0);
	EXPECT_EQ(atomic::sub_fetch(&d, 1.0), 2.0);
}

TEST(atomic, fetch_add_threads) {
	using namespace cryptanalysislib;
	constexpr uint32_t nthreads = 8, iters = 100000;
	uint64_t counter = 0;
	double fcounter = 0;
	std::vector<std::thread> threads;
	for (uint32_t t = 0; t < nthreads; ++t) {
		threads.emplace_back([&]() {
			for (uint32_t i = 0; i < iters; ++i) {
				atomic::fetch_add(&counter, 1u, std::memory_order_relaxed);
				atomic::fetch_add(&fcounter, 1.0, std::memory_order_relaxed);
			}
		});
	}
	for (auto &t : threads) { t.join(); }
	EXPECT_EQ(counter, uint64_t(nthreads) * iters);
	EXPECT_EQ(fcounter, double(nthreads) * iters);
}

#if __cpp_lib_atomic_wait
/// wait/notify on a volatile object (const volatile overloads) and on a
/// normal one
TEST(wait_notify, volatile_and_normal) {
	volatile uint32_t x = 0;
	std::thread t([&]() {
		__atomic_impl::store(&x, 1u, std::memory_order_release);
		__atomic_impl::notify_one(&x);
		__atomic_impl::notify_all(&x);
	});
	__atomic_impl::wait(&x, 0u);
	t.join();
	EXPECT_EQ(x, 1u);

	uint32_t y = 0;
	std::thread t2([&]() {
		__atomic_impl::store(&y, 2u, std::memory_order_release);
		__atomic_impl::notify_all(&y);
	});
	__atomic_impl::wait(&y, 0u);
	t2.join();
	EXPECT_EQ(y, 2u);
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
