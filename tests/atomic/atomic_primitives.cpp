#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <thread>

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
