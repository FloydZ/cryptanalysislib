#include <cstdint>
#include <cstring>
#include <vector>
#include <gtest/gtest.h>

#include "memory/memory.h"
#include "alloc/alloc.h"
#include "random.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace cryptanalysislib;

/// `memmove` (elements) and its byte kernels against ::memmove, in both
/// directions with overlap, for all small sizes and offsets
TEST(memory, memmove) {
	std::vector<uint8_t> buf(512), ref(512);
	for (size_t n = 0; n <= 200; n += (n < 70 ? 1 : 13)) {
		for (const long d : {-33L, -8L, -1L, 0L, 1L, 5L, 64L}) {
			for (auto &x : buf) { x = rng<uint8_t>(); }
			ref = buf;
			const size_t src = 100, dst = size_t(long(src) + d);
			::memmove(ref.data() + dst, ref.data() + src, n);
			cryptanalysislib::memmove(buf.data() + dst, buf.data() + src, n);
			ASSERT_EQ(buf, ref) << n << " " << d;
			if (n <= 64) {
				for (auto &x : buf) { x = rng<uint8_t>(); }
				ref = buf;
				::memmove(ref.data() + dst, ref.data() + src, n);
				internal::memmove_small_u8(buf.data() + dst, buf.data() + src, n);
				ASSERT_EQ(buf, ref) << "small " << n << " " << d;
			}
			for (auto &x : buf) { x = rng<uint8_t>(); }
			ref = buf;
			::memmove(ref.data() + dst, ref.data() + src, n);
			internal::memmove_u8(buf.data() + dst, buf.data() + src, n);
			ASSERT_EQ(buf, ref) << "u8 " << n << " " << d;
		}
	}
	// element counts for wider types
	std::vector<uint64_t> w(100), we;
	for (size_t i = 0; i < w.size(); ++i) { w[i] = i; }
	we = w;
	cryptanalysislib::memmove(w.data() + 3, w.data(), 50);
	::memmove(we.data() + 3, we.data(), 50 * sizeof(uint64_t));
	EXPECT_EQ(w, we);
}

/// the byte kernels of memcpy/memset
TEST(memory, byte_kernels) {
	std::vector<uint8_t> a(300), b(300, 0), e(300, 0);
	for (auto &x : a) { x = rng<uint8_t>(); }
	for (const size_t n : {size_t(0), size_t(1), size_t(31), size_t(32), size_t(33), size_t(255)}) {
		std::fill(b.begin(), b.end(), 0);
		std::fill(e.begin(), e.end(), 0);
		internal::memcpyU256(b.data() + 1, a.data() + 3, n);
		::memcpy(e.data() + 1, a.data() + 3, n);
		ASSERT_EQ(b, e) << n;
		std::fill(b.begin(), b.end(), 0);
		internal::memcpy(b.data() + 1, a.data() + 3, n);
		ASSERT_EQ(b, e) << n;

		std::fill(b.begin(), b.end(), 0);
		std::fill(e.begin(), e.end(), 0);
		internal::memset_bytes(b.data() + 1, uint8_t(0xab), n);
		::memset(e.data() + 1, 0xab, n);
		ASSERT_EQ(b, e) << n;
	}
}

TEST(alloc, aligned_alloc_reallocate) {
	for (const size_t al : {size_t(8), size_t(64), size_t(4096)}) {
		for (const size_t s : {size_t(1), size_t(100), size_t(10000)}) {
			auto *p = (uint8_t *)cryptanalysislib::aligned_alloc(al, s);
			ASSERT_NE(p, nullptr);
			EXPECT_EQ(((uintptr_t)p) % al, 0u);
			::memset(p, 0x5a, s);
			cryptanalysislib::aligned_free(p);
		}
	}

	// element counts with an element-count allocator (std::allocator)
	std::allocator<uint64_t> a;
	uint64_t *p = a.allocate(10);
	for (uint64_t i = 0; i < 10; ++i) { p[i] = i * i; }
	p = cryptanalysislib::reallocate(a, p, 10, 1000);
	ASSERT_NE(p, nullptr);
	for (uint64_t i = 0; i < 10; ++i) { EXPECT_EQ(p[i], i * i); }
	p[999] = 7;
	p = cryptanalysislib::reallocate(a, p, 1000, 5);
	for (uint64_t i = 0; i < 5; ++i) { EXPECT_EQ(p[i], i * i); }
	a.deallocate(p, 5);

	// `AlignmentMallocator` counts bytes: its own `reallocate` keeps the first bytes
	using AM = AlignmentMallocator<uint8_t, 64>;
	uint8_t *q = AM::allocate(10);
	for (uint8_t i = 0; i < 10; ++i) { q[i] = i; }
	AM am;
	q = cryptanalysislib::reallocate(am, q, 10, 100);
	EXPECT_EQ(((uintptr_t)q) % 64, 0u);
	for (uint8_t i = 0; i < 10; ++i) { EXPECT_EQ(q[i], i); }
	AM::deallocate(q, 100);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
