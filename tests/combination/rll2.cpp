#include <cstddef>
#include <set>
#include <gtest/gtest.h>

#include "combination/rll2.h"
#include "math/math.h"
#include "print/print.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

TEST(bit_rll2, p1) {
	using T = uint64_t;
	bit_rll2<T> b{};
    T w;
    uint32_t ctr = 0;
	do {
		EXPECT_TRUE(b.next());
		w = b.current();
		print_binary(w, 14);
        ctr += 1;
	} while (ctr < 100);
}

/// from `first()` to `last()`: F(bits + 2) words without three equal
/// consecutive bits, whose Fibonacci words (`fib()`) are distinct
TEST(bit_rll2, all_words) {
	bit_rll2<uint8_t> b{};
	std::set<uint8_t> words, fibs;
	do {
		const uint8_t w = b.current();
		for (uint32_t i = 0; i + 2 < 8; ++i) {
			const uint32_t t = (w >> i) & 7u;
			EXPECT_TRUE(t != 0 && t != 7) << uint32_t(w);
		}
		const uint8_t f = b.fib();
		EXPECT_EQ(f & (f >> 1u), 0u);
		words.insert(w);
		fibs.insert(f);
	} while (b.next());
	EXPECT_EQ(words.size(), 55u);
	EXPECT_EQ(fibs.size(), 55u);
}

int main(int argc, char **argv) {
	rng_seed(time(nullptr));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
