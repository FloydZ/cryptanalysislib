#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <vector>

#include "hash/hash.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib::hash;

/// official FNV test vectors (http://www.isthe.com/chongo/src/fnv/test_fnv.c)
TEST(fnv, test_vectors) {
	const auto *e = (const uint8_t *)"";
	const auto *a = (const uint8_t *)"a";
	const auto *f = (const uint8_t *)"foobar";
	EXPECT_EQ(fnv1a<uint32_t>(e, 0), 0x811c9dc5u);
	EXPECT_EQ(fnv1a<uint32_t>(a, 1), 0xe40c292cu);
	EXPECT_EQ(fnv1a<uint32_t>(f, 6), 0xbf9cf968u);
	EXPECT_EQ(fnv1a<uint64_t>(e, 0), 0xcbf29ce484222325ull);
	EXPECT_EQ(fnv1a<uint64_t>(a, 1), 0xaf63dc4c8601ec8cull);
	EXPECT_EQ(fnv1a<uint64_t>(f, 6), 0x85944171f73967e8ull);
	EXPECT_EQ(fnv1<uint32_t>(a, 1), 0x050c5d7eu);
	EXPECT_EQ(fnv1<uint32_t>(f, 6), 0x31f0b262u);
	EXPECT_EQ(fnv1<uint64_t>(a, 1), 0xaf63bd4c8601b7beull);
	EXPECT_EQ(fnv1<uint64_t>(f, 6), 0x340d8765a4dda9c2ull);
	static_assert(fnv1a<uint32_t>(nullptr, 0) == 0x811c9dc5u);
}

/// SIMD version against the scalar one, also with a tail (nr % lanes != 0)
template<typename T>
static void check_fnv1a_multi() {
	for (const size_t size : {0ul, 1ul, 4ul, 6ul, 8ul, 16ul, 33ul}) {
		for (const size_t nr : {0ul, 1ul, 7ul, 8ul, 21ul}) {
			std::vector<uint8_t> d(size * nr);
			for (size_t i = 0; i < d.size(); ++i) { d[i] = uint8_t(i * 131u + 7u); }
			std::vector<T> out(nr);
			fnv1a_multi<T>(out.data(), d.data(), size, nr);
			for (size_t i = 0; i < nr; ++i) {
				EXPECT_EQ(out[i], fnv1a<T>(d.data() + i * size, size)) << size << " " << nr;
			}
		}
	}
}

TEST(fnv, multi) {
	check_fnv1a_multi<uint32_t>();
	check_fnv1a_multi<uint64_t>();
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
