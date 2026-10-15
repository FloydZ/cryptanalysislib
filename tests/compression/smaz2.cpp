#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

#include "compression/compression.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using Label_Type = uint64_t;

TEST(smaz2, simple) {
	const unsigned char text[] = "this is an input text, test, test, test";
	unsigned char text2[sizeof(text)] = {0};
	unsigned char buf[256];

	const size_t olen  = smaz2_compress(buf, sizeof(buf), text, sizeof(text));
	printf("Compressed length (%lu): %.02f%%\n", olen, (float)olen/sizeof(text)*100);
	const size_t olen2 = smaz2_decompress(text2, sizeof(text2), buf, olen);
	EXPECT_EQ(olen2, sizeof(text));
	EXPECT_EQ(memcmp(text, text2, sizeof(text)), 0);
}

TEST(smaz2, small_output) {
	// words, bigrams, verbatim bytes and plain bytes
	const unsigned char text[] = "this is the information \x01\x02 test";
	unsigned char buf[256];
	const size_t olen = smaz2_compress(buf, sizeof(buf), text, sizeof(text));

	// every prefix of the output, without writing past `cap`
	for (size_t cap = 0; cap <= sizeof(text); cap++) {
		unsigned char *out = (unsigned char *)malloc(cap + 1);
		out[cap] = 0xAA;
		const size_t n = smaz2_decompress(out, cap, buf, olen);
		EXPECT_EQ(n, cap);
		EXPECT_EQ(memcmp(out, text, cap), 0);
		EXPECT_EQ(out[cap], 0xAA);
		free(out);
	}
}

TEST(smaz2, iterator) {
	const std::string text = "this is the information you have been looking for \x01\x02\x03 test";
	const std::vector<unsigned char> in(text.begin(), text.end());

	// worst case 6/5 of the input
	std::vector<unsigned char> c(in.size() * 6 / 5 + 1);
	const auto cend = smaz2_compress(in.begin(), in.end(), c.begin(), c.end());
	ASSERT_LT(size_t(cend - c.begin()), in.size());

	std::vector<unsigned char> d(in.size() + 16, 0xAA);
	const auto dend = smaz2_decompress(c.cbegin(), std::vector<unsigned char>::const_iterator(cend),
	                                   d.begin(), d.end());
	ASSERT_EQ(size_t(dend - d.begin()), in.size());
	EXPECT_EQ(memcmp(d.data(), in.data(), in.size()), 0);
	EXPECT_EQ(d[in.size()], 0xAA);

	// bounded output
	std::vector<unsigned char> e(10, 0);
	const auto eend = smaz2_decompress(c.cbegin(), std::vector<unsigned char>::const_iterator(cend),
	                                   e.begin(), e.begin() + 5);
	EXPECT_EQ(eend - e.begin(), 5);
	EXPECT_EQ(memcmp(e.data(), in.data(), 5), 0);
	EXPECT_EQ(e[5], 0);

	// incompressible: verbatim bytes, needs more than the input size
	const std::vector<unsigned char> v(20, 0x01);
	std::vector<unsigned char> vc(v.size() * 6 / 5 + 1);
	const auto vcend = smaz2_compress(v.begin(), v.end(), vc.begin(), vc.end());
	EXPECT_GT(size_t(vcend - vc.begin()), v.size());
	std::vector<unsigned char> vd(v.size());
	const auto vdend = smaz2_decompress(vc.begin(), vcend, vd.begin(), vd.end());
	EXPECT_EQ(size_t(vdend - vd.begin()), v.size());
	EXPECT_EQ(vd, v);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
