#include <gtest/gtest.h>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "compression/compress_float.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

static std::vector<double> inputs() {
	std::vector<double> v = {0.0, -0.0, 1.0, -1.0, 0.1, 1e-30, 1e-40 /* float denormal */, 3.0e38, 4.0e38 /* inf */,
	                         std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
	                         1.0 + 1.0 / 256, 1.0 + 1.0 / 512 /* tie: round to even */, 1.0 + 3.0 / 512};
	for (int i = 0; i < 1000; ++i) { v.push_back((rand() - RAND_MAX / 2) * std::ldexp(1.0, rand() % 200 - 100)); }
	return v;
}

/// round trip: relative error <= 2^-8, specials exact
TEST(compress_float, round_trip) {
	const auto v = inputs();
	for (size_t len = 0; len <= v.size(); len += (len < 40 ? 1 : 97)) {
		std::vector<uint16_t> h(len);
		std::vector<double> r(len);
		to_float16(h.data(), v.data(), len);
		from_float16(r.data(), h.data(), len);
		for (size_t i = 0; i < len; ++i) {
			const double x = v[i], y = r[i];
			const float fx = float(x);
			if (std::isinf(fx)) { EXPECT_EQ(y, double(fx)) << x; continue; }
			if (std::fpclassify(fx) == FP_SUBNORMAL) { EXPECT_EQ(y, 0.0) << x; continue; }
			EXPECT_LE(std::fabs(y - x), std::fabs(x) * (1.0 / 256)) << x << " " << y;
			EXPECT_EQ(std::signbit(x), std::signbit(y)) << x;
		}
	}
	// ties to even
	uint16_t h[2];
	const double t[2] = {1.0 + 1.0 / 256, 1.0 + 3.0 / 256}; // exactly between two bfloat16s
	to_float16(h, t, 2);
	EXPECT_EQ(h[0], 0x3f80u);
	EXPECT_EQ(h[1], 0x3f82u);
	// NaN stays NaN
	const double nan = std::numeric_limits<double>::quiet_NaN();
	double back;
	to_float16(h, &nan, 1);
	from_float16(&back, h, 1);
	EXPECT_TRUE(std::isnan(back));
}

/// the scalar versions give the same bits as the AVX512-BF16 ones (if the
/// test is compiled with BF16)
TEST(compress_float, scalar_equals_simd) {
	const auto v = inputs();
	std::vector<uint16_t> h(v.size());
	to_float16(h.data(), v.data(), v.size());
	for (size_t i = 0; i < v.size(); ++i) {
		EXPECT_EQ(h[i], internal::double_to_bfloat16(v[i])) << v[i];
	}
	std::vector<double> r(v.size());
	from_float16(r.data(), h.data(), v.size());
	for (size_t i = 0; i < v.size(); ++i) {
		EXPECT_EQ(r[i], internal::bfloat16_to_double(h[i]));
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
