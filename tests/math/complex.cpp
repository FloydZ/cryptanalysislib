#include <complex>
#include <gtest/gtest.h>

#include "math/math.h"
#include "random.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

using namespace cryptanalysislib::math;

// constexpr usable, Gaussian integers are exact
constexpr complex<int64_t> ga{3, 4}, gb{1, -2};
static_assert(ga + gb == complex<int64_t>{4, 2});
static_assert(ga - gb == complex<int64_t>{2, 6});
static_assert(ga * gb == complex<int64_t>{11, -2});
static_assert(ga * ga.conj() == complex<int64_t>{25, 0});
static_assert(ga.norm() == 25);
static_assert(-ga == complex<int64_t>{-3, -4});
static_assert(2 * ga == complex<int64_t>{6, 8});
static_assert((ga * gb) / gb == ga);

TEST(Complex, compare_std) {
	for (uint32_t i = 0; i < 1000; ++i) {
		const double a = (double)(rng() % 2001) / 100. - 10.;
		const double b = (double)(rng() % 2001) / 100. - 10.;
		const double c = (double)(rng() % 2001) / 100. - 10.;
		const double d = (double)(rng() % 2001) / 100. - 10.;
		if ((c == 0.) && (d == 0.)) { continue; }

		const complex<double> x{a, b}, y{c, d};
		const std::complex<double> sx{a, b}, sy{c, d};

		const auto check = [](const complex<double> &r, const std::complex<double> &s) {
			EXPECT_NEAR(r.real(), s.real(), 1e-9 * (1. + std::abs(s.real())));
			EXPECT_NEAR(r.imag(), s.imag(), 1e-9 * (1. + std::abs(s.imag())));
		};

		check(x + y, sx + sy);
		check(x - y, sx - sy);
		check(x * y, sx * sy);
		check(x / y, sx / sy);
		check(x.conj(), std::conj(sx));
		EXPECT_NEAR(x.norm(), std::norm(sx), 1e-9 * (1. + std::norm(sx)));
		EXPECT_NEAR(x.abs(), std::abs(sx), 1e-9 * (1. + std::abs(sx)));
	}
}

TEST(Complex, print) {
	std::stringstream ss;
	ss << complex<int>{1, -2};
	EXPECT_EQ(ss.str(), "(1,-2)");
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
