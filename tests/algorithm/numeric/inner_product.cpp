#include <gtest/gtest.h>
#include <cstdint>
#include <functional>
#include <list>
#include <numeric>
#include <string>
#include <vector>

#include "random.h"
#include "algorithm/numeric/inner_product.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// `inner_product` against `std::inner_product`
TEST(inner_product, simple) {
	for (size_t n = 0; n <= 200; ++n) {
		std::vector<uint32_t> a(n), b(n);
		for (size_t i = 0; i < n; ++i) { a[i] = uint32_t(rng()); b[i] = uint32_t(rng()); }
		EXPECT_EQ(cryptanalysislib::inner_product(a.begin(), a.end(), b.begin(), uint64_t(5)),
		          std::inner_product(a.begin(), a.end(), b.begin(), uint64_t(5))) << n;
		// the result type is the type of `init`: wraps around mod 2^32
		EXPECT_EQ(cryptanalysislib::inner_product(a.begin(), a.end(), b.begin(), uint32_t(0)),
		          std::inner_product(a.begin(), a.end(), b.begin(), uint32_t(0))) << n;
	}

	// generalized: number of equal positions, input iterators
	const std::list<int> a = {1, 2, 3, 4};
	const std::vector<int> b = {1, 0, 3, 0};
	EXPECT_EQ(cryptanalysislib::inner_product(a.begin(), a.end(), b.begin(), 0, std::plus<>(), std::equal_to<>()), 2);

	// order of the operations: string concatenation is not commutative
	const std::vector<std::string> s = {"a", "b", "c"}, t = {"x", "y", "z"};
	EXPECT_EQ(cryptanalysislib::inner_product(s.begin(), s.end(), t.begin(), std::string(">"), std::plus<>(), std::plus<>()),
	          ">axbycz");
}

/// usable in constant expressions
TEST(inner_product, constexpr_inner_product) {
	constexpr auto r = []() {
		const int a[3] = {1, 2, 3}, b[3] = {4, 5, 6};
		return cryptanalysislib::inner_product(a, a + 3, b, 1);
	}();
	static_assert(r == 33);
	EXPECT_EQ(r, 33);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
