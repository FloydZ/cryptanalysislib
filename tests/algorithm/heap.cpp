#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <functional>
#include <vector>

#include "random.h"
#include "algorithm/heap.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

template <typename T>
class Heap : public testing::Test {};
TYPED_TEST_SUITE_P(Heap);

/// random values with many duplicates
template<typename T>
static std::vector<T> random_vector(const size_t n) {
	std::vector<T> v(n);
	for (auto &x : v) { x = T(rng() % 64u); }
	return v;
}

TYPED_TEST_P(Heap, is_heap) {
	for (size_t n = 0; n < 400; n += (n < 40) ? 1 : 13) {
		for (uint32_t r = 0; r < 20; ++r) {
			auto v = random_vector<TypeParam>(n);
			std::make_heap(v.begin(), v.end());
			if ((r & 1u) && (n > 1)) {
				// a child larger than its parent
				const size_t c = 1 + (rng() % (n - 1));
				v[c] = v[(c - 1) / 2] + 1;
			}

			EXPECT_EQ(cryptanalysislib::is_heap(v.begin(), v.end()), std::is_heap(v.begin(), v.end()));
			EXPECT_EQ(cryptanalysislib::is_heap_until(v.begin(), v.end()), std::is_heap_until(v.begin(), v.end()));
			EXPECT_EQ(cryptanalysislib::is_heap(v.begin(), v.end(), std::greater<>{}), std::is_heap(v.begin(), v.end(), std::greater<>{}));
		}
	}
}

TYPED_TEST_P(Heap, make_push_pop_sort) {
	for (const size_t n : {0ul, 1ul, 2ul, 3ul, 10ul, 255ul, 256ul, 257ul, 1000ul, 40000ul}) {
		auto v = random_vector<TypeParam>(n);
		auto e = v;

		cryptanalysislib::make_heap(v.begin(), v.end());
		EXPECT_TRUE(std::is_heap(v.begin(), v.end()));
		EXPECT_TRUE(std::is_permutation(v.begin(), v.end(), e.begin()));

		// push a new maximum, a new minimum and a random value
		for (const TypeParam x : {TypeParam(100), TypeParam(0), TypeParam(rng() % 64u)}) {
			v.push_back(x);
			e.push_back(x);
			cryptanalysislib::push_heap(v.begin(), v.end());
			EXPECT_TRUE(std::is_heap(v.begin(), v.end()));
		}
		EXPECT_TRUE(std::is_permutation(v.begin(), v.end(), e.begin()));

		cryptanalysislib::pop_heap(v.begin(), v.end());
		EXPECT_EQ(v.back(), *std::max_element(e.begin(), e.end()));
		EXPECT_TRUE(std::is_heap(v.begin(), v.end() - 1));
		cryptanalysislib::push_heap(v.begin(), v.end());

		cryptanalysislib::sort_heap(v.begin(), v.end());
		std::sort(e.begin(), e.end());
		EXPECT_EQ(v, e);
	}
}

TYPED_TEST_P(Heap, min_heap) {
	auto v = random_vector<TypeParam>(1000);
	auto e = v;
	cryptanalysislib::make_heap(v.begin(), v.end(), std::greater<>{});
	EXPECT_TRUE(std::is_heap(v.begin(), v.end(), std::greater<>{}));
	cryptanalysislib::sort_heap(v.begin(), v.end(), std::greater<>{});
	std::sort(e.begin(), e.end(), std::greater<>{});
	EXPECT_EQ(v, e);
}

REGISTER_TYPED_TEST_SUITE_P(Heap, is_heap, make_push_pop_sort, min_heap);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t, int32_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Heap, MyTypes);

TEST(heap, non_contiguous_and_constexpr) {
	std::deque<int32_t> d = {5, 1, 9, 3, 7, 2};
	cryptanalysislib::make_heap(d.begin(), d.end());
	EXPECT_TRUE(std::is_heap(d.begin(), d.end()));
	cryptanalysislib::sort_heap(d.begin(), d.end());
	EXPECT_EQ(d, (std::deque<int32_t>{1, 2, 3, 5, 7, 9}));

	constexpr auto r = []() {
		std::array<int32_t, 6> a = {5, 1, 9, 3, 7, 2};
		cryptanalysislib::make_heap(a.begin(), a.end());
		const bool h = cryptanalysislib::is_heap(a.begin(), a.end());
		cryptanalysislib::sort_heap(a.begin(), a.end());
		return std::make_pair(h, a);
	}();
	static_assert(r.first);
	static_assert(r.second[0] == 1 && r.second[5] == 9);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
