#include <gtest/gtest.h>
#include <cstdint>

#include "container/array.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

using A = const_array<int, 5>;
constexpr A a{5, 3, 4, 1, 2};

template<typename T, size_t N, size_t M>
constexpr bool eq(const const_array<T, N> &x, const int (&y)[M]) {
	if (N != M) { return false; }
	for (size_t i = 0; i < N; ++i) { if (x[i] != y[i]) { return false; } }
	return true;
}

// everything at compile time
static_assert(a.size() == 5);
static_assert(a[0] == 5 && a.at(4) == 2);
static_assert(eq(a.map([](int x) { return 2 * x; }), {10, 6, 8, 2, 4}));
static_assert(eq(a.map([](int x, int y) { return x + y; }, const_array<int, 3>{1, 1, 1}), {6, 4, 5}));
static_assert(eq(a.push_back(9), {5, 3, 4, 1, 2, 9}));
static_assert(eq(a.push_front(9), {9, 5, 3, 4, 1, 2}));
static_assert(eq(a.concat(const_array<int, 2>{7, 8}), {5, 3, 4, 1, 2, 7, 8}));
static_assert(eq(a.slice<1, 3>(), {3, 4}));
static_assert(eq(a.tail(), {3, 4, 1, 2}));
static_assert(eq(a.tail<2>(), {4, 1, 2}));
static_assert(eq(a.init(), {5, 3, 4, 1}));
static_assert(eq(a.insert<0>(0), {0, 5, 3, 4, 1, 2}));
static_assert(eq(a.insert<5>(0), {5, 3, 4, 1, 2, 0}));
static_assert(eq(a.mergesort([](int x, int y) { return x < y; }), {1, 2, 3, 4, 5}));
static_assert(eq(a.mergesort([](int x, int y) { return x > y; }), {5, 4, 3, 2, 1}));
static_assert(eq(a.partition([](int x) { return x % 2 == 0; }), {4, 2, 5, 3, 1})); // stable
static_assert(eq(transform(a, [](int x) { return x - 1; }), {4, 2, 3, 0, 1}));
// lexicographical comparison
static_assert(const_array<int, 2>{1, 5}.less(const_array<int, 2>{2, 1}));
static_assert(!const_array<int, 2>{2, 1}.less(const_array<int, 2>{1, 5}));
static_assert(const_array<int, 2>{1, 2}.less(const_array<int, 3>{1, 2, 0}));
static_assert(!const_array<int, 2>{1, 2}.less(const_array<int, 2>{1, 2}));

TEST(const_array, runtime) {
	A b{1, 2, 3, 4, 5};
	int sum = 0;
	for (const int x : b) { sum += x; }
	EXPECT_EQ(sum, 15);
	EXPECT_EQ(b.end() - b.begin(), 5);
	EXPECT_EQ(uintptr_t(b.begin()) % std_config_array.base_alignment, 0u);
	EXPECT_TRUE(eq(a.mergesort([](int x, int y) { return x < y; }), {1, 2, 3, 4, 5}));
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
