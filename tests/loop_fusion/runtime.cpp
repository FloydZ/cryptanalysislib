#include <gtest/gtest.h>
#include <array>
#include <numeric>

#include "loop_fusion/runtime/loop_fusion.hpp"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace loop_fusion::runtime;

TEST(LoopFusion, NoDuplFkt) {
	std::array<int, 41> a {};
	std::vector<int> b;

	auto fill = [&a](std::size_t i) { a[i] = static_cast<int>(i); };
	auto loop1 = loop_to(a.size(), fill);
	auto fibonacci = [&a](std::size_t i) { a[i] = a[i - 2] + a[i - 1]; };
	auto loop2 = loop_from_to(2, a.size(), fibonacci);
	auto addOne = [&a, &b](std::size_t i) { b.push_back(a[i] + 1); };
	auto loop3 = loop(range { a.size() - 16, a.size() }) | addOne;

	(loop1 | loop2 | loop3).run();

	EXPECT_EQ(a[0], 0);
	EXPECT_EQ(a[1], 1);
	EXPECT_EQ(a[2], 1);
	EXPECT_EQ(a[20], 6'765);
	EXPECT_EQ(a[40], 102'334'155);
	ASSERT_EQ(b.size(), 16);
	EXPECT_EQ(b[0], 75'025 + 1);
	EXPECT_EQ(b[15], 102'334'155 + 1);
}

TEST(LoopFusion, Basic) {
	std::vector<int> vec { 0, 0, 0, 0, 0 };
	const int vecsize { 5 };
	const auto set_one = [&vec](size_t i) {
		if (vec[i] == 0) {
			vec[i] = 1;
		} else {
			vec[i] = 100;
		}
	};
	const auto set_two = [&vec](size_t i) {
		if (vec[i] == 1) {
			vec[i] = 2;
		} else {
			vec[i] = 100;
		}
	};
	const auto set_three = [&vec](size_t i) {
		if (vec[i] == 2) {
			vec[i] = 3;
		} else {
			vec[i] = 100;
		}
	};
	const auto set_four = [&vec](size_t i) {
		if (vec[i] == 3) {
			vec[i] = 4;
		} else {
			vec[i] = 100;
		}
	};

	auto merged = loop_to(vecsize, set_one) | loop_to(vecsize, set_two) //
	              | loop_to(vecsize, set_three) | loop_to(2, set_four);
	merged.run();

	const int sum = std::accumulate(vec.cbegin(), vec.cend(), 0);
	ASSERT_EQ(sum, vec.size() * 3 + 2);
}

TEST(LoopFusion, DuplicateFkt) {
	std::size_t sum = 0;

	auto addOne = [&sum](std::size_t /*unused*/) { sum += 1; };

	auto loop1 = loop(range { 100, 1000 }, addOne); // + 900
	auto loop2 = loop_to(2000, addOne, addOne); // + 4000
	auto loop3 = loop_from_to(500, 1500) | addOne; // + 1000

	auto looper_union = (loop1 | loop2 | loop3);
	looper_union.run();

	EXPECT_EQ(sum, 5900);
}

TEST(LoopFusion, Merge) {
	std::size_t sum = 0;
	auto addOne = [&sum](std::size_t /*unused*/) { sum += 1; };

	const auto r1 = loop({ 100, 1000 }, addOne);
	const auto r2 = loop({ 100, 1000 }, addOne);
	(r1 | r2).run();
	ASSERT_EQ(sum, 2*900);
}

/**
 * A simple unit test for multiple unions merged together.
 * Unfortunately, the implementation is missing a final part, see
 * runtime/looper_union.hpp for more information.
 */
TEST(LoopFusion, MultipleMerge) {
    std::size_t sum = 0;
    auto op1 = [&sum](std::size_t /*unused*/) { sum += 1; };
    auto op2 = [&sum](std::size_t /*unused*/) { sum += 2; };
    auto op3 = [&sum](std::size_t /*unused*/) { sum += 3; };

    auto l1 = loop({ 100, 1000 }, op1);
    auto l2 = loop_to(2000, op2);
    auto l3 = loop_from_to(500, 1500, op3);
    auto u1 = l1 | l2;
    auto u2 = l2 | l3;
	// TODO
    // (u1 | u2).run();
    // ASSERT_EQ(sum, (900 + 4 * 2000 + 3 * 1000));
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
