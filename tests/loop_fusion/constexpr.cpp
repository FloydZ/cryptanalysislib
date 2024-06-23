#include <gtest/gtest.h>
#include <iostream>
#include <array>
#include <numeric>
#include <vector>

#include "loop_fusion/compiletime/loop_fusion.hpp"

using ::testing::InitGoogleTest;
using ::testing::Test;

// TODO probably need inc type: so that -1/i-- is possible
using namespace loop_fusion::compiletime;


TEST(LoopFusion, Types) {
	using seq_4_5 = std::integer_sequence<std::size_t, 4, 5>;
	using seq_6_7 = std::integer_sequence<std::size_t, 6, 7>;
	using seq_4_5_6_7 = std::integer_sequence<std::size_t, 4, 5, 6, 7>;

	static_assert(std::is_same_v<seq_4_5_6_7, types::index_sequence_from_to<4, 8>>);
	static_assert(std::is_same_v<seq_4_5_6_7, types::index_sequence_cat<seq_4_5, seq_6_7>>);

	using nested = std::tuple<int, std::tuple<int, char>, char>;
	using flattened = std::tuple<int, int, char, char>;
	static_assert(std::is_same_v<std::tuple<char>, types::as_tuple_t<char>>);
	static_assert(std::is_same_v<std::tuple<char>, types::as_tuple_t<std::tuple<char>>>);
	static_assert(std::is_same_v<flattened, types::flatten_tuple_t<nested>>);
}

TEST(LoopFusion, Basic) {
	std::vector<int> vec;
	auto fill = [&vec](int i) { vec.push_back(i); };
	auto l = basic_looper<int, -100, 101, decltype(fill)>(std::make_tuple(fill));
	l.run();
	ASSERT_EQ(vec.size(), 201);
	EXPECT_EQ(vec.at(0), -100);
	EXPECT_EQ(vec.at(100), 0);
	EXPECT_EQ(vec.at(200), 100);
}

TEST(LoopFusion, Basic2) {
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

	auto merged = loop<0, vecsize>(set_one) | loop<0, vecsize>(set_two) //
	              | loop<0, vecsize>(set_three) | loop<0, 2>(set_four);
	merged.run();

	const int sum = std::accumulate(vec.cbegin(), vec.cend(), 0);
	EXPECT_EQ(sum, vec.size() * 3 + 2);
}

TEST(LoopFusion, Basic3) {
	std::vector<size_t> vec;
	auto fill = [&vec](size_t i) { vec.push_back(i); };
	auto l = loop_to<10>(fill) | fill; // 2x
	l.run();
	EXPECT_EQ(vec.size(), 20);
}

TEST(LoopFusion, Union) {
	// Example from MergingLoops.md
	std::array<std::vector<size_t>, 5> vec;
	const auto fill_0 = [&vec](size_t /*unused*/) { vec[0].push_back(0); };
	const auto fill_1 = [&vec](size_t /*unused*/) { vec[1].push_back(1); };
	const auto fill_2 = [&vec](size_t /*unused*/) { vec[2].push_back(2); };
	const auto fill_3 = [&vec](size_t /*unused*/) { vec[3].push_back(3); };
	const auto fill_4 = [&vec](size_t /*unused*/) { vec[4].push_back(4); };

	// Note that the second integer is exclusive.
	using loop_fill_0 = basic_looper_union_range<std::size_t, 0, 2, std::index_sequence<1, 2, 4>>;
	using loop_fill_1 = basic_looper_union_range<std::size_t, 2, 100, std::index_sequence<1, 2, 3, 4>>;
	using loop_fill_2 = basic_looper_union_range<std::size_t, 100, 201, std::index_sequence<0, 1, 2, 3, 4>>;
	using loop_fill_3 = basic_looper_union_range<std::size_t, 201, 202, std::index_sequence<1, 2, 4>>;
	using loop_fill_4 = basic_looper_union_range<std::size_t, 202, 206, std::index_sequence<1, 4>>;

	using loop_tuple = std::tuple<loop_fill_0, loop_fill_1, loop_fill_2, loop_fill_3, loop_fill_4>;
	using function_tuple
	        = std::tuple<decltype(fill_0), decltype(fill_1), decltype(fill_2), decltype(fill_3), decltype(fill_4)>;
	// pair for required types
	using pair = std::pair<loop_tuple, function_tuple>;
	const auto loops = std::make_tuple(fill_0, fill_1, fill_2, fill_3, fill_4);
	basic_looper_union<std::size_t, pair> union_loop { loops };

	union_loop.run();

	EXPECT_EQ(vec[0].size(), 101);
	EXPECT_EQ(vec[1].size(), 206);
	EXPECT_EQ(vec[2].size(), 202);
	EXPECT_EQ(vec[3].size(), 199);
	EXPECT_EQ(vec[4].size(), 206);
}

TEST(LoopFusion, Merge) {
	// Example from MergingLoops.md
	std::array<std::vector<size_t>, 4> vec;
	const auto fill_0 = [&vec](size_t /*unused*/) { vec[0].push_back(0); };
	const auto fill_1 = [&vec](size_t /*unused*/) { vec[1].push_back(1); };
	const auto fill_2 = [&vec](size_t /*unused*/) { vec[2].push_back(2); };
	const auto fill_3 = [&vec](size_t /*unused*/) { vec[3].push_back(3); };

	auto merged = loop<10, 20>(fill_0) | loop<2, 20>(fill_1);
	auto merged_2 = merged | loop<30, 60>(fill_2);
	auto merged_3 = merged_2 | loop<10, 40>(fill_3);
	merged_3.run();

	// For Debugging purposes:
	// typedef decltype(merged_3)::something_made_up X;
	EXPECT_EQ(vec[0].size(), 10);
	EXPECT_EQ(vec[1].size(), 18);
	EXPECT_EQ(vec[2].size(), 30);
	EXPECT_EQ(vec[3].size(), 30);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
