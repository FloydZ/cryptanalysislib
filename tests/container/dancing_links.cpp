#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "random.h"
#include "container/dancing_links.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

constexpr size_t N = 1u<<5u;
using T = uint64_t;
using D = dancing_links<T>;


TEST(DancingLinks, Dev) {
	D d;
	std::vector<D::node *> ret;
	for (size_t i = 0; i < N; i++) {
		ret.emplace_back(d.push_back(i));
	}

	for (size_t i = 0; i < N; i++) {
	}
}

TEST(DancingLinks, find_back) {
	D d;
	std::vector<D::node *> ret;
	for (size_t i = 0; i < N; i++) {
		ret.emplace_back(d.push_back(i));
	}

	/// make the longest search possible
	for (size_t i = 0; i < N; i++) {
		const auto t = d.find_back(i);
		EXPECT_NE(t, nullptr);
		EXPECT_EQ(ret[i], t);
	}
}

TEST(DancingLinks, find_front) {
	D d;
	std::vector<D::node *> ret;
	for (size_t i = 0; i < N; i++) {
		ret.emplace_back(d.push_front(i));
	}

	/// make the longest search possible
	for (size_t i = 0; i < N; i++) {
		const auto t = d.find_front(i);
		EXPECT_NE(t, nullptr);
		EXPECT_EQ(ret[i], t);
	}
}

TEST(DancingLinks, erase) {
	D d;
	std::vector<D::node *> ret;
	for (size_t i = 0; i < N; i++) {
		ret.emplace_back(d.push_front(i));
	}
	for (size_t i = 0; i < N; i++) {
		d.erase(ret[i]);
	}
	EXPECT_EQ(d.size(), 0);
}

TEST(DancingLinks, restore) {
	D d;
	std::vector<D::node *> ret;
	for (size_t i = 0; i < N; i++) {
		ret.emplace_back(d.push_front(i));
	}

	for (size_t i = 0; i < N; i++) {
		d.erase(ret[i]);
	}
	EXPECT_EQ(d.size(), 0);

	for (size_t i = 0; i < N; i++) {
		d.restore(ret[i]);
	}
	EXPECT_EQ(d.size(), N);
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
