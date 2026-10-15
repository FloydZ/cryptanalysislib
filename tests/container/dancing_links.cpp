#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>
#include <vector>

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

/// iteration over the linked nodes, also after erase/restore and backwards
TEST(DancingLinks, iterator) {
	D d;
	std::vector<D::node *> ret;
	for (size_t i = 0; i < 10; i++) { ret.emplace_back(d.push_back(i)); }
	d.erase(ret[0]); d.erase(ret[5]); d.erase(ret[9]);
	EXPECT_EQ(std::vector<T>(d.begin(), d.end()), (std::vector<T>{1, 2, 3, 4, 6, 7, 8}));

	std::vector<T> rev;
	for (auto it = d.end(); it != d.begin();) { --it; rev.push_back(*it); }
	EXPECT_EQ(rev, (std::vector<T>{8, 7, 6, 4, 3, 2, 1}));

	d.restore(ret[9]); d.restore(ret[5]); d.restore(ret[0]);
	EXPECT_EQ(std::vector<T>(d.begin(), d.end()), (std::vector<T>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}));

	for (auto &x : d) { x *= 2; }
	const D &cd = d;
	T sum = 0;
	for (const auto x : cd) { sum += x; }
	EXPECT_EQ(sum, 90u);
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
