#include <gtest/gtest.h>
#include <iostream>
#include <ranges>
#include <vector>

#include "container/hashmap.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

TEST(hopscotch, simple) {
	hopscotch_map<std::string, int> map = {{"a", 1}, {"b", 2}};
	map["c"] = 3;
	map["d"] = 4;

	map.insert({"e", 5});
	map.erase("b");

	for(auto it = map.begin(); it != map.end(); ++it) {
		// NOTE: not implemented
		// it.value() += 2;
	}

	// {d, 6} {a, 3} {e, 7} {c, 5}
	for(const auto& key_value : map) {
		std::cout << "{" << key_value.first << ", " << key_value.second << "}" << std::endl;
	}


	if(map.find("a") != map.end()) {
		std::cout << "Found \"a\"." << std::endl;
	}

	const std::size_t precalculated_hash = std::hash<std::string>()("a");
	// If we already know the hash beforehand, we can pass it in parameter to speed-up lookups.
	if(map.find("a", precalculated_hash) != map.end()) {
		std::cout << "Found \"a\" with hash " << precalculated_hash << "." << std::endl;
	}


	/*
     * Calculating the hash and comparing two std::string may be slow.
     * We can store the hash of each std::string in the hash map to make
     * the inserts and lookups faster by setting StoreHash to true.
     */
	hopscotch_map<std::string, int, std::hash<std::string>,
	                   std::equal_to<std::string>,
	                   std::allocator<std::pair<std::string, int>>,
	                   30, true> map2;

	map2["a"] = 1;
	map2["b"] = 2;

	// {a, 1} {b, 2}
	for(const auto& key_value : map2) {
		std::cout << "{" << key_value.first << ", " << key_value.second << "}" << std::endl;
	}

}

/// range constructor and insert_range
TEST(hopscotch, ranges) {
	std::vector<std::pair<int, int>> v;
	for (int i = 0; i < 1000; ++i) { v.emplace_back(i, i * i); }
	hopscotch_map<int, int> m(std::from_range, v);
	EXPECT_EQ(m.size(), 1000u);
	for (int i = 0; i < 1000; ++i) { EXPECT_EQ(m.at(i), i * i); }

	// a view, duplicates keep the first value
	auto odd = v | std::views::filter([](const auto &p) { return p.first % 2; });
	hopscotch_map<int, int> m2;
	m2.insert_range(odd);
	m2.insert_range(std::vector<std::pair<int, int>>{{1, -1}, {2000, 4}});
	EXPECT_EQ(m2.size(), 501u);
	EXPECT_EQ(m2.at(1), 1);
	EXPECT_EQ(m2.at(2000), 4);
	EXPECT_EQ(m2.count(2), 0u);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
