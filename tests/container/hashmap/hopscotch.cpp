#include <gtest/gtest.h>
#include <iostream>

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
		// TODO it.value() += 2;
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

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
