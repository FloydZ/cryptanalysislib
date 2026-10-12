#include <gtest/gtest.h>
#include <cstdio>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "container/critbit.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

// Simple struct to use as a critbit_key for testing
//struct TestKey {
//    uint8_t data[16];
//
//    TestKey() noexcept {
//        memset(data, 0, sizeof(data));
//    }
//
//    TestKey(uint32_t value) noexcept {
//        memset(data, 0, sizeof(data));
//        data[0] = (value >> 24) & 0xFF;
//        data[1] = (value >> 16) & 0xFF;
//        data[2] = (value >> 8) & 0xFF;
//        data[3] = value & 0xFF;
//    }
//
//    bool operator==(const TestKey& other) const noexcept {
//        return memcmp(data, other.data, sizeof(data)) == 0;
//    }
//};

using K = std::string;

// Reference class to use as critbit_ref in our tests
class T {
public:
    uint32_t value;

	K k{"a"};

    T() noexcept : value(7) {}
    explicit T(const uint32_t v) noexcept : value(v) {}
};

/// Compare class to compare two strings
class Cmp {
public:
	constexpr inline int operator()(const K *k1,
                                    const K *k2) const noexcept {
        return k1->compare(*k2);
	}
};

/// Helper class computing the length of a key
class KLen {
public:
	constexpr inline size_t operator()(const K *k1) const noexcept {
        return k1->length();
	}
};

/// Helper class hashing a value T to a key K
class THash {
public:
	constexpr inline const K* operator()(const T *k1) const noexcept {
		return &(k1->k);
	}
};

/// Helper class accessing the binary data of a key k
class KBinary {
public:
	constexpr inline uint8_t* operator()(const K *k1) const noexcept {
		return (uint8_t *)k1->data();
	}
};

// For the tests, we'll specialize the critbit_tree class
using CritBitTree = critbit_tree<T, K, THash, Cmp, KLen, KBinary>;

class CritBitTest : public testing::Test {
protected:
    void SetUp() override {
        // Any setup needed for tests
    }

    void TearDown() override {
        // Any cleanup needed after tests
    }
};

TEST_F(CritBitTest, Initialization) {
    CritBitTree tree{};
    
    // At this point we should have an empty tree
    // Since most methods are private, we need to test indirectly
    // For example, we can verify that a lookup returns nullptr
    
    K key("a");
    K key2("aa");
	T t;
	t.k = key;
	T t2; t2.k = key2;

    using critbit_node = CritBitTree::critbit_node;
    critbit_node *nnode = (critbit_node *)calloc(1, sizeof(critbit_node));
    tree.critbit_insert_impl(nnode, &t);

    critbit_node *nnode2 = (critbit_node *)calloc(1, sizeof(critbit_node));
    auto r1 = tree.critbit_insert_impl(nnode2, &t2);

    auto r2= tree.critbit_get_impl(&key);
    
    // EXPECT_NE(r1, nullptr);
    EXPECT_NE(r2, nullptr);
    std::cout << r2->value << std::endl;

	K *s1 = tree.critbit_remove_impl(&key);
	K *s2 = tree.critbit_remove_impl(&key2);
	std::cout << *s1 << std::endl;
	std::cout << *s2 << std::endl;
}

// This test would test the lookup functionality if it was implemented
/* 
TEST_F(CritBitTest, LookupTest) {
    void* freearg = nullptr;
    size_t keylen = sizeof(TestKey);
    
    CritBitTree tree(freearg, keylen);
    
    // Insert some keys
    TestKey key1(1);
    TestKey key2(2);
    TestKey key3(3);
    
    // Insert keys (this would require the insert method to be public)
    // tree.insert(key1);
    // tree.insert(key2);
    // tree.insert(key3);
    
    // Look up a key that exists
    auto result1 = tree.critbit_get_impl(&key1, keylen, CritBitTree::critbit_buf_keycmp, CritBitTree::critbit_buf_keybuf);
    EXPECT_NE(result1, nullptr);
    
    // Look up a key that doesn't exist
    TestKey key4(4);
    auto result2 = tree.critbit_get_impl(&key4, keylen, CritBitTree::critbit_buf_keycmp, CritBitTree::critbit_buf_keybuf);
    EXPECT_EQ(result2, nullptr);
}
*/

/// random strings: insert, lookup and remove against std::set
TEST(critbit, random_vs_std_set) {
	for (const uint32_t alphabet : {2u, 26u}) {
		CritBitTree tree{};
		std::set<std::string> e;
		std::vector<std::unique_ptr<T>> values;
		for (size_t i = 0; i < 3000; ++i) {
			std::string s;
			const size_t len = 1 + rand() % 6;
			for (size_t j = 0; j < len; ++j) { s.push_back(char('a' + rand() % alphabet)); }
			auto t = std::make_unique<T>(uint32_t(i));
			t->k = s;
			auto *node = (CritBitTree::critbit_node *)calloc(1, sizeof(CritBitTree::critbit_node));
			const K *existing = tree.critbit_insert_impl(node, t.get());
			const bool fresh = e.insert(s).second;
			EXPECT_EQ(existing == nullptr, fresh) << s;
			if (fresh) { values.push_back(std::move(t)); }
		}
		for (const auto &v : values) {
			const T *r = tree.critbit_get_impl(&v->k);
			ASSERT_NE(r, nullptr) << v->k;
			EXPECT_EQ(r->value, v->value);
		}
		const K missing("zzzzzzzz");
		EXPECT_EQ(tree.critbit_get_impl(&missing), nullptr);

		// remove half, the rest must still be found
		for (size_t i = 0; i < values.size(); i += 2) {
			const K *k = tree.critbit_remove_impl(&values[i]->k);
			ASSERT_NE(k, nullptr);
			EXPECT_EQ(*k, values[i]->k);
			EXPECT_EQ(tree.critbit_get_impl(&values[i]->k), nullptr);
		}
		for (size_t i = 1; i < values.size(); i += 2) {
			EXPECT_NE(tree.critbit_get_impl(&values[i]->k), nullptr) << values[i]->k;
		}

		// all elements with the prefix "ab"
		std::set<std::string> got, expect;
		for (size_t i = 1; i < values.size(); i += 2) {
			if (values[i]->k.rfind("ab", 0) == 0) { expect.insert(values[i]->k); }
		}
		tree.allprefixed((const uint8_t *)"ab", 2, [&](const T *r) { got.insert(r->k); return true; });
		EXPECT_EQ(got, expect);
		// stopping the traversal: the handle returns false at the third call
		size_t calls = 0;
		EXPECT_FALSE(tree.allprefixed((const uint8_t *)"", 0, [&](const T *) { return ++calls < 3; }));
		EXPECT_EQ(calls, 3u);

		tree.clear();
		EXPECT_EQ(tree.critbit_get_impl(&values[1]->k), nullptr);
	}
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
