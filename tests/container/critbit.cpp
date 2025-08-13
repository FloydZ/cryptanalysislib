#include <gtest/gtest.h>
#include <cstdio>
#include <string>

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

	K k{"abc"};

    T() noexcept : value(7) {}
    T(uint32_t v) noexcept : value(v) {}
};

/// Compare class to compare two strings
class Cmp {
public:
	constexpr inline int operator()(const K *k1,
                                    const K *k2) const noexcept {
		std::cout << 'cmp()' << std::endl;
		std::cout << *k1 << std::endl;
		std::cout << *k2 << std::endl;
        return k1->compare(*k2);
	}
};

/// Helper class computing the length of a key
class KLen {
public:
	constexpr inline int operator()(const K *k1) const noexcept {
        return k1->length();
	}
};

/// Helper class hashing a value T to a key K
class THash {
public:
	constexpr inline K operator()(const T *k1) const noexcept {
		return k1->k;
	}
};

/// Helper class accessing the binary data of a key k
class KBinary {
public:
	constexpr inline uint8_t*operator()(const K *k1) const noexcept {
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
    // Test that we can create a critbit tree
    void* freearg = nullptr;
    
    // Create a new critbit_tree
    CritBitTree tree{};
    
    // At this point we should have an empty tree
    // Since most methods are private, we need to test indirectly
    // For example, we can verify that a lookup returns nullptr
    
    K key("a");
    K key2("ab");
	T t;
	T t2; t2.k = key2;

    using critbit_node = CritBitTree::critbit_node;
    critbit_node *nnode = (critbit_node *)malloc(sizeof(critbit_node));
    tree.critbit_insert(nnode, &t);

    critbit_node *nnode2 = (critbit_node *)malloc(sizeof(critbit_node));
    auto r1 = tree.critbit_insert(nnode2, &t2);

    auto r2= tree.critbit_get_impl(&key);
    
    // EXPECT_NE(r1, nullptr);
    EXPECT_NE(r2, nullptr);
    std::cout << r2->value << std::endl;
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

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
