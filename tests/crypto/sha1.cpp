#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "helper.h"
#include "random.h"

#include "crypto/sha1.h"
using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;
// 
// using namespace cryptanalysislib;
// 
TEST(SHA1, simple) {
    static_assert(""_sha1          == "da39a3ee5e6b4b0d3255bfef95601890afd80709"_hex_bytes);
    static_assert("abc"_sha1       == "a9993e364706816aba3e25717850c26c9cd0d89d"_hex_bytes);
    static_assert("The quick brown fox jumps over the lazy dog"_sha1 == "2fd4e1c67a2d28fced849ee1bb76e7391b93eb12"_hex_bytes);
}

TEST(SHA1, multiblock) {
    // 56 bytes: the length no longer fits into the first block
    static_assert("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"_sha1 == "84983e441c3bd26ebaae4aa1f95129e5e54670f1"_hex_bytes);
    // 64 bytes: exactly one full message block
    static_assert("0123456701234567012345670123456701234567012345670123456701234567"_sha1 == "e0c094e867ef46c350ef54a7f59dd60bed92ae83"_hex_bytes);
    // 112 bytes
    static_assert("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"_sha1 == "a49b2446a02c645bf419f995b67091253a04a259"_hex_bytes);
}


int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
