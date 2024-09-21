#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "helper.h"
#include "random.h"
#include "container/imap.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

constexpr uint64_t N = 10000000;

static int u32cmp(const void *x,
                  const void *y) {
	return (int)(*(uint32_t *)x - *(uint32_t *)y);
}

static uint32_t *test_bsearch(const uint32_t x,
                              uint32_t *array,
                              const int count) {
	int lo = 0, hi = count - 1, mi;
	int di;

	while (lo <= hi) {
		mi = (unsigned)(lo + hi) >> 1;
		di = array[mi] - x;
		if (0 > di)
			lo = mi + 1;
		else if (0 < di)
			hi = mi - 1;
		else
			return &array[mi];
	}
	return &array[lo];
}

// TODO: more tests
TEST(imap, first) {
	imap_tree_t tree;

	auto *slot = tree.lookup(0xA0000056);
	EXPECT_EQ(nullptr, slot);
	slot = tree.assign(0xA0000056);
	EXPECT_NE(nullptr, slot);

	EXPECT_FALSE(tree.hasval(slot));
	EXPECT_EQ(0, tree.getval(slot));
	tree.setval(slot, 0x56);
	EXPECT_TRUE(tree.hasval(slot));
	EXPECT_EQ(0x56, tree.getval(slot));

	slot = tree.lookup(0xA0000056);
	EXPECT_NE(nullptr, slot);
	tree.delval(slot);
	EXPECT_FALSE(tree.hasval(slot));
	slot = tree.lookup(0xA0000056);
	EXPECT_EQ(nullptr, slot);
}

//static void imap_primitives_test() {
//	uint32_t vec32[16];
//	uint32_t val32;
//	uint64_t val64;
//
//	memset(vec32, 0, sizeof vec32);
//	val64 = 0xFEDCBA9876543210;
//	imap__deposit_lo4__(vec32, val64);
//	val64 = imap__extract_lo4__(vec32);
//	EXPECT_EQ(0xFEDCBA9876543210ull == val64);
//
//	memset(vec32, 0, sizeof vec32);
//	EXPECT_EQ(0 == imap__popcnt_hi28__(vec32, &val32));
//	memset(vec32, 0, sizeof vec32);
//	vec32[0] = 0xff;
//	EXPECT_EQ(1 == imap__popcnt_hi28__(vec32, &val32) && 0xff == val32);
//	memset(vec32, 0, sizeof vec32);
//	vec32[1] = 0xef, vec32[3] = 0xd0;
//	EXPECT_EQ(2 == imap__popcnt_hi28__(vec32, &val32));
//	memset(vec32, 0, sizeof vec32);
//	vec32[3] = 0xd0;
//	EXPECT_EQ(1 == imap__popcnt_hi28__(vec32, &val32) && 0xd0 == val32);
//}

TEST(imap, assign) {
	uint32_t *slot;
	{
		imap_tree_t tree;
		slot = tree.lookup(0xA0000056);
		EXPECT_EQ(nullptr, slot);
		slot = tree.assign(0xA0000056);
		EXPECT_NE(nullptr, slot);
		EXPECT_TRUE(!tree.hasval(slot));
		EXPECT_EQ(0, tree.getval(slot));
		tree.setval(slot, 0x56);
		EXPECT_TRUE(tree.hasval(slot));
		EXPECT_EQ(0x56, tree.getval(slot));
		slot = tree.lookup(0xA0000056);
		EXPECT_NE(nullptr, slot);
		tree.delval(slot);
		EXPECT_TRUE(!tree.hasval(slot));
		slot = tree.lookup(0xA0000056);
		EXPECT_EQ(nullptr, slot);
	}
	{
		imap_tree_t tree(2);
		slot = tree.assign(0xA0000056);
		EXPECT_NE(nullptr, slot);
		tree.setval(slot, 0x56);
		slot = tree.assign(0xA0000057);
		EXPECT_NE(nullptr, slot);
		tree.setval(slot, 0x57);
		slot = tree.lookup(0xA0000056);
		EXPECT_NE(nullptr, slot);
		EXPECT_EQ(0x56, tree.getval(slot));
		tree.delval(slot);
		slot = tree.lookup(0xA0000057);
		EXPECT_NE(nullptr, slot);
		EXPECT_EQ(0x57, tree.getval(slot));
		tree.delval(slot);
		slot = tree.lookup(0xA0000056);
		EXPECT_EQ(0, slot);
		slot = tree.lookup(0xA0000057);
		EXPECT_EQ(0, slot);
	}

//	tree = 0;
//	tree = tree.ensure(+3);
//
//	slot = tree.assign(0xA0000056);
//	EXPECT_EQ(nullptr, slot);
//	tree.setval(slot, 0x56);
//	slot = tree.assign(0xA0000057);
//	EXPECT_EQ(nullptr, slot);
//	tree.setval(slot, 0x57);
//	slot = tree.assign(0xA0008009);
//	EXPECT_EQ(nullptr, slot);
//	tree.setval(slot, 0x8009);
//	slot = tree.lookup(0xA0000056);
//	EXPECT_EQ(nullptr, slot);
//	EXPECT_EQ(0x56 == tree.getval(slot));
//	tree.delval(slot);
//	slot = tree.lookup(0xA0000057);
//	EXPECT_EQ(nullptr, slot);
//	EXPECT_EQ(0x57 == tree.getval(slot));
//	tree.delval(slot);
//	slot = tree.lookup(0xA0008009);
//	EXPECT_EQ(nullptr, slot);
//	EXPECT_EQ(0x8009 == tree.getval(slot));
//	tree.delval(slot);
//	slot = tree.lookup(0xA0000056);
//	EXPECT_EQ(0 == slot);
//	slot = tree.lookup(0xA0000057);
//	EXPECT_EQ(0 == slot);
//	slot = tree.lookup(0xA0008009);
//	EXPECT_EQ(0 == slot);
//	imap_free(tree);
//
//	tree = 0;
//	tree = tree.ensure(+4);
//
//	slot = tree.assign(0xA0000056);
//	EXPECT_EQ(nullptr, slot);
//	tree.setval(slot, 0x56);
//	slot = tree.assign(0xA0000057);
//	EXPECT_EQ(nullptr, slot);
//	tree.setval(slot, 0x57);
//	slot = tree.assign(0xA0008009);
//	EXPECT_EQ(nullptr, slot);
//	tree.setval(slot, 0x8009);
//	slot = tree.assign(0xA0008059);
//	EXPECT_EQ(nullptr, slot);
//	tree.setval(slot, 0x8059);
//	slot = tree.lookup(0xA0000056);
//	EXPECT_EQ(nullptr, slot);
//	EXPECT_EQ(0x56 == tree.getval(slot));
//	tree.delval(slot);
//	slot = tree.lookup(0xA0000057);
//	EXPECT_EQ(nullptr, slot);
//	EXPECT_EQ(0x57 == tree.getval(slot));
//	tree.delval(slot);
//	slot = tree.lookup(0xA0008009);
//	EXPECT_EQ(nullptr, slot);
//	EXPECT_EQ(0x8009 == tree.getval(slot));
//	tree.delval(slot);
//	slot = tree.lookup(0xA0008059);
//	EXPECT_EQ(nullptr, slot);
//	EXPECT_EQ(0x8059 == tree.getval(slot));
//	tree.delval(slot);
//	slot = tree.lookup(0xA0000056);
//	EXPECT_EQ(0 == slot);
//	slot = tree.lookup(0xA0000057);
//	EXPECT_EQ(0 == slot);
//	slot = tree.lookup(0xA0008009);
//	EXPECT_EQ(0 == slot);
//	slot = tree.lookup(0xA0008059);
//	EXPECT_EQ(0 == slot);
//	imap_free(tree);
//
//	tree = 0;
//	tree = tree.ensure(+5);
//
//	slot = tree.assign(0xA0000056);
//	EXPECT_EQ(nullptr, slot);
//	tree.setval(slot, 0x56);
//	slot = tree.assign(0xA0000057);
//	EXPECT_EQ(nullptr, slot);
//	tree.setval(slot, 0x57);
//	slot = tree.assign(0xA0008009);
//	EXPECT_EQ(nullptr, slot);
//	tree.setval(slot, 0x8009);
//	slot = tree.assign(0xA0008059);
//	EXPECT_EQ(nullptr, slot);
//	tree.setval(slot, 0x8059);
//	slot = tree.assign(0xA0008069);
//	EXPECT_EQ(nullptr, slot);
//	tree.setval(slot, 0x8069);
//	slot = tree.lookup(0xA0000056);
//	EXPECT_EQ(nullptr, slot);
//	EXPECT_EQ(0x56 == tree.getval(slot));
//	tree.delval(slot);
//	slot = tree.lookup(0xA0000057);
//	EXPECT_EQ(nullptr, slot);
//	EXPECT_EQ(0x57 == tree.getval(slot));
//	tree.delval(slot);
//	slot = tree.lookup(0xA0008009);
//	EXPECT_EQ(nullptr, slot);
//	EXPECT_EQ(0x8009 == tree.getval(slot));
//	tree.delval(slot);
//	slot = tree.lookup(0xA0008059);
//	EXPECT_EQ(nullptr, slot);
//	EXPECT_EQ(0x8059 == tree.getval(slot));
//	tree.delval(slot);
//	slot = tree.lookup(0xA0008069);
//	EXPECT_EQ(nullptr, slot);
//	EXPECT_EQ(0x8069 == tree.getval(slot));
//	tree.delval(slot);
//	slot = tree.lookup(0xA0000056);
//	EXPECT_EQ(0 == slot);
//	slot = tree.lookup(0xA0000057);
//	EXPECT_EQ(0 == slot);
//	slot = tree.lookup(0xA0008009);
//	EXPECT_EQ(0 == slot);
//	slot = tree.lookup(0xA0008059);
//	EXPECT_EQ(0 == slot);
//	slot = tree.lookup(0xA0008069);
//	EXPECT_EQ(0 == slot);
//	imap_free(tree);
}

TEST(imap, assign_bigval) {
	const unsigned N = 100;
	imap_tree_t tree(1);
	uint32_t *slot;

	for (uint32_t i = 0; N > i; i++) {
		tree.ensure(+1);
		slot = tree.assign(i);
		EXPECT_NE(nullptr, slot);
		tree.setval(slot, 0x8000000000000000ull | i);
	}
	for (unsigned i = 0; N > i; i++) {
		slot = tree.lookup(i);
		EXPECT_NE(nullptr, slot);
		EXPECT_TRUE(tree.hasval(slot));
		EXPECT_EQ((0x8000000000000000ull | i), tree.getval(slot));
	}

	for (unsigned i = 0; N > i; i++) {
		tree.ensure(+1);
		slot = tree.assign(i);
		EXPECT_NE(nullptr, slot);
		tree.setval(slot, i);
	}

	for (unsigned i = 0; N > i; i++) {
		slot = tree.lookup(i);
		EXPECT_NE(nullptr, slot);
		EXPECT_TRUE(tree.hasval(slot));
		EXPECT_EQ(i, tree.getval(slot));
	}

	for (unsigned i = 0; N > i; i++) {
		tree.ensure(+1);
		slot = tree.assign(i);
		EXPECT_NE(nullptr, slot);
		tree.setval(slot, 0x8000000000000000ull | i);
	}

	for (unsigned i = 0; N > i; i++) {
		slot = tree.lookup(i);
		EXPECT_NE(nullptr, slot);
		EXPECT_TRUE(tree.hasval(slot));
		EXPECT_EQ((0x8000000000000000ull | i) , tree.getval(slot));
	}

	for (unsigned i = 0; N > i; i++) {
		slot = tree.lookup(i);
		EXPECT_NE(nullptr, slot);
		tree.delval(slot);
		EXPECT_TRUE(!tree.hasval(slot));
	}

	for (unsigned i = 0; N > i; i++){
		slot = tree.lookup(i);
		EXPECT_EQ(nullptr, slot);
	}

	for (unsigned i = 0; N > i; i++) {
		tree.ensure(+1);
		slot = tree.assign(i);
		EXPECT_NE(nullptr, slot);
		tree.setval(slot, 0x8000000000000000ull | i);
	}

	for (unsigned i = 0; N > i; i++) {
		slot = tree.lookup(i);
		EXPECT_NE(nullptr, slot);
		EXPECT_TRUE(tree.hasval(slot));
		EXPECT_EQ((0x8000000000000000ull | i), tree.getval(slot));
	}
}

TEST(imap, assign_val0) {
	const unsigned N = 100;
	imap_tree_t tree(1);
	uint32_t *slot;

	for (unsigned i = 0; N > i; i++) {
	 	tree.ensure0(+1);
	 	slot = tree.assign(i);
	 	EXPECT_NE(nullptr, slot);
		tree.setval0(slot, 0x800000ull | i);
	}

	for (unsigned i = 0; N > i; i++) {
		slot = tree.lookup(i);
		EXPECT_NE(nullptr, slot);
		EXPECT_TRUE(tree.hasval(slot));
		EXPECT_EQ((0x800000ull | i), tree.getval0(slot));
	}

	for (unsigned i = 0; N > i; i++) {
		tree.ensure0(+1);
		slot = tree.assign(i);
		EXPECT_NE(nullptr, slot);
		tree.setval0(slot, i);
	}

	for (unsigned i = 0; N > i; i++) {
		slot = tree.lookup(i);
		EXPECT_NE(nullptr, slot);
		EXPECT_TRUE(tree.hasval(slot));
		EXPECT_EQ(i, tree.getval0(slot));
	}

	for (unsigned i = 0; N > i; i++) {
		tree.ensure0(+1);
		slot = tree.assign(i);
		EXPECT_NE(nullptr, slot);
		tree.setval0(slot, 0x800000ull | i);
	}

	for (unsigned i = 0; N > i; i++) {
		slot = tree.lookup(i);
		EXPECT_NE(nullptr, slot);
		EXPECT_TRUE(tree.hasval(slot));
		EXPECT_EQ((0x800000ull | i), tree.getval0(slot));
	}

	for (unsigned i = 0; N > i; i++) {
		slot = tree.lookup(i);
		EXPECT_NE(nullptr, slot);
		tree.delval(slot);
		EXPECT_TRUE(!tree.hasval(slot));
	}

	for (unsigned i = 0; N > i; i++) {
		slot = tree.lookup(i);
		EXPECT_EQ(nullptr, slot);
	}

	for (unsigned i = 0; N > i; i++) {
		tree.ensure0(+1);
		slot = tree.assign(i);
		EXPECT_NE(nullptr, slot);
		tree.setval0(slot, 0x800000ull | i);
	}

	for (unsigned i = 0; N > i; i++) {
		slot = tree.lookup(i);
		EXPECT_NE(nullptr, slot);
		EXPECT_TRUE(tree.hasval(slot));
		EXPECT_EQ((0x800000ull | i), tree.getval0(slot));
	}
}

TEST(imap, assign_val64) {
	const unsigned N = 100;
	imap_tree_t tree(1);
	uint32_t *slot;

	for (unsigned i = 0; N > i; i++) {
		tree.ensure64(+1);
		slot = tree.assign(i);
		EXPECT_NE(nullptr, slot);
		tree.setval64(slot, 0x8000000000000000ull | i);
	}

	for (unsigned i = 0; N > i; i++) {
		slot = tree.lookup(i);
		EXPECT_NE(nullptr, slot);
		EXPECT_TRUE(tree.hasval(slot));
		EXPECT_EQ((0x8000000000000000ull | i), tree.getval64(slot));
	}

	for (unsigned i = 0; N > i; i++) {
		tree.ensure64(+1);
		slot = tree.assign(i);
		EXPECT_NE(nullptr, slot);
		tree.setval64(slot, i);
	}

	for (unsigned i = 0; N > i; i++) {
		slot = tree.lookup(i);
		EXPECT_NE(nullptr, slot);
		EXPECT_TRUE(tree.hasval(slot));
		EXPECT_EQ(i, tree.getval64(slot));
	}

	for (unsigned i = 0; N > i; i++) {
		tree.ensure64(+1);
		slot = tree.assign(i);
		EXPECT_NE(nullptr, slot);
		tree.setval64(slot, 0x8000000000000000ull | i);
	}

	for (unsigned i = 0; N > i; i++) {
		slot = tree.lookup(i);
		EXPECT_NE(nullptr, slot);
		EXPECT_TRUE(tree.hasval(slot));
		EXPECT_EQ((0x8000000000000000ull | i), tree.getval64(slot));
	}

	for (unsigned i = 0; N > i; i++) {
		slot = tree.lookup(i);
		EXPECT_NE(nullptr, slot);
		tree.delval(slot);
		EXPECT_TRUE(!tree.hasval(slot));
	}

	for (unsigned i = 0; N > i; i++) {
		slot = tree.lookup(i);
		EXPECT_EQ(nullptr, slot);
	}

	for (unsigned i = 0; N > i; i++) {
		tree.ensure(+1);
		slot = tree.assign(i);
		EXPECT_NE(nullptr, slot);
		tree.setval64(slot, 0x8000000000000000ull | i);
	}

	for (unsigned i = 0; N > i; i++) {
		slot = tree.lookup(i);
		EXPECT_NE(nullptr, slot);
		EXPECT_TRUE(tree.hasval(slot));
		EXPECT_EQ((0x8000000000000000ull | i), tree.getval64(slot));
	}
}

// TODO not working, something with ensure is not right
//TEST(imap, assign_val128) {
//	const unsigned N = 100;
//	imap_tree_t tree(1);
//	uint32_t *slot;
//	imap_u128_t val128;
//
//	for (unsigned i = 0; N > i; i++) {
//		tree.ensure128(+1);
//		slot = tree.assign(i);
//		EXPECT_NE(nullptr, slot);
//		val128.v[0] = 0x8000000000000000ull | i;
//		val128.v[1] = 0x9000000000000000ull | i;
//		tree.setval128(slot, val128);
//	}
//
//	for (unsigned i = 0; N > i; i++) {
//		slot = tree.lookup(i);
//		EXPECT_NE(nullptr, slot);
//		EXPECT_TRUE(tree.hasval(slot));
//		EXPECT_EQ((0x8000000000000000ull | i), tree.getval128(slot).v[0]);
//		EXPECT_EQ((0x9000000000000000ull | i), tree.getval128(slot).v[1]);
//	}
//
//	for (unsigned i = 0; N > i; i++) {
//		tree.ensure128(+1);
//		slot = tree.assign(i);
//		EXPECT_NE(nullptr, slot);
//		val128.v[0] = i;
//		val128.v[1] = i;
//		tree.setval128(slot, val128);
//	}
//
//	for (unsigned i = 0; N > i; i++) {
//		slot = tree.lookup(i);
//		EXPECT_EQ(nullptr, slot);
//		EXPECT_TRUE(tree.hasval(slot));
//		EXPECT_EQ(i, tree.getval128(slot).v[0]);
//		EXPECT_EQ(i, tree.getval128(slot).v[1]);
//	}
//
//	for (unsigned i = 0; N > i; i++) {
//		tree.ensure(+1);
//		slot = tree.assign(i);
//		EXPECT_EQ(nullptr, slot);
//		val128.v[0] = 0x8000000000000000ull | i;
//		val128.v[1] = 0x9000000000000000ull | i;
//		tree.setval128(slot, val128);
//	}
//
//	for (unsigned i = 0; N > i; i++) {
//		slot = tree.lookup(i);
//		EXPECT_NE(nullptr, slot);
//		EXPECT_TRUE(tree.hasval(slot));
//		EXPECT_EQ((0x8000000000000000ull | i), tree.getval128(slot).v[0]);
//		EXPECT_EQ((0x9000000000000000ull | i), tree.getval128(slot).v[1]);
//	}
//
//	for (unsigned i = 0; N > i; i++) {
//		slot = tree.lookup(i);
//		EXPECT_EQ(nullptr, slot);
//		tree.delval(slot);
//		EXPECT_TRUE(!tree.hasval(slot));
//	}
//
//	for (unsigned i = 0; N > i; i++) {
//		slot = tree.lookup(i);
//		EXPECT_NE(nullptr, slot);
//	}
//
//	for (unsigned i = 0; N > i; i++) {
//		tree.ensure(+1);
//		slot = tree.assign(i);
//		EXPECT_NE(nullptr, slot);
//		val128.v[0] = 0x8000000000000000ull | i;
//		val128.v[1] = 0x9000000000000000ull | i;
//		tree.setval128(slot, val128);
//	}
//
//	for (unsigned i = 0; N > i; i++) {
//		slot = tree.lookup(i);
//		EXPECT_EQ(nullptr, slot);
//		EXPECT_TRUE(tree.hasval(slot));
//		EXPECT_EQ((0x8000000000000000ull | i), tree.getval128(slot).v[0]);
//		EXPECT_EQ((0x9000000000000000ull | i), tree.getval128(slot).v[1]);
//	}
//}

TEST(imap, assign_shuffle) {
	const unsigned N = 10000000;
	uint32_t *array;
	imap_tree_t tree(1);
	uint32_t *slot;

	array = (uint32_t *)malloc(N * sizeof(uint32_t));
	EXPECT_NE(nullptr, array);

	for (unsigned i = 0; N > i; i++) {
		array[i] = i;
	}

	for (unsigned i = 0; N > i; i++) {
		uint32_t r = rng() % N;
		uint32_t t = array[i];
		array[i] = array[r];
		array[r] = t;
	}

	for (unsigned i = 0; N > i; i++){
		tree.ensure(+1);
		slot = tree.assign(array[i]);
		EXPECT_NE(nullptr, slot);
		tree.setval(slot, i);
	}

	for (unsigned i = 0; N > i; i++){
		slot = tree.lookup(array[i]);
		EXPECT_NE(nullptr, slot);
		EXPECT_EQ(i, tree.getval(slot));
	}

	free(array);
}

TEST(imap, remove) {
	imap_tree_t tree(1);
	uint32_t *slot;

	tree.ensure(+5);
	slot = tree.assign(0xA0000056);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 0x56);
	slot = tree.assign(0xA0000057);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 0x57);
	slot = tree.assign(0xA0008009);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 0x8009);
	slot = tree.assign(0xA0008059);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 0x8059);
	slot = tree.assign(0xA0008069);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 0x8069);

	slot = tree.lookup(0xA0000056);
	EXPECT_NE(nullptr, slot);
	EXPECT_EQ(0x56, tree.getval(slot));
	tree.remove(0xA0000056);
	slot = tree.lookup(0xA0000056);
	EXPECT_EQ(nullptr, slot);

	slot = tree.lookup(0xA0000057);
	EXPECT_NE(nullptr, slot);
	EXPECT_EQ(0x57, tree.getval(slot));
	tree.remove(0xA0000057);
	slot = tree.lookup(0xA0000057);
	EXPECT_EQ(nullptr, slot);

	slot = tree.lookup(0xA0008009);
	EXPECT_NE(nullptr, slot);
	EXPECT_EQ(0x8009, tree.getval(slot));
	tree.remove(0xA0008009);
	slot = tree.lookup(0xA0008009);
	EXPECT_EQ(nullptr, slot);

	slot = tree.lookup(0xA0008059);
	EXPECT_NE(nullptr, slot);
	EXPECT_EQ(0x8059, tree.getval(slot));
	tree.remove(0xA0008059);
	slot = tree.lookup(0xA0008059);
	EXPECT_EQ(nullptr, slot);

	slot = tree.lookup(0xA0008069);
	EXPECT_NE(nullptr, slot);
	EXPECT_EQ(0x8069, tree.getval(slot));
	tree.remove(0xA0008069);
	slot = tree.lookup(0xA0008069);
	EXPECT_EQ(nullptr, slot);

	// uint64_t mark = tree->vec32[imap__tree_mark__];

	tree.ensure(+5);
	slot = tree.assign(0xA0000056);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 0x56);
	slot = tree.assign(0xA0000057);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 0x57);
	slot = tree.assign(0xA0008009);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 0x8009);
	slot = tree.assign(0xA0008059);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 0x8059);
	slot = tree.assign(0xA0008069);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 0x8069);
	slot = tree.lookup(0xA0000056);
	EXPECT_NE(nullptr, slot);
	EXPECT_EQ(0x56, tree.getval(slot));
	slot = tree.lookup(0xA0000057);
	EXPECT_NE(nullptr, slot);
	EXPECT_EQ(0x57, tree.getval(slot));
	slot = tree.lookup(0xA0008009);
	EXPECT_NE(nullptr, slot);
	EXPECT_EQ(0x8009, tree.getval(slot));
	slot = tree.lookup(0xA0008059);
	EXPECT_NE(nullptr, slot);
	EXPECT_EQ(0x8059, tree.getval(slot));
	slot = tree.lookup(0xA0008069);
	EXPECT_NE(nullptr, slot);
	EXPECT_EQ(0x8069, tree.getval(slot));
}

TEST(imap, remove_shuffle) {
	uint32_t *array;
	imap_tree_t tree (1);
	uint32_t *slot;

	array = (uint32_t *)malloc(N * sizeof(uint32_t));
	EXPECT_NE(nullptr, array);

	for (unsigned i = 0; N > i; i++) {
		array[i] = i;
	}

	for (unsigned i = 0; N > i; i++){
		tree.ensure(+1);
		slot = tree.assign(array[i]);
		EXPECT_NE(nullptr, slot);
		tree.setval(slot, i);
	}

	for (unsigned i = 0; N > i; i++) {
		slot = tree.lookup(array[i]);
		EXPECT_NE(nullptr, slot);
		EXPECT_EQ(i, tree.getval(slot));
	}

	for (unsigned i = 0; N > i; i++) {
		uint32_t r = rng() % N;
		uint32_t t = array[i];
		array[i] = array[r];
		array[r] = t;
	}

	for (unsigned i = 0; N / 2 > i; i++) {
		tree.remove(array[i]);
	}

	for (unsigned i = 0; N / 2 > i; i++) {
		slot = tree.lookup(array[i]);
		EXPECT_EQ(nullptr, slot);
	}

	for (unsigned i = N / 2; N > i; i++) {
		slot = tree.lookup(array[i]);
		EXPECT_NE(nullptr, slot);
		EXPECT_EQ(array[i], tree.getval(slot));
	}

	for (unsigned i = N / 2; N > i; i++) {
		tree.remove(array[i]);
	}

	for (unsigned i = N / 2; N > i; i++) {
		slot = tree.lookup(array[i]);
		EXPECT_EQ(nullptr, slot);
	}

	free(array);
}

TEST(imap, remove_iterate) {
	imap_tree_t tree(1);
	uint32_t *slot;
	imap_iter_t iter;
	imap_pair_t pair;

	tree.ensure(+1);
	pair = tree.iterate( &iter, 1);
	EXPECT_EQ(0, pair.x);
	EXPECT_EQ(0, pair.slot);

	tree.ensure(+5);

	slot = tree.assign(0xA0000056);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 0x56);
	slot = tree.assign(0xA0000057);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 0x57);
	slot = tree.assign(0xA0008009);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 0x8009);
	slot = tree.assign(0xA0008059);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 0x8059);
	slot = tree.assign(0xA0008069);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 0x8069);
	//
	pair = tree.iterate( &iter, 1);
	EXPECT_EQ(0xA0000056, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x56, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_EQ(0xA0000057, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x57, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_EQ(0xA0008009, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x8009, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_EQ(0xA0008059, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x8059, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_EQ(0xA0008069, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x8069, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_EQ(0, pair.x);
	EXPECT_EQ(nullptr, pair.slot);
	//
	slot = tree.lookup(0xA0000056);
	EXPECT_NE(nullptr, slot);
	EXPECT_EQ(0x56, tree.getval(slot));
	tree.delval(slot);
	slot = tree.lookup(0xA0000057);
	EXPECT_NE(nullptr, slot);
	EXPECT_EQ(0x57, tree.getval(slot));
	tree.delval(slot);
	slot = tree.lookup(0xA0008009);
	EXPECT_NE(nullptr, slot);
	EXPECT_EQ(0x8009, tree.getval(slot));
	tree.delval(slot);
	slot = tree.lookup(0xA0008059);
	EXPECT_NE(nullptr, slot);
	EXPECT_EQ(0x8059, tree.getval(slot));
	tree.delval(slot);
	slot = tree.lookup(0xA0008069);
	EXPECT_NE(nullptr, slot);
	EXPECT_EQ(0x8069, tree.getval(slot));
	tree.delval(slot);
	slot = tree.lookup(0xA0000056);
	EXPECT_EQ(nullptr, slot);
	slot = tree.lookup(0xA0000057);
	EXPECT_EQ(nullptr, slot);
	slot = tree.lookup(0xA0008009);
	EXPECT_EQ(nullptr, slot);
	slot = tree.lookup(0xA0008059);
	EXPECT_EQ(nullptr, slot);
	slot = tree.lookup(0xA0008069);
	EXPECT_EQ(nullptr, slot);

	const unsigned N = 257;
	for (unsigned i = 0; N > i; i++) {
		tree.ensure(+1);
		slot = tree.assign(i);
		EXPECT_NE(nullptr, slot);
		tree.setval(slot, i);
	}

	pair = tree.iterate( &iter, 1);
	for (unsigned i = 0; N > i; i++){
		EXPECT_EQ(i, pair.x);
		EXPECT_NE(nullptr, pair.slot);
		EXPECT_EQ(i, tree.getval(pair.slot));
		pair = tree.iterate( &iter, 0);
	}
	EXPECT_EQ(0, pair.x);
	EXPECT_EQ(nullptr, pair.slot);
}

TEST(imap, iterate_shuffle) {
	const unsigned N = 10000000;
	uint32_t *array;
	imap_tree_t tree(1);
	uint32_t *slot;
	imap_iter_t iter;
	imap_pair_t pair;

	array = (uint32_t *)malloc(N * sizeof(uint32_t));
	EXPECT_NE(nullptr, array);

	for (unsigned i = 0; N > i; i++) {
		array[i] = i;
	}

	for (unsigned i = 0; N > i; i++) {
		tree.ensure(+1);
		slot = tree.assign(array[i]);
		EXPECT_NE(nullptr, slot);
		tree.setval(slot, array[i]);
	}

	pair = tree.iterate( &iter, 1);
	for (unsigned i = 0; N > i; i++) {
		EXPECT_EQ(array[i], pair.x);
		EXPECT_NE(nullptr, pair.slot);
		EXPECT_EQ(array[i], tree.getval(pair.slot));
		pair = tree.iterate( &iter, 0);
	}

	EXPECT_EQ(0, pair.x);
	EXPECT_EQ(nullptr, pair.slot);

	for (unsigned i = 0; N > i; i++) {
		uint32_t r = rng() % N;
		uint32_t t = array[i];
		array[i] = array[r];
		array[r] = t;
	}

	for (unsigned i = 0; N / 2 > i; i++) {
		tree.remove(array[i]);
	}

	qsort(array + N / 2, N / 2, sizeof array[0], u32cmp);

	pair = tree.iterate( &iter, 1);
	for (unsigned i = N / 2; N > i; i++) {
		EXPECT_EQ(array[i], pair.x);
		EXPECT_NE(nullptr, pair.slot);
		EXPECT_EQ(array[i], tree.getval(pair.slot));
		pair = tree.iterate( &iter, 0);
	}

	EXPECT_EQ(0, pair.x);
	EXPECT_EQ(nullptr, pair.slot);
	free(array);
}

TEST(imap, locate) {
	imap_tree_t tree(0);
	uint32_t *slot;
	imap_iter_t iter;
	imap_pair_t pair;

	tree.ensure(+1);
	pair = tree.locate( &iter, 0);
	EXPECT_EQ(0, pair.x);
	EXPECT_EQ(nullptr, pair.slot);

	tree.ensure(+1);

	slot = tree.assign(1200);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 1100);
	pair = tree.locate( &iter, 1100);
	EXPECT_EQ(1200, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(1100, tree.getval(slot));
	pair = tree.iterate(&iter, 0);
	EXPECT_EQ(0, pair.x);
	EXPECT_EQ(nullptr, pair.slot);

	tree.ensure(+1);

	slot = tree.assign(1200);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 1100);
	tree.ensure(+1);

	slot = tree.assign(1100);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 1000);
	pair = tree.locate( &iter, 1300);
	EXPECT_EQ(0, pair.x);
	EXPECT_EQ(nullptr, pair.slot);

	tree.ensure(+5);

	slot = tree.assign(0xA0000056);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 0x56);
	pair = tree.locate( &iter, 0xA00000560);
	EXPECT_EQ(0, pair.x);
	EXPECT_EQ(nullptr, pair.slot);

	tree.ensure(+5);

	slot = tree.assign(0xA0000056);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 0x56);
	slot = tree.assign(0xA0000057);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 0x57);
	slot = tree.assign(0xA0008009);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 0x8009);
	slot = tree.assign(0xA0008059);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 0x8059);
	slot = tree.assign(0xA0008069);
	EXPECT_NE(nullptr, slot);
	tree.setval(slot, 0x8069);
	//
	pair = tree.locate(&iter, 1100);
	EXPECT_EQ(0xA0000056, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x56, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_EQ(0xA0000057, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x57, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_EQ(0xA0008009, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x8009, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_EQ(0xA0008059, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x8059, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_EQ(0xA0008069, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x8069, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_NE(0, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	//
	//
	pair = tree.locate( &iter, 0xA0000057);
	EXPECT_EQ(0xA0000057, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x57, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_EQ(0xA0008009, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x8009, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_EQ(0xA0008059, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x8059, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_EQ(0xA0008069, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x8069, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_EQ(0, pair.x);
	EXPECT_EQ(nullptr, pair.slot);
	//
	//
	pair = tree.locate( &iter, 0xA0007000);
	EXPECT_EQ(0xA0008009, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x8009, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_EQ(0xA0008059, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x8059, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_EQ(0xA0008069, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x8069, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_EQ(0, pair.x);
	EXPECT_EQ(nullptr, pair.slot);
	//
	//
	pair = tree.locate( &iter, 0x90008059);
	EXPECT_EQ(0xA0000056, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x56, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_EQ(0xA0000057, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x57, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_EQ(0xA0008009, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x8009, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_EQ(0xA0008059, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x8059, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_EQ(0xA0008069, pair.x);
	EXPECT_NE(nullptr, pair.slot);
	EXPECT_EQ(0x8069, tree.getval(pair.slot));
	pair = tree.iterate( &iter, 0);
	EXPECT_EQ(0, pair.x);
	EXPECT_EQ(nullptr, pair.slot);
	//
	//
	pair = tree.locate( &iter, 0xB0008059);
	EXPECT_EQ(0, pair.x);
	EXPECT_EQ(nullptr, pair.slot);
	//
	slot = tree.lookup(0xA0000056);
	EXPECT_NE(nullptr, slot);
	EXPECT_EQ(0x56, tree.getval(slot));
	tree.delval(slot);
	slot = tree.lookup(0xA0000057);
	EXPECT_NE(nullptr, slot);
	EXPECT_EQ(0x57, tree.getval(slot));
	tree.delval(slot);
	slot = tree.lookup(0xA0008009);
	EXPECT_NE(nullptr, slot);
	EXPECT_EQ(0x8009, tree.getval(slot));
	tree.delval(slot);
	slot = tree.lookup(0xA0008059);
	EXPECT_NE(nullptr, slot);
	EXPECT_EQ(0x8059, tree.getval(slot));
	tree.delval(slot);
	slot = tree.lookup(0xA0008069);
	EXPECT_NE(nullptr, slot);
	EXPECT_EQ(0x8069, tree.getval(slot));
	tree.delval(slot);
	slot = tree.lookup(0xA0000056);
	EXPECT_EQ(0, slot);
	slot = tree.lookup(0xA0000057);
	EXPECT_EQ(0, slot);
	slot = tree.lookup(0xA0008009);
	EXPECT_EQ(0, slot);
	slot = tree.lookup(0xA0008059);
	EXPECT_EQ(0, slot);
	slot = tree.lookup(0xA0008069);
	EXPECT_EQ(0, slot);
}

TEST(imap, locate_random) {
	const unsigned M = 1000000;
	uint32_t *array;
	imap_tree_t tree(1);
	uint32_t *slot;
	imap_iter_t iter;
	imap_pair_t pair;
	uint32_t r, *p;

	array = (uint32_t *)malloc(N * sizeof(uint32_t));
	EXPECT_NE(nullptr, array);

	for (unsigned i = 0; N > i; i++) {
		array[i] = 0x1000000 | (rng() & 0x3ffffff);
	}

	for (unsigned i = 0; N > i; i++) {
		tree.ensure(+1);
		slot = tree.assign(array[i]);
		EXPECT_NE(nullptr, slot);
		tree.setval(slot, array[i]);
	}

	qsort(array, N, sizeof array[0], u32cmp);

	for (unsigned i = 0; M > i; i++) {
		r = rng() & 0x3ffffff;
		pair = tree.locate( &iter, r);
		p = test_bsearch(r, array, N);
		if (array + N > p) {
			EXPECT_EQ(*p, pair.x);
			EXPECT_NE(nullptr, pair.slot);
			EXPECT_EQ(*p, tree.getval(pair.slot));
			pair = tree.iterate( &iter, 0);
			p++;
			if (array + N > p) {
				EXPECT_EQ(*p, pair.x);
				EXPECT_NE(nullptr, pair.slot);
				EXPECT_EQ(*p, tree.getval(pair.slot));
			} else {
				EXPECT_EQ(0, pair.x);
				EXPECT_EQ(nullptr, pair.slot);
			}
		} else {
			EXPECT_EQ(0, pair.x);
			EXPECT_EQ(nullptr, pair.slot);
		}
	}

	for (unsigned i = 0; M > i; i++) {
		r = rng() % N;
		pair = tree.locate( &iter, array[r]);
		p = test_bsearch(array[r], array, N);
		if (array + N > p) {
			EXPECT_EQ(*p, pair.x);
			EXPECT_NE(nullptr, pair.slot);
			EXPECT_EQ(*p, tree.getval(pair.slot));
			pair = tree.iterate( &iter, 0);
			p++;
			if (array + N > p){
				EXPECT_EQ(*p, pair.x);
				EXPECT_NE(nullptr, pair.slot);
				EXPECT_EQ(*p, tree.getval(pair.slot));
			} else {
				EXPECT_EQ(0, pair.x);
				EXPECT_EQ(nullptr, pair.slot);
			}
		} else {
			EXPECT_EQ(0, pair.x);
			EXPECT_EQ(nullptr, pair.slot);
		}
	}

	free(array);
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	rng_seed(time(nullptr));
    return RUN_ALL_TESTS();
}
