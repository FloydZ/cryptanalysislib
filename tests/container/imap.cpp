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
//	EXPECT_EQ(0 != tree);
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
//	EXPECT_EQ(0 != tree);
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
//	EXPECT_EQ(0 != tree);
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
		uint32_t r = fastrandombytes_uint64() % N;
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

//static void imap_remove_shuffle_dotest(uint64_t seed) {
//	const unsigned N = 10000000;
//	uint32_t *array;
//	imap_node_t *tree = 0;
//	imap_slot_t *slot;
//
//	tlib_printf("seed=%llu ", (unsigned long long)seed);
//	test_srand(seed);
//
//	array = (uint32_t *)malloc(N * sizeof(imap_u32_t));
//	EXPECT_EQ(0 != array);
//
//	for (unsigned i = 0; N > i; i++)
//		array[i] = i;
//
//	for (unsigned i = 0; N > i; i++)
//	{
//		tree = tree.ensure(+1);
//		EXPECT_EQ(0 != tree);
//		slot = tree.assign(array[i]);
//		EXPECT_EQ(nullptr, slot);
//		tree.setval(slot, i);
//	}
//	for (unsigned i = 0; N > i; i++)
//	{
//		slot = tree.lookup(array[i]);
//		EXPECT_EQ(nullptr, slot);
//		EXPECT_EQ(i == tree.getval(slot));
//	}
//
//	for (unsigned i = 0; N > i; i++)
//	{
//		uint32_t r = test_rand() % N;
//		uint32_t t = array[i];
//		array[i] = array[r];
//		array[r] = t;
//	}
//
//	for (unsigned i = 0; N / 2 > i; i++)
//		tree.remove(array[i]);
//	for (unsigned i = 0; N / 2 > i; i++)
//	{
//		slot = tree.lookup(array[i]);
//		EXPECT_EQ(0 == slot);
//	}
//	for (unsigned i = N / 2; N > i; i++)
//	{
//		slot = tree.lookup(array[i]);
//		EXPECT_EQ(nullptr, slot);
//		EXPECT_EQ(array[i] == tree.getval(slot));
//	}
//	for (unsigned i = N / 2; N > i; i++)
//		tree.remove(array[i]);
//	for (unsigned i = N / 2; N > i; i++)
//	{
//		slot = tree.lookup(array[i]);
//		EXPECT_EQ(0 == slot);
//	}
//
//	EXPECT_EQ(0 == tree->vec32[0]);
//
//	imap_free(tree);
//
//	free(array);
//}
//
//static void imap_iterate_test(void) {
//	imap_node_t *tree;
//	imap_slot_t *slot;
//	imap_iter_t iter;
//	imap_pair_t pair;
//
//	tree = 0;
//	tree = tree.ensure(+1);
//	EXPECT_EQ(0 != tree);
//	pair = imap_iterate(tree, &iter, 1);
//	EXPECT_EQ(0 == pair.x && 0 == pair.slot);
//	imap_free(tree);
//
//	tree = 0;
//	tree = tree.ensure(+5);
//	EXPECT_EQ(0 != tree);
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
//	//
//	pair = imap_iterate(tree, &iter, 1);
//	EXPECT_EQ(0xA0000056 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x56 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0xA0000057 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x57 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0xA0008009 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x8009 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0xA0008059 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x8059 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0xA0008069 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x8069 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x && 0 == pair.slot);
//	//
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
//
//	const unsigned N = 257;
//	tree = 0;
//	for (unsigned i = 0; N > i; i++)
//	{
//		tree = tree.ensure(+1);
//		EXPECT_EQ(0 != tree);
//		slot = tree.assign(i);
//		EXPECT_EQ(nullptr, slot);
//		tree.setval(slot, i);
//	}
//	pair = imap_iterate(tree, &iter, 1);
//	for (unsigned i = 0; N > i; i++)
//	{
//		EXPECT_EQ(i == pair.x);
//		EXPECT_EQ(0 != pair.slot);
//		EXPECT_EQ(i == tree.getval(pair.slot));
//		pair = imap_iterate(tree, &iter, 0);
//	}
//	EXPECT_EQ(0 == pair.x && 0 == pair.slot);
//	imap_free(tree);
//}
//
//static void imap_iterate_shuffle_dotest(uint64_t seed) {
//	const unsigned N = 10000000;
//	uint32_t *array;
//	imap_node_t *tree = 0;
//	imap_slot_t *slot;
//	imap_iter_t iter;
//	imap_pair_t pair;
//
//	tlib_printf("seed=%llu ", (unsigned long long)seed);
//	test_srand(seed);
//
//	array = (uint32_t *)malloc(N * sizeof(imap_u32_t));
//	EXPECT_EQ(0 != array);
//
//	for (unsigned i = 0; N > i; i++)
//		array[i] = i;
//
//	for (unsigned i = 0; N > i; i++)
//	{
//		tree = tree.ensure(+1);
//		EXPECT_EQ(0 != tree);
//		slot = tree.assign(array[i]);
//		EXPECT_EQ(nullptr, slot);
//		tree.setval(slot, array[i]);
//	}
//	pair = imap_iterate(tree, &iter, 1);
//	for (unsigned i = 0; N > i; i++)
//	{
//		EXPECT_EQ(array[i] == pair.x);
//		EXPECT_EQ(0 != pair.slot);
//		EXPECT_EQ(array[i] == tree.getval(pair.slot));
//		pair = imap_iterate(tree, &iter, 0);
//	}
//	EXPECT_EQ(0 == pair.x && 0 == pair.slot);
//
//	for (unsigned i = 0; N > i; i++)
//	{
//		uint32_t r = test_rand() % N;
//		uint32_t t = array[i];
//		array[i] = array[r];
//		array[r] = t;
//	}
//
//	for (unsigned i = 0; N / 2 > i; i++)
//		tree.remove(array[i]);
//
//	qsort(array + N / 2, N / 2, sizeof array[0], u32cmp);
//
//	pair = imap_iterate(tree, &iter, 1);
//	for (unsigned i = N / 2; N > i; i++)
//	{
//		EXPECT_EQ(array[i] == pair.x);
//		EXPECT_EQ(0 != pair.slot);
//		EXPECT_EQ(array[i] == tree.getval(pair.slot));
//		pair = imap_iterate(tree, &iter, 0);
//	}
//	EXPECT_EQ(0 == pair.x && 0 == pair.slot);
//
//	imap_free(tree);
//
//	free(array);
//}
//
//static void imap_locate_test(void){
//	imap_node_t *tree;
//	imap_slot_t *slot;
//	imap_iter_t iter;
//	imap_pair_t pair;
//
//	tree = 0;
//	tree = tree.ensure(+1);
//	EXPECT_EQ(0 != tree);
//	pair = imap_locate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x && 0 == pair.slot);
//	imap_free(tree);
//
//	tree = 0;
//	tree = tree.ensure(+1);
//	EXPECT_EQ(0 != tree);
//	slot = tree.assign(1200);
//	EXPECT_EQ(nullptr, slot);
//	tree.setval(slot, 1100);
//	pair = imap_locate(tree, &iter, 1100);
//	EXPECT_EQ(1200 == pair.x && 0 != pair.slot && 1100 == tree.getval(slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x && 0 == pair.slot);
//	imap_free(tree);
//
//	tree = 0;
//	tree = tree.ensure(+1);
//	EXPECT_EQ(0 != tree);
//	slot = tree.assign(1200);
//	EXPECT_EQ(nullptr, slot);
//	tree.setval(slot, 1100);
//	tree = tree.ensure(+1);
//	EXPECT_EQ(0 != tree);
//	slot = tree.assign(1100);
//	EXPECT_EQ(nullptr, slot);
//	tree.setval(slot, 1000);
//	pair = imap_locate(tree, &iter, 1300);
//	EXPECT_EQ(0 == pair.x && 0 == pair.slot);
//	imap_free(tree);
//
//	tree = 0;
//	tree = tree.ensure(+5);
//	EXPECT_EQ(0 != tree);
//	slot = tree.assign(0xA0000056);
//	EXPECT_EQ(nullptr, slot);
//	tree.setval(slot, 0x56);
//	pair = imap_locate(tree, &iter, 0xA00000560);
//	EXPECT_EQ(0 == pair.x && 0 == pair.slot);
//	imap_free(tree);
//
//	tree = 0;
//	tree = tree.ensure(+5);
//	EXPECT_EQ(0 != tree);
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
//	//
//	pair = imap_locate(tree, &iter, 1100);
//	EXPECT_EQ(0xA0000056 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x56 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0xA0000057 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x57 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0xA0008009 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x8009 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0xA0008059 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x8059 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0xA0008069 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x8069 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x && 0 == pair.slot);
//	//
//	//
//	pair = imap_locate(tree, &iter, 0xA0000057);
//	EXPECT_EQ(0xA0000057 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x57 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0xA0008009 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x8009 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0xA0008059 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x8059 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0xA0008069 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x8069 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x && 0 == pair.slot);
//	//
//	//
//	pair = imap_locate(tree, &iter, 0xA0007000);
//	EXPECT_EQ(0xA0008009 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x8009 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0xA0008059 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x8059 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0xA0008069 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x8069 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x && 0 == pair.slot);
//	//
//	//
//	pair = imap_locate(tree, &iter, 0x90008059);
//	EXPECT_EQ(0xA0000056 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x56 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0xA0000057 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x57 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0xA0008009 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x8009 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0xA0008059 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x8059 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0xA0008069 == pair.x && 0 != pair.slot);
//	EXPECT_EQ(0x8069 == tree.getval(pair.slot));
//	pair = imap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x && 0 == pair.slot);
//	//
//	//
//	pair = imap_locate(tree, &iter, 0xB0008059);
//	EXPECT_EQ(0 == pair.x && 0 == pair.slot);
//	//
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
//}
//
//static void imap_locate_random_dotest(uint64_t seed) {
//	const unsigned N = 10000000;
//	const unsigned M = 1000000;
//	uint32_t *array;
//	imap_node_t *tree = 0;
//	imap_slot_t *slot;
//	imap_iter_t iter;
//	imap_pair_t pair;
//	uint32_t r, *p;
//
//	tlib_printf("seed=%llu ", (unsigned long long)seed);
//	test_srand(seed);
//
//	array = (uint32_t *)malloc(N * sizeof(imap_u32_t));
//	EXPECT_EQ(0 != array);
//
//	for (unsigned i = 0; N > i; i++)
//		array[i] = 0x1000000 | (test_rand() & 0x3ffffff);
//
//	for (unsigned i = 0; N > i; i++)
//	{
//		tree = tree.ensure(+1);
//		EXPECT_EQ(0 != tree);
//		slot = tree.assign(array[i]);
//		EXPECT_EQ(nullptr, slot);
//		tree.setval(slot, array[i]);
//	}
//
//	qsort(array, N, sizeof array[0], u32cmp);
//
//	for (unsigned i = 0; M > i; i++)
//	{
//		r = test_rand() & 0x3ffffff;
//		pair = imap_locate(tree, &iter, r);
//		p = test_bsearch(r, array, N);
//		if (array + N > p)
//		{
//			EXPECT_EQ(*p == pair.x);
//			EXPECT_EQ(0 != pair.slot);
//			EXPECT_EQ(*p == tree.getval(pair.slot));
//			pair = imap_iterate(tree, &iter, 0);
//			p++;
//			if (array + N > p)
//			{
//				EXPECT_EQ(*p == pair.x);
//				EXPECT_EQ(0 != pair.slot);
//				EXPECT_EQ(*p == tree.getval(pair.slot));
//			}
//			else
//			{
//				EXPECT_EQ(0 == pair.x);
//				EXPECT_EQ(0 == pair.slot);
//			}
//		}
//		else
//		{
//			EXPECT_EQ(0 == pair.x);
//			EXPECT_EQ(0 == pair.slot);
//		}
//	}
//
//	for (unsigned i = 0; M > i; i++)
//	{
//		r = test_rand() % N;
//		pair = imap_locate(tree, &iter, array[r]);
//		p = test_bsearch(array[r], array, N);
//		if (array + N > p)
//		{
//			EXPECT_EQ(*p == pair.x);
//			EXPECT_EQ(0 != pair.slot);
//			EXPECT_EQ(*p == tree.getval(pair.slot));
//			pair = imap_iterate(tree, &iter, 0);
//			p++;
//			if (array + N > p)
//			{
//				EXPECT_EQ(*p == pair.x);
//				EXPECT_EQ(0 != pair.slot);
//				EXPECT_EQ(*p == tree.getval(pair.slot));
//			}
//			else
//			{
//				EXPECT_EQ(0 == pair.x);
//				EXPECT_EQ(0 == pair.slot);
//			}
//		}
//		else
//		{
//			EXPECT_EQ(0 == pair.x);
//			EXPECT_EQ(0 == pair.slot);
//		}
//	}
//
//	imap_free(tree);
//
//	free(array);
//}
//
//static void imap_dump_test(void) {
//	imap_node_t *tree;
//	imap_slot_t *slot;
//	char *dump;
//
//	tree = 0;
//	tree = tree.ensure(+1);
//	EXPECT_EQ(0 != tree);
//	dump = 0;
//	imap_dump(tree, test_concat_sprintf, &dump);
//	EXPECT_EQ(0 == dump);
//	free(dump);
//	imap_free(tree);
//
//	tree = 0;
//	tree = tree.ensure(+5);
//	EXPECT_EQ(0 != tree);
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
//	//
//	dump = 0;
//	imap_dump(tree, test_concat_sprintf, &dump);
//	EXPECT_EQ(0 == strcmp(dump, ""
//	                         "00000080: 00000000a0000000/3 0->*40 8->*100\n"
//	                         "00000040: 00000000a0000050/0 6->56 7->57\n"
//	                         "00000100: 00000000a0008000/1 0->*c0 5->*140 6->*180\n"
//	                         "000000c0: 00000000a0008000/0 9->8009\n"
//	                         "00000140: 00000000a0008050/0 9->8059\n"
//	                         "00000180: 00000000a0008060/0 9->8069\n"
//	                         ""));
//	free(dump);
//	//
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
//}


//static void iset_assign_test(void) {
//	iset_node_t *tree;
//
//	tree = 0;
//	tree = iset_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	EXPECT_EQ(!iset_lookup(tree, 0));
//	iset_assign(tree, 0);
//	EXPECT_EQ(iset_lookup(tree, 0));
//	EXPECT_EQ(!iset_lookup(tree, 1));
//	tree = iset_ensure(tree, +1);
//	iset_assign(tree, 1);
//	EXPECT_EQ(iset_lookup(tree, 1));
//	EXPECT_EQ(!iset_lookup(tree, 1000));
//	tree = iset_ensure(tree, +1);
//	iset_assign(tree, 1000);
//	EXPECT_EQ(iset_lookup(tree, 1000));
//	iset_free(tree);
//}
//
//static void iset_remove_test() {
//	iset_node_t *tree;
//
//	tree = 0;
//	tree = iset_ensure(tree, +3);
//	EXPECT_EQ(0 != tree);
//	EXPECT_EQ(!iset_lookup(tree, 0));
//	EXPECT_EQ(!iset_lookup(tree, 1));
//	EXPECT_EQ(!iset_lookup(tree, 2));
//	EXPECT_EQ(!iset_lookup(tree, 1000));
//	EXPECT_EQ(!iset_lookup(tree, 2000));
//	iset_assign(tree, 0);
//	iset_assign(tree, 1);
//	iset_assign(tree, 1000);
//	EXPECT_EQ(iset_lookup(tree, 0));
//	EXPECT_EQ(iset_lookup(tree, 1));
//	EXPECT_EQ(!iset_lookup(tree, 2));
//	EXPECT_EQ(iset_lookup(tree, 1000));
//	EXPECT_EQ(!iset_lookup(tree, 2000));
//	iset_remove(tree, 2);
//	EXPECT_EQ(iset_lookup(tree, 0));
//	EXPECT_EQ(iset_lookup(tree, 1));
//	EXPECT_EQ(!iset_lookup(tree, 2));
//	EXPECT_EQ(iset_lookup(tree, 1000));
//	EXPECT_EQ(!iset_lookup(tree, 2000));
//	iset_remove(tree, 2000);
//	EXPECT_EQ(iset_lookup(tree, 0));
//	EXPECT_EQ(iset_lookup(tree, 1));
//	EXPECT_EQ(!iset_lookup(tree, 2));
//	EXPECT_EQ(iset_lookup(tree, 1000));
//	EXPECT_EQ(!iset_lookup(tree, 2000));
//	iset_remove(tree, 0);
//	EXPECT_EQ(!iset_lookup(tree, 0));
//	EXPECT_EQ(iset_lookup(tree, 1));
//	EXPECT_EQ(!iset_lookup(tree, 2));
//	EXPECT_EQ(iset_lookup(tree, 1000));
//	EXPECT_EQ(!iset_lookup(tree, 2000));
//	iset_remove(tree, 1);
//	EXPECT_EQ(!iset_lookup(tree, 0));
//	EXPECT_EQ(!iset_lookup(tree, 1));
//	EXPECT_EQ(!iset_lookup(tree, 2));
//	EXPECT_EQ(iset_lookup(tree, 1000));
//	EXPECT_EQ(!iset_lookup(tree, 2000));
//	iset_remove(tree, 1000);
//	EXPECT_EQ(!iset_lookup(tree, 0));
//	EXPECT_EQ(!iset_lookup(tree, 1));
//	EXPECT_EQ(!iset_lookup(tree, 2));
//	EXPECT_EQ(!iset_lookup(tree, 1000));
//	EXPECT_EQ(!iset_lookup(tree, 2000));
//	iset_free(tree);
//}
//
//static void iset_locate_test() {
//	iset_node_t *tree;
//	iset_iter_t iter;
//	iset_pair_t pair;
//
//	tree = 0;
//	tree = iset_ensure(tree, +5);
//	EXPECT_EQ(0 != tree);
//	pair = iset_locate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x && !pair.elemof);
//	pair = iset_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x && !pair.elemof);
//	iset_free(tree);
//
//	tree = 0;
//	tree = iset_ensure(tree, +5);
//	EXPECT_EQ(0 != tree);
//	iset_assign(tree, 0);
//	iset_assign(tree, 1);
//	iset_assign(tree, 10);
//	iset_assign(tree, 1000);
//	iset_assign(tree, 1002);
//	pair = iset_locate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x && pair.elemof);
//	pair = iset_iterate(tree, &iter, 0);
//	EXPECT_EQ(1 == pair.x && pair.elemof);
//	pair = iset_locate(tree, &iter, 1);
//	EXPECT_EQ(1 == pair.x && pair.elemof);
//	pair = iset_iterate(tree, &iter, 0);
//	EXPECT_EQ(10 == pair.x && pair.elemof);
//	pair = iset_locate(tree, &iter, 2);
//	EXPECT_EQ(10 == pair.x && pair.elemof);
//	pair = iset_iterate(tree, &iter, 0);
//	EXPECT_EQ(1000 == pair.x && pair.elemof);
//	pair = iset_locate(tree, &iter, 999);
//	EXPECT_EQ(1000 == pair.x && pair.elemof);
//	pair = iset_iterate(tree, &iter, 0);
//	EXPECT_EQ(1002 == pair.x && pair.elemof);
//	pair = iset_locate(tree, &iter, 1000);
//	EXPECT_EQ(1000 == pair.x && pair.elemof);
//	pair = iset_iterate(tree, &iter, 0);
//	EXPECT_EQ(1002 == pair.x && pair.elemof);
//	pair = iset_locate(tree, &iter, 1001);
//	EXPECT_EQ(1002 == pair.x && pair.elemof);
//	pair = iset_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x && !pair.elemof);
//	pair = iset_locate(tree, &iter, 1002);
//	EXPECT_EQ(1002 == pair.x && pair.elemof);
//	pair = iset_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x && !pair.elemof);
//	pair = iset_locate(tree, &iter, 1003);
//	EXPECT_EQ(0 == pair.x && !pair.elemof);
//	iset_free(tree);
//}
//
//static void iset_iterate_test() {
//	iset_node_t *tree;
//	iset_iter_t iter;
//	iset_pair_t pair;
//
//	tree = 0;
//	tree = iset_ensure(tree, +5);
//	EXPECT_EQ(0 != tree);
//	pair = iset_iterate(tree, &iter, 1);
//	EXPECT_EQ(0 == pair.x && !pair.elemof);
//	pair = iset_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x && !pair.elemof);
//	iset_free(tree);
//
//	tree = 0;
//	tree = iset_ensure(tree, +5);
//	EXPECT_EQ(0 != tree);
//	iset_assign(tree, 0);
//	iset_assign(tree, 1);
//	iset_assign(tree, 10);
//	iset_assign(tree, 1000);
//	iset_assign(tree, 1002);
//	pair = iset_iterate(tree, &iter, 1);
//	EXPECT_EQ(0 == pair.x && pair.elemof);
//	pair = iset_iterate(tree, &iter, 0);
//	EXPECT_EQ(1 == pair.x && pair.elemof);
//	pair = iset_iterate(tree, &iter, 0);
//	EXPECT_EQ(10 == pair.x && pair.elemof);
//	pair = iset_iterate(tree, &iter, 0);
//	EXPECT_EQ(1000 == pair.x && pair.elemof);
//	pair = iset_iterate(tree, &iter, 0);
//	EXPECT_EQ(1002 == pair.x && pair.elemof);
//	pair = iset_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x && !pair.elemof);
//	pair = iset_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x && !pair.elemof);
//	iset_free(tree);
//}
//
//static void ivmap_insert_test() {
//	ivmap_node_t *tree;
//	ivmap_u64_t *y;
//
//	tree = 0;
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	y = ivmap_insert(tree, 1100, 1200);
//	EXPECT_EQ(0 != y);
//	*y = 101100;
//	y = ivmap_lookup(tree, 1099);
//	EXPECT_EQ(0 == y);
//	y = ivmap_lookup(tree, 1100);
//	EXPECT_EQ(0 != y && 101100 == *y);
//	y = ivmap_lookup(tree, 1150);
//	EXPECT_EQ(0 != y && 101100 == *y);
//	y = ivmap_lookup(tree, 1199);
//	EXPECT_EQ(0 != y && 101100 == *y);
//	y = ivmap_lookup(tree, 1200);
//	EXPECT_EQ(0 == y);
//	y = ivmap_insert(tree, 1100, 1200);
//	EXPECT_EQ(0 == y);
//	y = ivmap_insert(tree, 1199, 1200);
//	EXPECT_EQ(0 == y);
//	y = ivmap_insert(tree, 1100, 1101);
//	EXPECT_EQ(0 == y);
//	y = ivmap_insert(tree, 1000, 1101);
//	EXPECT_EQ(0 == y);
//	y = ivmap_insert(tree, 1000, 1200);
//	EXPECT_EQ(0 == y);
//	y = ivmap_insert(tree, 1000, 1300);
//	EXPECT_EQ(0 == y);
//	y = ivmap_insert(tree, 1199, 1300);
//	EXPECT_EQ(0 == y);
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	y = ivmap_insert(tree, 1000, 1100);
//	EXPECT_EQ(0 != y);
//	*y = 101000;
//	y = ivmap_lookup(tree, 999);
//	EXPECT_EQ(0 == y);
//	y = ivmap_lookup(tree, 1000);
//	EXPECT_EQ(0 != y && 101000 == *y);
//	y = ivmap_lookup(tree, 1050);
//	EXPECT_EQ(0 != y && 101000 == *y);
//	y = ivmap_lookup(tree, 1099);
//	EXPECT_EQ(0 != y && 101000 == *y);
//	y = ivmap_lookup(tree, 1100);
//	EXPECT_EQ(0 != y && 101100 == *y);
//	y = ivmap_lookup(tree, 1200);
//	EXPECT_EQ(0 == y);
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	y = ivmap_insert(tree, 1200, 1300);
//	EXPECT_EQ(0 != y);
//	*y = 101200;
//	y = ivmap_lookup(tree, 999);
//	EXPECT_EQ(0 == y);
//	y = ivmap_lookup(tree, 1000);
//	EXPECT_EQ(0 != y && 101000 == *y);
//	y = ivmap_lookup(tree, 1100);
//	EXPECT_EQ(0 != y && 101100 == *y);
//	y = ivmap_lookup(tree, 1200);
//	EXPECT_EQ(0 != y && 101200 == *y);
//	y = ivmap_lookup(tree, 1299);
//	EXPECT_EQ(0 != y && 101200 == *y);
//	y = ivmap_lookup(tree, 1300);
//	EXPECT_EQ(0 == y);
//	y = ivmap_insert(tree, 1050, 1150);
//	EXPECT_EQ(0 == y);
//	y = ivmap_insert(tree, 1150, 1250);
//	EXPECT_EQ(0 == y);
//	ivmap_free(tree);
//
//	tree = 0;
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	y = ivmap_insert(tree, 1100, 1200);
//	EXPECT_EQ(0 != y);
//	*y = 101100;
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	y = ivmap_insert(tree, 1300, 1400);
//	EXPECT_EQ(0 != y);
//	*y = 101300;
//	y = ivmap_insert(tree, 1000, 1500);
//	EXPECT_EQ(0 == y);
//	y = ivmap_insert(tree, 1000, 1350);
//	EXPECT_EQ(0 == y);
//	y = ivmap_insert(tree, 1350, 1500);
//	EXPECT_EQ(0 == y);
//	y = ivmap_lookup(tree, 1099);
//	EXPECT_EQ(0 == y);
//	y = ivmap_lookup(tree, 1100);
//	EXPECT_EQ(0 != y && 101100 == *y);
//	y = ivmap_lookup(tree, 1199);
//	EXPECT_EQ(0 != y && 101100 == *y);
//	y = ivmap_lookup(tree, 1200);
//	EXPECT_EQ(0 == y);
//	y = ivmap_lookup(tree, 1299);
//	EXPECT_EQ(0 == y);
//	y = ivmap_lookup(tree, 1300);
//	EXPECT_EQ(0 != y && 101300 == *y);
//	y = ivmap_lookup(tree, 1399);
//	EXPECT_EQ(0 != y && 101300 == *y);
//	y = ivmap_lookup(tree, 1400);
//	EXPECT_EQ(0 == y);
//	ivmap_free(tree);
//}
//
//static void ivmap_remove_test() {
//	ivmap_node_t *tree;
//	ivmap_u64_t *y;
//
//	tree = 0;
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	y = ivmap_insert(tree, 1100, 1200);
//	EXPECT_EQ(0 != y);
//	*y = 101100;
//	y = ivmap_lookup(tree, 1100);
//	EXPECT_EQ(0 != y && 101100 == *y);
//	ivmap_remove(tree, 1100);
//	y = ivmap_lookup(tree, 1100);
//	EXPECT_EQ(0 == y);
//	ivmap_free(tree);
//
//	tree = 0;
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	y = ivmap_insert(tree, 1100, 1200);
//	EXPECT_EQ(0 != y);
//	*y = 101100;
//	y = ivmap_insert(tree, 1200, 1300);
//	EXPECT_EQ(0 != y);
//	*y = 101200;
//	y = ivmap_lookup(tree, 1100);
//	EXPECT_EQ(0 != y && 101100 == *y);
//	y = ivmap_lookup(tree, 1200);
//	EXPECT_EQ(0 != y && 101200 == *y);
//	ivmap_remove(tree, 1100);
//	y = ivmap_lookup(tree, 1100);
//	EXPECT_EQ(0 == y);
//	y = ivmap_lookup(tree, 1200);
//	EXPECT_EQ(0 != y && 101200 == *y);
//	ivmap_remove(tree, 1200);
//	y = ivmap_lookup(tree, 1200);
//	EXPECT_EQ(0 == y);
//	ivmap_free(tree);
//
//	tree = 0;
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	y = ivmap_insert(tree, 1100, 1200);
//	EXPECT_EQ(0 != y);
//	*y = 101100;
//	y = ivmap_insert(tree, 1300, 1400);
//	EXPECT_EQ(0 != y);
//	*y = 101300;
//	y = ivmap_lookup(tree, 1100);
//	EXPECT_EQ(0 != y && 101100 == *y);
//	y = ivmap_lookup(tree, 1300);
//	EXPECT_EQ(0 != y && 101300 == *y);
//	ivmap_remove(tree, 1100);
//	y = ivmap_lookup(tree, 1100);
//	EXPECT_EQ(0 == y);
//	y = ivmap_lookup(tree, 1300);
//	EXPECT_EQ(0 != y && 101300 == *y);
//	ivmap_remove(tree, 1300);
//	y = ivmap_lookup(tree, 1300);
//	EXPECT_EQ(0 == y);
//	ivmap_free(tree);
//}
//
//static void ivmap_locate_test() {
//	ivmap_node_t *tree;
//	ivmap_u64_t *y;
//	ivmap_iter_t iter;
//	ivmap_pair_t pair;
//
//	tree = 0;
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	pair = ivmap_locate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//	ivmap_free(tree);
//
//	tree = 0;
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	y = ivmap_insert(tree, 1100, 1200);
//	EXPECT_EQ(0 != y);
//	*y = 101100;
//	pair = ivmap_locate(tree, &iter, 0);
//	EXPECT_EQ(1100 == pair.x0 && 1200 == pair.x1 && 0 != pair.y && 101100 == *pair.y);
//	pair = ivmap_locate(tree, &iter, 1099);
//	EXPECT_EQ(1100 == pair.x0 && 1200 == pair.x1 && 0 != pair.y && 101100 == *pair.y);
//	pair = ivmap_locate(tree, &iter, 1100);
//	EXPECT_EQ(1100 == pair.x0 && 1200 == pair.x1 && 0 != pair.y && 101100 == *pair.y);
//	pair = ivmap_locate(tree, &iter, 1199);
//	EXPECT_EQ(1100 == pair.x0 && 1200 == pair.x1 && 0 != pair.y && 101100 == *pair.y);
//	pair = ivmap_locate(tree, &iter, 1200);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//	ivmap_free(tree);
//
//	tree = 0;
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	y = ivmap_insert(tree, 1100, 1200);
//	EXPECT_EQ(0 != y);
//	*y = 101100;
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	y = ivmap_insert(tree, 1200, 1300);
//	EXPECT_EQ(0 != y);
//	*y = 101200;
//	pair = ivmap_locate(tree, &iter, 1100);
//	EXPECT_EQ(1100 == pair.x0 && 1200 == pair.x1 && 0 != pair.y && 101100 == *pair.y);
//	pair = ivmap_locate(tree, &iter, 1200);
//	EXPECT_EQ(1200 == pair.x0 && 1300 == pair.x1 && 0 != pair.y && 101200 == *pair.y);
//	pair = ivmap_locate(tree, &iter, 1300);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//	ivmap_free(tree);
//
//	tree = 0;
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	y = ivmap_insert(tree, 1100, 1200);
//	EXPECT_EQ(0 != y);
//	*y = 101100;
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	y = ivmap_insert(tree, 1300, 1400);
//	EXPECT_EQ(0 != y);
//	*y = 101300;
//	pair = ivmap_locate(tree, &iter, 1100);
//	EXPECT_EQ(1100 == pair.x0 && 1200 == pair.x1 && 0 != pair.y && 101100 == *pair.y);
//	pair = ivmap_locate(tree, &iter, 1200);
//	EXPECT_EQ(1300 == pair.x0 && 1400 == pair.x1 && 0 != pair.y && 101300 == *pair.y);
//	pair = ivmap_locate(tree, &iter, 1300);
//	EXPECT_EQ(1300 == pair.x0 && 1400 == pair.x1 && 0 != pair.y && 101300 == *pair.y);
//	pair = ivmap_locate(tree, &iter, 1400);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//	ivmap_free(tree);
//}
//
//static void ivmap_iterate_test() {
//	ivmap_node_t *tree;
//	ivmap_u64_t *y;
//	ivmap_iter_t iter;
//	ivmap_pair_t pair;
//
//	tree = 0;
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	pair = ivmap_iterate(tree, &iter, 1);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//	ivmap_free(tree);
//
//	tree = 0;
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	y = ivmap_insert(tree, 1100, 1200);
//	EXPECT_EQ(0 != y);
//	*y = 101100;
//	pair = ivmap_iterate(tree, &iter, 1);
//	EXPECT_EQ(1100 == pair.x0 && 1200 == pair.x1 && 0 != pair.y && 101100 == *pair.y);
//	pair = ivmap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//	ivmap_free(tree);
//
//	tree = 0;
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	y = ivmap_insert(tree, 1100, 1200);
//	EXPECT_EQ(0 != y);
//	*y = 101100;
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	y = ivmap_insert(tree, 1200, 1300);
//	EXPECT_EQ(0 != y);
//	*y = 101200;
//	pair = ivmap_iterate(tree, &iter, 1);
//	EXPECT_EQ(1100 == pair.x0 && 1200 == pair.x1 && 0 != pair.y && 101100 == *pair.y);
//	pair = ivmap_iterate(tree, &iter, 0);
//	EXPECT_EQ(1200 == pair.x0 && 1300 == pair.x1 && 0 != pair.y && 101200 == *pair.y);
//	pair = ivmap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//	ivmap_free(tree);
//
//	tree = 0;
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	y = ivmap_insert(tree, 1100, 1200);
//	EXPECT_EQ(0 != y);
//	*y = 101100;
//	pair = ivmap_locate(tree, &iter, 0);
//	EXPECT_EQ(1100 == pair.x0 && 1200 == pair.x1 && 0 != pair.y && 101100 == *pair.y);
//	pair = ivmap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//	pair = ivmap_locate(tree, &iter, 1099);
//	EXPECT_EQ(1100 == pair.x0 && 1200 == pair.x1 && 0 != pair.y && 101100 == *pair.y);
//	pair = ivmap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//	pair = ivmap_locate(tree, &iter, 1100);
//	EXPECT_EQ(1100 == pair.x0 && 1200 == pair.x1 && 0 != pair.y && 101100 == *pair.y);
//	pair = ivmap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//	pair = ivmap_locate(tree, &iter, 1199);
//	EXPECT_EQ(1100 == pair.x0 && 1200 == pair.x1 && 0 != pair.y && 101100 == *pair.y);
//	pair = ivmap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//	pair = ivmap_locate(tree, &iter, 1200);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//	pair = ivmap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//	ivmap_free(tree);
//
//	tree = 0;
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	y = ivmap_insert(tree, 1100, 1200);
//	EXPECT_EQ(0 != y);
//	*y = 101100;
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	y = ivmap_insert(tree, 1200, 1300);
//	EXPECT_EQ(0 != y);
//	*y = 101200;
//	pair = ivmap_locate(tree, &iter, 1100);
//	EXPECT_EQ(1100 == pair.x0 && 1200 == pair.x1 && 0 != pair.y && 101100 == *pair.y);
//	pair = ivmap_iterate(tree, &iter, 0);
//	EXPECT_EQ(1200 == pair.x0 && 1300 == pair.x1 && 0 != pair.y && 101200 == *pair.y);
//	pair = ivmap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//	pair = ivmap_locate(tree, &iter, 1200);
//	EXPECT_EQ(1200 == pair.x0 && 1300 == pair.x1 && 0 != pair.y && 101200 == *pair.y);
//	pair = ivmap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//	pair = ivmap_locate(tree, &iter, 1300);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//	pair = ivmap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//	ivmap_free(tree);
//
//	tree = 0;
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	y = ivmap_insert(tree, 1100, 1200);
//	EXPECT_EQ(0 != y);
//	*y = 101100;
//	tree = ivmap_ensure(tree, +1);
//	EXPECT_EQ(0 != tree);
//	y = ivmap_insert(tree, 1300, 1400);
//	EXPECT_EQ(0 != y);
//	*y = 101300;
//	pair = ivmap_locate(tree, &iter, 1100);
//	EXPECT_EQ(1100 == pair.x0 && 1200 == pair.x1 && 0 != pair.y && 101100 == *pair.y);
//	pair = ivmap_iterate(tree, &iter, 0);
//	EXPECT_EQ(1300 == pair.x0 && 1400 == pair.x1 && 0 != pair.y && 101300 == *pair.y);
//	pair = ivmap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//	pair = ivmap_locate(tree, &iter, 1200);
//	EXPECT_EQ(1300 == pair.x0 && 1400 == pair.x1 && 0 != pair.y && 101300 == *pair.y);
//	pair = ivmap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//	pair = ivmap_locate(tree, &iter, 1300);
//	EXPECT_EQ(1300 == pair.x0 && 1400 == pair.x1 && 0 != pair.y && 101300 == *pair.y);
//	pair = ivmap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//	pair = ivmap_locate(tree, &iter, 1400);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//	pair = ivmap_iterate(tree, &iter, 0);
//	EXPECT_EQ(0 == pair.x0 && 0 == pair.x1 && 0 == pair.y);
//
//	ivmap_free(tree);
//}
int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	random_seed(time(NULL));
    return RUN_ALL_TESTS();
}
