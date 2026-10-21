#include <algorithm>
#include <functional>
#include <vector>
#include <gtest/gtest.h>
#include <cstdint>

#include "random.h"
#include "helper.h"
#include "sort/sort.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;

constexpr size_t listsize = 128;

/// generate some random data
template<typename T>
void generate_list(std::vector<T> &data,
				   const size_t len) noexcept {
	data.resize(len);
	for (size_t i = 0; i < len; ++i) {
		data[i] = rng();
	}
}

using namespace cryptanalysislib;

/// the pluggable parallel sorts, for all unsigned limb types
template<typename T>
class PluggableSort : public testing::Test {};
using SortTypes = testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
TYPED_TEST_SUITE(PluggableSort, SortTypes);

/// sorts with `f` and compares with `cryptanalysislib::sort`
template<typename T, typename F>
static void check_pluggable(F f) {
	for (const size_t s : {0ul, 1ul, 100ul, 10000ul}) {
		std::vector<T> data;
		generate_list(data, s);
		auto expect = data;
		cryptanalysislib::sort(expect.begin(), expect.end());
		f(data);
		EXPECT_EQ(data, expect);
	}
}

TYPED_TEST(PluggableSort, std_sort) {
	using T = TypeParam;
	check_pluggable<T>([](std::vector<T> &d) {
		pluggable_sort(par_if(true), d.begin(), d.end(), std::sort);
	});
}

TYPED_TEST(PluggableSort, mergesort) {
	using T = TypeParam;
	check_pluggable<T>([](std::vector<T> &d) {
		pluggable_mergesort(par_if(true), d.begin(), d.end(), std::sort);
	});
}

TYPED_TEST(PluggableSort, quicksort) {
	using T = TypeParam;
	check_pluggable<T>([](std::vector<T> &d) {
		pluggable_quicksort(par_if(true), d.begin(), d.end(), std::sort);
	});
}

/// ips4o against std::sort, for the key type `T`
template<typename T, typename... Allocator>
static void check_ips4o() {
	for (const size_t s : {0ul, 1ul, 100ul, 513ul, 10000ul, 300000ul}) {
		std::vector<T> data;
		generate_list(data, s);
		// few distinct values (equality buckets)
		if (s > 1000) { for (size_t i = 0; i < s; i += 2) { data[i] = T(data[i] % T(7)); } }
		auto expect = data;
		std::sort(expect.begin(), expect.end());
		ips4o::sort<typename std::vector<T>::iterator, Allocator...>(data.begin(), data.end());
		EXPECT_EQ(data, expect) << s;
	}
}

TEST(ips4o, kek) {
	check_ips4o<uint32_t>();
}

/// other key types
TEST(ips4o, types) {
	check_ips4o<uint8_t>();
	check_ips4o<uint16_t>();
	check_ips4o<uint64_t>();
	check_ips4o<int32_t>();
	check_ips4o<int64_t>();
}

/// other allocator for the internal buffers
TEST(ips4o, allocator) {
	check_ips4o<uint32_t, AlignmentMallocator<uint32_t, 4096>>();
}

/// the pluggable sorts with a comparator (descending), the comparator `sort`
/// and `counting_sort_u8`
TEST(PluggableSortComparator, descending) {
	const auto gt = std::greater<uint32_t>();
	const auto stdsort = [](std::vector<uint32_t>::iterator f, std::vector<uint32_t>::iterator l, std::greater<uint32_t> c) { std::sort(f, l, c); };
	for (const size_t s : {0ul, 1ul, 100ul, 10000ul, 300000ul}) {
		std::vector<uint32_t> data;
		generate_list(data, s);
		auto expect = data;
		std::sort(expect.begin(), expect.end(), gt);

		auto d1 = data, d2 = data, d3 = data, d4 = data;
		pluggable_sort(par_if(true), d1.begin(), d1.end(), gt, +stdsort);
		EXPECT_EQ(d1, expect) << s;
		pluggable_mergesort(par_if(true), d2.begin(), d2.end(), gt, +stdsort);
		EXPECT_EQ(d2, expect) << s;
		pluggable_quicksort(par_if(true), d3.begin(), d3.end(), gt, +stdsort);
		EXPECT_EQ(d3, expect) << s;
		cryptanalysislib::sort(d4.begin(), d4.end(), gt);
		EXPECT_EQ(d4, expect) << s;
	}
}

TEST(counting_sort, u8) {
	for (const size_t s : {0ul, 1ul, 100ul, 511ul, 512ul, 100000ul}) {
		std::vector<uint8_t> d(s);
		for (auto &x : d) { x = rng<uint8_t>(); }
		auto e = d;
		std::sort(e.begin(), e.end());
		counting_sort_u8(d.data(), d.size());
		EXPECT_EQ(d, e) << s;
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}




