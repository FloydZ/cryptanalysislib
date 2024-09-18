#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <omp.h>


#include "../test.h"
#include "container/kAry_type.h"
#include "container/linkedlist.h"
#include "helper.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

constexpr uint64_t N = 10;//1000000;
constexpr uint64_t THREADS = 6;
struct TestStruct {
public:
	uint64_t data;

	///
	bool operator==(const TestStruct &b) const {
		return data == b.data;
	}

	std::strong_ordering operator<=>(const TestStruct &b) const {
		return data <=> b.data;
	}

	friend std::ostream &operator<<(std::ostream &os, TestStruct const &tc) {
		return os << tc.data;
	}
};

/// just generate some structured data
TestStruct new_data(const uint32_t tid = -1) {
	static std::atomic<uint64_t> counter = 1;
	TestStruct ret = TestStruct{};

	if (tid == uint32_t(-1)) {
		ret.data = counter;
	} else {
		ret.data = tid * N + counter;
	}

	counter += 1;
	return ret;
}

/// synchronously fills the data
template<class LinkedList>
void SyncedFill(LinkedList &ll) {
	for (size_t j = 0; j < N; ++j) {
		auto d = new_data();
		EXPECT_EQ(ll.insert(d), 0);
	}
}

///
/// \tparam LinkedList
/// \param ll
template<class LinkedList>
void Fill(LinkedList &ll) {
	std::atomic<uint64_t> ret = 0;

#pragma omp parallel default(none) shared(ll, ret) num_threads(THREADS)
	{
		const uint32_t tid = omp_get_thread_num();
		TestStruct t = {0};
		for (size_t j = 0; j < N; ++j) {
			t.data = tid * N + j + 1;
			ret += ll.insert(t);
			EXPECT_EQ(ret, 0);
		}
	}

	std::cout << ret << " " << N << std::endl;
}

TEST(FreeList, Synced) {
	auto ll = FreeList<TestStruct>();
	TestStruct t;
	t.data = 1;

	SyncedFill(ll);
	EXPECT_EQ(std::is_sorted(ll.begin(), ll.end()), true);
	size_t ctr = 0;
	for (auto const &a: ll) {
		(void) a;
		ctr++;
	}
	EXPECT_EQ(ll.size(), N);
	EXPECT_EQ(ll.size(), ctr);

	// this destroys the linked list, as it cannot handle
	// multiple of the same value.
	//std::fill(ll.begin(), ll.end(), t);

	// how to use the iterator
	//for (auto const &i: ll) {
	//	std::cout << i << " ";
	//}

	// checks the `contains` function
	for (auto const &i: ll) {
		EXPECT_EQ(ll.contains(i), 1);
	}

	for (size_t i = 1; i < N; i++) {
		t.data = i;
		EXPECT_EQ(ll.contains(t), 1);
	}

	// false check for `contains`
	t.data = 0xffffffff - 1;
	EXPECT_EQ(ll.contains(t), 0);

	for (size_t i = 1; i < N; i++) {
		t.data = i;
		EXPECT_EQ(ll.remove(t), 0);
	}

	ll.clear();
}

TEST(FreeList, MultiThreaded) {
	auto ll = FreeList<TestStruct>();
	Fill(ll);
	TestStruct t;

	EXPECT_EQ(std::is_sorted(ll.begin(), ll.end()), true);

	size_t ctr = 0;
	for (auto const &i: ll) {
		(void) i;
		ctr++;
	}
	EXPECT_EQ(ll.size(), THREADS * N);
	EXPECT_EQ(ll.size(), ctr);

	for (size_t i = 1; i < N; i++) {
		t.data = i;
		EXPECT_EQ(ll.remove(t), 0);
	}

	for (auto const &i: ll) {
		EXPECT_EQ(ll.contains(i), 1);
	}

	// false check for `contains`
	t.data = 0xffffffff - 1;
	EXPECT_EQ(ll.contains(t), 0);
	ll.clear();
}


TEST(ConstFreeList, Synced) {
	auto ll = ConstFreeList<TestStruct>();
	TestStruct t;

	// insert backward to enforce to be sorted
	for (size_t j = N; j > 0; j--) {
		t.data = j;
		EXPECT_EQ(ll.insert_front(t), 0);
	}
	ll.print();
	EXPECT_EQ(std::is_sorted(ll.begin(), ll.end()), true);

	size_t ctr = 0;
	for (auto const &i: ll) {
		(void) i;
		ctr++;
	}
	EXPECT_EQ(ll.size(), N);
	EXPECT_EQ(ll.size(), ctr);

	// checks the `contains` function
	for (auto const &i: ll) {
		EXPECT_EQ(ll.contains(i), 1);
	}

	// false check for `contains`
	t.data = 0xffffffff - 1;
	EXPECT_EQ(ll.contains(t), 0);

	// check fill
	t.data = 0;
	std::fill(ll.begin(), ll.end(), t);
	for (auto const &i: ll) {
		EXPECT_EQ(i.data, 0);
	}

	ll.clear();
}


TEST(ConstFreeList, MultiThreaded) {
	auto ll = ConstFreeList<TestStruct>();

#pragma omp parallel default(none) shared(ll) num_threads(THREADS)
	{
		TestStruct t;
		const uint32_t tid = omp_get_thread_num();

		// insert backward
		for (size_t j = N; j > 0; j--) {
			t.data = j + tid * N;
			EXPECT_EQ(ll.insert_front(t), 0);
		}

// NOTE the barrier. If the barrier is missing, some threads
// may already start with counting the elements, whereas
// some still insert elements into the list.
#pragma omp barrier
		size_t ctr = 0;
		for (auto const &i: ll) {
			(void) i;
			ctr++;
		}

		EXPECT_EQ(ll.size(), N * THREADS);
		EXPECT_EQ(ll.size(), ctr);

		// checks the `contains` function
		for (auto const &i: ll) {
			EXPECT_EQ(ll.contains(i), 1);
		}

		//// false check for `contains`
		t.data = 0xffffffff - 1;
		EXPECT_EQ(ll.contains(t), 0);
	}

	// IMPORTANT: this function is not thread save
	ll.clear();
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
