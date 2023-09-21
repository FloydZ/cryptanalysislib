#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <thread>

#include "../test.h"
#include "container/linked_list.h"
#include "helper.h"
#include "kAry_type.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

constexpr uint64_t N = 1000;
constexpr uint64_t TT = 1;
struct TestStruct {
	uint64_t data;
	std::atomic<TestStruct *> next;
};

int compare(const TestStruct *a, const TestStruct *b) {
	return a->data < b->data;
}

TestStruct *new_data(const uint32_t tid = -1){
	static uint64_t counter = 0;
	TestStruct *ret = new TestStruct;

	if (tid == -1)
		ret->data = counter;
	else
		ret->data = tid;

	ret->next.store(nullptr);
	if (tid == -1)
		counter += 1;
	return ret;
}

template<class LinkedList>
void Fill(LinkedList &ll){
	std::vector<std::thread> pool(TT);
	for (size_t i = 0; i < TT; ++i) {
		pool[i] = std::thread([&ll](){
			for (int j = 0; j < N; ++j) {
				uint32_t tid = std::hash<std::thread::id>{}(std::this_thread::get_id());
				//auto d = new_data(tid);
				auto d = new_data();
				ll.insert(d);
			}
		});
	}

	for (auto &t: pool) {
		t.join();
	}
}


TEST(LinkedList, LinkedList) {
	auto ll = ConstNonBlockingLinkedList<TestStruct, compare>();
	Fill(ll);
	ll.print();
}

TEST(LinkedList, MiddleNode) {
	auto ll = ConstNonBlockingLinkedList<TestStruct, compare>();
	Fill(ll);
	auto middle = ll.middle_node();
	std::cout << middle->data << "\n";

}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
    return RUN_ALL_TESTS();
}
