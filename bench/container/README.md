Linked List:
============

The library provides two different implementations of a linked list:
- `FreeList`: sorted lock-free double-linked unique list with deletion
- `ConstFreeList` unsorted lock-free linked list without deletion

Both lists are perfectly suited to be used in a multithreaded setting. 

FreeList:
---------
The only requirement is that `T` must be `comparable` and `copyable`. 
Additional note that each element can only be inserted once. The list
does not allow double elements.

Usage:
```c++
#include "container/linkedlist/linkedlist.h"
constexpr uint64_t N = 10000;   // Number of elements to insert
constexpr uint64_t THREADS = 6; // Number of concurrent threads

// just a dummy data struct
struct TestStruct {
public:
	uint64_t data;
	bool operator==(const TestStruct& b) const { return data == b.data; }
	std::strong_ordering operator<=>(const TestStruct& b) const { return data <=> b.data; }
	friend std::ostream& operator<<(std::ostream& os, TestStruct const &tc) { return os << tc.data; }
};

void main() {
	auto ll = FreeList<TestStruct>();
	#pragma omp parallel default(none) shared(ll) num_threads(THREADS)
	{
		const uint32_t tid = omp_get_thread_num();
		TestStruct t = {0};
		
		// example for insertion
		for (size_t j = 0; j < N; ++j) {
		    // generate some unique test data 
			t.data = tid*N + j + 1;
			assert(ll.add(t) == 0);
		}
		
		// example for iteration
	    size_t ctr = 0;
	    for (auto const &i: ll) {
	        std::cout << i << std::endl;
	    }
	    
	    // example of the `contains` function
	    for (auto const &i: ll) {
	    	assert(ll.contains(i));
	    }
	   
	    // example for deletion
	    for (size_t i = 1; i < N; i++) {
	    	t.data = i;
	    	assert(ll.remove(t) == 0);
	    }
	}
}
```


ConstFreeList:
--------------
Prequirements: None

Usage:
```c++
	auto ll = ConstFreeList<TestStruct>();

	#pragma omp parallel default(none) shared(ll) num_threads(THREADS)
	{
		TestStruct t;
		const uint32_t tid = omp_get_thread_num();

		// insert backward, so the stuff is sorted.
		// not really needed
		for (size_t j = N; j > 0; j--) {
			t.data = j + tid*N;
			EXPECT_EQ(ll.insert_front(t), 0);
		}
		
		size_t ctr = 0;
		for (auto const &i: ll) {
			ctr++;
		}

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
```