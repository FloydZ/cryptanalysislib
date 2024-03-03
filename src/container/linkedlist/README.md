Two different linked lists are implemented. Note for both lists the container 
class need to implement the C++20 ternary comparison operator

### Lock-Free-Const-Multithreaded-SingleLinkedlist:

This implementation is useful if you need:
- insert/read elements multithreaded 
- never delete a single element. If a element needs to be removed, all elements 
    are removed.
- Unidirectional iterator


### Lock-Free-Mutlithreaded-DoubleLinkedList:
This implementation is useful if you need:
- insert/read/remove elements multithreaded
- Bidirectional Iterator

### Usage:

```C
struct TestStruct {
public:
	uint64_t data;
	bool operator==(const TestStruct &b) const {
		return data == b.data;
	}

	std::strong_ordering operator<=>(const TestStruct &b) const {
		return data <=> b.data;
	}
};

auto ll = FreeList<TestStruct>();
ll.insert(TestStruct{.data = 12});
assert(std::is_sorted(ll.begin(), ll.end())
```
