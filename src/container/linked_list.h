#ifndef SMALLSECRETLWE_LIST_H
#define SMALLSECRETLWE_LIST_H

#include <stdint.h>

#include "helper.h"

namespace cryptanalysislib {
	struct config_linked_list {
		bool sort = true;
	};


	template<class T, const size_t N>
	class linked_list {

	};



};

// https://www.cs.purdue.edu/homes/xyzhang/fall14/lock_free_set.pdf
// https://moodycamel.com/blog/2014/solving-the-aba-problem-for-lock-free-free-lists
// https://users.fmi.uni-jena.de/~nwk/LockFree.pdf
// IMPORTANT: T must implement a field next
template<typename T, int (*c)(const T*, const T*)>
class ConstNonBlockingLinkedList {
private:
	// Atomic Type
	using AT = std::atomic<T *>;
public:
	ConstNonBlockingLinkedList() {
		head.store(nullptr);
	}

	/// IDEA: preallocate len elements?
	explicit ConstNonBlockingLinkedList(const uint32_t len) {
		// make sure that the head is correctly initialised
		head.store(nullptr);
	}

#if defined(__x86_64__)
	// only needed for the C version.
	int CAS(void **mem, void *o, void *n) {
		int res;
		asm("lock cmpxchg %3,%1; "
			"mov $0,%0;"
			"jnz 1f; "
			"inc %0; 1:"
				: "=a" (res) : "m" (*mem), "a" (o), "d" (n));
		return res;
	}
#else
	int CAS(void **mem, void *o, void *n) {
		// TODO ,ove alll of this into seperate class
		return 0;
	}
#endif

	/// insert front, unsorted
	void insert(T *a) {
		auto newhead = head.load();
		do {
			a->next.store(newhead);
		} while (!head.compare_exchange_weak(newhead, a, std::memory_order_release, std::memory_order_relaxed));
	}

	/// insert back, unsorted
	void insert_back(T *a) {
		auto back = traverse();
		auto newback = back->load();
		back->compare_exchange_strong(newback, a, std::memory_order_release, std::memory_order_relaxed);
	}

	// return the last element in the list.
	AT* traverse() {
		// catch the case where the list is empty
		if (head.load() == nullptr)
			return &head;

		auto newhead = &head;
		while (newhead->load()->next != nullptr){
			newhead = &newhead->load()->next;
		}

		return newhead;
	}

	/// returns the middle node of the linked list.
	T* middle_node() {
		T *first = head.load();
		T *last = traverse()->load();

		if (first == nullptr) {
			return nullptr;
		}

		T *sm = first;
		T *fm = first->next;
		while (fm != last) {
			fm = fm->next;
			if (fm != last) {
				sm = sm->next;
				fm = fm->next;
			}
		}

		return sm;
	}

	T* binary_search(const T *p) {
		T *fn= head.load();
		T *ln = nullptr;
		T *cn = nullptr;
		int tmp;

		if (fn == nullptr) {
			return nullptr;
		}

		cn = middle_node(fn, ln);
		while ((cn != nullptr) && (fn != ln)) {
			tmp = c(cn, p);
			if (tmp == 0) {
				return cn;
			} else if (tmp == -1) {
				fn = cn->next;
			} else {
				ln = cn;
			}

			if (fn != ln) {
				cn = middle_node(fn, ln);
			}
		}

		return nullptr;
	}

	void insert_sorted(const T *a) {
		T *fn= head.load();
		T *ln = nullptr;
		T *cn = nullptr;

		if (fn == nullptr) {
			// input list empty
			insert(a);
			return;
		} else if (c(fn->next, a) >= 1) {
			insert(a);
			return;
		}

		cn = middle_node(fn, ln);
		while ((cn->next != nullptr)  && (fn != ln)) {
			if (c(cn->next, a) == -1) {
				fn = cn->next;
			} else {
				ln = cn;
			}

			if (fn != ln) {
				cn = middle_node(fn, ln);
			}
		}

		a->next = fn->next;
		fn->next = a;
	}


	void print(){
		auto next = head.load();
		while (next != nullptr) {
			std::cout << next->data << "\n";
			next = next->next.load();
		}
	}
	AT head;
};

#endif //SMALLSECRETLWE_LIST_H
