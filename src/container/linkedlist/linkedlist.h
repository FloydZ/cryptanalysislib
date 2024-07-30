#ifndef CRYPTANALYSISLIB_CONTAINER_LINKEDLIST_H
#define CRYPTANALYSISLIB_CONTAINER_LINKEDLIST_H

#if !defined(CRYPTANALYSISLIB_LINKEDLIST_H)
#error "Do not include this file directly. Use: `#include <container/linkedlist.h>`"
#endif

#include <stdint.h>
#include <cstring> // for memset

#include "helper.h"

/// main src: https://moodycamel.com/blog/2014/solving-the-aba-problem-for-lock-free-free-lists
/// (sorted) Lock Free Double Linked List
/// Note:
///		- at all time each value T is only allowed once in the list.
///			e.g. if you run std::fill(...) you destroy the list
///		- cannot insert the same value as head. Thus you can insert
/// 		a custom head via the constructor, which is smaller than every element
/// 		you insert and will never be deleted.
/// IMPROVEMENTS:
/// 	- introduce Free Node which or direct delete
///
/// \tparam T
template<typename T>
#if __cplusplus > 201709L
    requires std::copyable<T> && std::three_way_comparable<T>
#endif
struct FreeList {
private:
	/// internal struct
	struct Node {
		Node() : next(nullptr), prev(nullptr) {}
		Node(T data) : next(nullptr), prev(nullptr), data(data) {}

		std::atomic<Node *> next;
		std::atomic<Node *> prev;
		Node *free;
		T data;
	};

	struct Iterator {
	public:
		using iterator_category = std::bidirectional_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using value_type = T;
		using pointer = T *;
		using reference = T &;
		using internal_pointer = Node *;

		Iterator(internal_pointer ptr) : m_ptr(ptr) {}
		reference operator*() const { return m_ptr->data; }
		pointer operator->() { return &(m_ptr->data); }

		// Prefix increment
		Iterator &operator++() {
			m_ptr = m_ptr->next.load();
			return *this;
		}

		// Postfix increment
		Iterator operator++(int) {
			Iterator tmp = *this;
			m_ptr = m_ptr->next.load();
			return tmp;
		}

		friend bool operator==(const Iterator &a, const Iterator &b) { return a.m_ptr == b.m_ptr; };
		friend bool operator!=(const Iterator &a, const Iterator &b) { return a.m_ptr != b.m_ptr; };

	private:
		internal_pointer m_ptr;
	};

	// read as node to be freed
	struct FreeNode {
		Node *free;
	};


	/// internal pointers
	Node *head = nullptr,                           // start of the linked list
	        *tail = nullptr,                        // end of the linked list
	                *__free = nullptr,              // start of a second linked list of removed (but not freed) elements
	                        *curr = nullptr,        //
	                                *pred = nullptr;//

	/// pointer stuff: we need to mark/tag pointers to counter the ABA problem
	constexpr static uintptr_t UNMARK_MASK = ~1;
	constexpr static uintptr_t MARK_BIT = 1;
	constexpr inline Node *getpointer(const Node *ptr) noexcept { return (Node *) ((uintptr_t) ptr & UNMARK_MASK); }
	constexpr inline bool ismarked(const Node *ptr) noexcept { return (((uintptr_t) ptr) & MARK_BIT) != 0; }
	constexpr inline Node *setmark(const Node *ptr) noexcept { return (Node *) (((uintptr_t) ptr) | MARK_BIT); }

	/// allocate the first `LEN` nodes into this buffer,
	constexpr static bool USE_BUFFER = false;
	constexpr static size_t LEN = 1024;
	Node __internal_array[LEN];

	/// keep track of the size of the linked list
	std::atomic<size_t> __size = 0;

	/// finds the position of `data` within the linked list
	/// internal function, dont use it.
	inline void pos(const T &data) noexcept {
		Node *__pred, *__succ, *__curr, *__next;
		__pred = pred;
	retry:
		while (ismarked(__pred->next.load()) || data <= __pred->data) {
			__pred = __pred->prev.load();
		}
		__curr = getpointer(__pred->next.load());
		ASSERT(__pred->data < data);

		do {
			__succ = __curr->next.load();
			while (ismarked(__succ)) {
				__succ = getpointer(__succ);
				if (!__pred->next.compare_exchange_weak(__curr, __succ)) {
					__next = __pred->next.load();
					if (ismarked(__next)) {
						goto retry;
					}

					__succ = __next;
				} else {
					__succ->prev.store(__pred);
				}

				__curr = getpointer(__succ);
				__succ = __succ->next.load();
			}

			if (__curr->prev.load() != __pred) {
				__curr->prev.store(__pred);
			}

			/// set
			if (data <= __curr->data) {
				ASSERT(__pred->data < __curr->data);
				pred = __pred;
				curr = __curr;
				return;
			}

			__pred = __curr;
			__curr = getpointer(__curr->next.load());
		} while (true);
	}

public:
	Iterator begin() { return Iterator(head); }
	Iterator end() { return Iterator(tail->prev.load()); }

	constexpr FreeList(Node *__head = nullptr, Node *__tail = nullptr) {
		if (__head == nullptr) {
			head = new Node;
			// this is kind of strange. But the start and the end need to
			// initialized to the lowest possible value.
			std::memset(&head->data, 0, sizeof(T));
		}

		if (__tail == nullptr) {
			tail = new Node;
			std::memset(&tail->data, -1, sizeof(T));
		}

		// initialize the start and the end of the linked list to point to
		// each other.
		head->prev = nullptr;
		head->next = tail;
		tail->prev = head;
		tail->next = nullptr;

		pred = head;
		curr = nullptr;
	}

	///
	constexpr ~FreeList() {}

	/// return 0 on success, 1 else
	inline int insert(const T &data) noexcept {
		Node *__pred, *__curr, *__node;

		if constexpr (USE_BUFFER) {
			/// if the flag is set
			static std::atomic<size_t> ctr = 0;
			if (ctr >= LEN) {
				const size_t c = ctr.fetch_add(1u);
				__node = &__internal_array[c];
				__node->data = data;
				__node->next = nullptr;
				__node->prev = nullptr;
				__node->free = nullptr;
			} else {
				__node = new Node{data};
			}
		} else {
			__node = new Node{data};
		}

		do {
			pos(data);
			__pred = pred;
			__curr = curr;

			/// data already inserted
			if (__curr->data == data) {
				// delete __node;
				return 1;
			}

			__node->next = __curr;
			__node->prev = __pred;

			if (__pred->next.compare_exchange_weak(__curr, __node)) {
				__curr->prev.store(__node);
				__size.fetch_add(1u);
				return 0;
			}
		} while (true);
	}

	/// returns 1 if element is in list, 0 else
	constexpr inline int contains(const T &data) {
		Node *__curr = pred;
		while (data < __curr->data) {
			__curr = __curr->prev.load();
		}

		assert(__curr->data <= data);

		while (data > __curr->data) {
			__curr = getpointer(__curr->next.load());
		}

		pred = __curr;
		return ((__curr->data == data) && (!ismarked(__curr->next.load())));
	}

	/// returns 1 on error (no element in), 0 else
	constexpr inline int remove(const T &data) {
		Node *__pred, *__succ, *__node, *__markedsucc;

		do {
			pos(data);
			__pred = pred;
			__node = curr;
			if (__node->data != data) {
				return 1;
			}

			__succ = __node->next.load();
			do {
				if (ismarked(__succ)) {
					return 1;
				}

				__markedsucc = setmark(__succ);
				if (__node->next.compare_exchange_weak(__succ, __markedsucc)) {
					break;
				}
			} while (1);

			if (!__pred->next.compare_exchange_weak(__node, __succ)) {
				__node = curr;
			}

			__succ->prev.store(__pred);
			__node->free = __free;
			__free = __node;
			return 0;
		} while (true);
	}

	///
	constexpr size_t size() noexcept {
		return __size;
	}

	/// clean the all __free->free->free and so so.
	/// those are all the elements which where removed
	constexpr void clean() noexcept {
		Node *node, *next = __free;
		while (next != nullptr) {
			node = next;
			next = node->free;
			delete node;
		}

		__free = nullptr;
	}

	/// clears the whole list
	constexpr void clear() noexcept {
		Node *__curr = head->next.load(), *next;

		while (getpointer(__curr) != nullptr) {
			next = __curr->next.load();
			//make sure that we do not clear the tail
			if ((getpointer(next) == nullptr)) {
				break;
			}
			remove(__curr->data);

			// for sure not correct
			__curr = next;
		}

		clean();
	}

	/// print
	constexpr void print(const bool backward = false) const noexcept {
		if (backward) {
			Node *next = tail;
			while (next != nullptr) {
				std::cout << next->data << "\n";
				next = next->prev.load();
			}
		} else {
			Node *next = head;
			while (next != nullptr) {
				std::cout << next->data << "\n";
				next = next->next.load();
			}
		}
	}
};
#endif
