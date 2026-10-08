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
template<typename T> // TODO allocator
#if __cplusplus > 201709L 
    requires std::copyable<T> && std::three_way_comparable<T>
#endif
struct FreeList {
private:
	/// internal struct
    // TODO something like this template<template<typename> typename A=std::atomic>
	struct Node {
		Node() : next(nullptr), prev(nullptr) {}
		Node(const T data) : next(nullptr), prev(nullptr), data(data) {}

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
			m_ptr = getpointer(m_ptr->next.load());
			return *this;
		}

		// Postfix increment
		Iterator operator++(int) {
			Iterator tmp = *this;
			m_ptr = getpointer(m_ptr->next.load());
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
	Node *head = nullptr,  // start of the linked list
	     *tail = nullptr;  // end of the linked list
	// start of a second linked list of removed (but not freed) elements
	// NOTE: atomic, `remove` pushes from several threads
	std::atomic<Node *> __free = nullptr;
	// NOTE: only a hint where to start searching. Before, `pos()` returned
	// 	its result through the shared members `pred`/`curr`, which other
	// 	threads overwrote in between.
	std::atomic<Node *> hint = nullptr;
	// whether the sentinels were allocated by the constructor
	bool own_head = false, own_tail = false;

	/// pointer stuff: we need to mark/tag pointers to counter the ABA problem
	constexpr static uintptr_t UNMARK_MASK = ~1;
	constexpr static uintptr_t MARK_BIT = 1;
	constexpr static inline Node *getpointer(const Node *ptr) noexcept { return (Node *) ((uintptr_t) ptr & UNMARK_MASK); }
	constexpr static inline bool ismarked(const Node *ptr) noexcept { return (((uintptr_t) ptr) & MARK_BIT) != 0; }
	constexpr static inline Node *setmark(const Node *ptr) noexcept { return (Node *) (((uintptr_t) ptr) | MARK_BIT); }

	/// \return true if `data` lies strictly between the two sentinels, i.e.
	/// 	if it can be stored in the list
	constexpr inline bool in_range(const T &data) const noexcept {
		return (head->data < data) && (data < tail->data);
	}

	/// allocate the first `LEN` nodes into this buffer,
	constexpr static bool USE_BUFFER = false;
	constexpr static size_t LEN = 1024;
	// NOTE: only allocated if used. Before, every list contained `LEN`
	// 	nodes, also with `USE_BUFFER == false`.
	Node __internal_array[USE_BUFFER ? LEN : 1];
	// NOTE: per list. Was a `static` in `insert`, i.e. shared by all lists.
	std::atomic<size_t> __buffer_ctr = 0;

	/// frees `n`, if it was allocated with `new` (and not from the buffer)
	constexpr inline void release(Node *n) noexcept {
		if constexpr (USE_BUFFER) {
			if ((n >= __internal_array) && (n < __internal_array + LEN)) {
				return;
			}
		}
		delete n;
	}

	/// keep track of the size of the linked list
	std::atomic<size_t> __size = 0;

	/// finds the position of `data` within the linked list
	/// internal function, dont use it.
	/// \param out_pred[out]: last node with `data > out_pred->data`
	/// \param out_curr[out]: first node with `data <= out_curr->data`
	inline void pos(const T &data,
	                Node *&out_pred,
	                Node *&out_curr) noexcept {
		Node *__pred, *__succ, *__curr, *__next;
		__pred = hint.load(std::memory_order_relaxed);
	retry:
		while (ismarked(__pred->next.load()) || data <= __pred->data) {
			__pred = __pred->prev.load();
		}
		__curr = getpointer(__pred->next.load());
		assert(__pred->data < data);

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
				assert(__pred->data < __curr->data);
				hint.store(__pred, std::memory_order_relaxed);
				out_pred = __pred;
				out_curr = __curr;
				return;
			}

			__pred = __curr;
			__curr = getpointer(__curr->next.load());
		} while (true);
	}

public:
	// NOTE: `head` and `tail` are sentinels and not part of the list
	Iterator begin() { return Iterator(getpointer(head->next.load())); }
	Iterator end() { return Iterator(tail); }

	constexpr FreeList(Node *__head = nullptr, Node *__tail = nullptr) {
		// NOTE: before, user supplied sentinels were never stored
		head = __head;
		tail = __tail;
		if (__head == nullptr) {
			head = new Node;
			own_head = true;
			// this is kind of strange. But the start and the end need to
			// initialized to the lowest possible value.
			std::memset(&head->data, 0, sizeof(T));
		}

		if (__tail == nullptr) {
			tail = new Node;
			own_tail = true;
			std::memset(&tail->data, -1, sizeof(T));
		}

		if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
			// the bit patterns 0...0 and 1...1 are not the min/max of
			// signed integers
			using U = std::make_unsigned_t<T>;
			constexpr uint32_t bits = sizeof(T) * 8u;
			if (__head == nullptr) { head->data = T(U(1) << (bits - 1u)); }
			if (__tail == nullptr) { tail->data = T(U(~U(0)) >> 1u); }
		}

		// initialize the start and the end of the linked list to point to
		// each other.
		head->prev = nullptr;
		head->next = tail;
		tail->prev = head;
		tail->next = nullptr;

		hint.store(head);
	}

	// NOTE: the nodes are owned by the list
	FreeList(const FreeList &) = delete;
	FreeList &operator=(const FreeList &) = delete;

	/// NOTE: not thread safe. Before, no node was ever freed.
	~FreeList() noexcept {
		// nodes still linked and not marked; marked ones are on `__free`
		Node *c = getpointer(head->next.load());
		while ((c != nullptr) && (c != tail)) {
			Node *next = c->next.load();
			if (!ismarked(next)) {
				release(c);
			}
			c = getpointer(next);
		}

		clean();
		if (own_head) { delete head; }
		if (own_tail) { delete tail; }
	}

	/// return 0 on success, 1 else
	inline int insert(const T &data) noexcept {
		Node *__pred, *__curr, *__node;

		// the values of the two sentinels cannot be stored
		if (!in_range(data)) {
			return 1;
		}

		if constexpr (USE_BUFFER) {
			/// if the flag is set
			// NOTE: take a buffer node while there are some left. Was
			// 	`if (ctr >= LEN)`, i.e. reading past the buffer.
			const size_t c = __buffer_ctr.fetch_add(1u);
			if (c < LEN) {
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
			pos(data, __pred, __curr);

			/// data already inserted
			if (__curr->data == data) {
				// NOTE: was leaked, `__node` has not been published
				release(__node);
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
		if (!in_range(data)) {
			return 0;
		}

		Node *__curr = hint.load(std::memory_order_relaxed);
		while (data < __curr->data) {
			__curr = __curr->prev.load();
		}

		assert(__curr->data <= data);

		while (data > __curr->data) {
			__curr = getpointer(__curr->next.load());
		}

		return ((__curr->data == data) && (!ismarked(__curr->next.load())));
	}

	/// returns 1 on error (no element in), 0 else
	constexpr inline int remove(const T &data) {
		Node *__pred, *__succ, *__node, *__markedsucc;

		if (!in_range(data)) {
			return 1;
		}

		do {
			pos(data, __pred, __node);
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

			// NOTE: a failed CAS overwrites its expected argument, so do not
			// 	pass `__node` itself. If the unlink fails, the node is marked
			// 	and the next `pos()` unlinks it.
			Node *expected = __node;
			__pred->next.compare_exchange_weak(expected, __succ);

			__succ->prev.store(__pred);
			// NOTE: atomic push, several threads may remove at the same time
			__node->free = __free.load();
			while (!__free.compare_exchange_weak(__node->free, __node)) {}
			__size.fetch_sub(1u);
			return 0;
		} while (true);
	}

	///
	constexpr size_t size() noexcept {
		return __size;
	}

	/// clean the all __free->free->free and so so.
	/// those are all the elements which where removed
	/// NOTE: not thread safe
	constexpr void clean() noexcept {
		Node *node, *next = __free.load();
		while (next != nullptr) {
			node = next;
			next = node->free;
			release(node);
		}

		__free = nullptr;
		// NOTE: the hint may point to a node freed above
		hint.store(head);
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
