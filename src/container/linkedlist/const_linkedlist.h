#ifndef CRYPTANALYSISLIB_CONST_LINKEDLIST_H
#define CRYPTANALYSISLIB_CONST_LINKEDLIST_H

#if !defined(CRYPTANALYSISLIB_LINKEDLIST_H)
#error "Do not include this file directly. Use: `#include <container/linkedlist.h>`"
#endif

#include <atomic>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <simd/simd.h>

#include "alloc/alloc.h"

/// Source:
///  https://www.cs.purdue.edu/homes/xyzhang/fall14/lock_free_set.pdf
///  https://moodycamel.com/blog/2014/solving-the-aba-problem-for-lock-free-free-lists
///  https://users.fmi.uni-jena.de/~nwk/LockFree.pdf
/// IMPORTANT: this linkedlist does not implement the remove operator.
/// 		Hence, the ABA problem is not a thing
/// unsorted single-linked list
/// \tparam T
template<typename T,
		 class Allocator = cryptanalysislib::alloc::allocator,
         class A = std::atomic<T>>
class ConstFreeList {
private:
	struct Node {
		constexpr Node() noexcept : next(nullptr) {}
		constexpr Node(T data) noexcept : next(nullptr), data(data) {}

		std::atomic<Node *> next;
		T data;

	public:
		bool operator==(const Node &b) const noexcept {
			return (uintptr_t)data == (uintptr_t)b.data;
		}

		constexpr Node(Node &t) noexcept {
			this->next.store(t.next.load());
			this->data = t.data;
		}
	};

	struct Iterator {
	public:
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using value_type = T;
		using pointer = T *;
		using reference = T &;
		using internal_pointer = Node *;

		///
		constexpr Iterator(const internal_pointer ptr) : m_ptr(ptr) {}

		///
		constexpr reference operator*() noexcept { return m_ptr->data; }
		constexpr reference operator*() const noexcept { return m_ptr->data; }

		constexpr pointer operator->() noexcept { return &(m_ptr->data); }

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

		[[nodiscard]] constexpr friend bool operator==(const Iterator &a,
													   const Iterator &b) noexcept {
			return a.m_ptr == b.m_ptr;
		};
		[[nodiscard]] constexpr friend bool operator!=(const Iterator &a,
													   const Iterator &b) noexcept {
			return a.m_ptr != b.m_ptr; 
		};

	private:
		internal_pointer m_ptr;
	};

	/// allocate the first `LEN` nodes into this buffer,
	constexpr static bool USE_BUFFER = false;
	constexpr static size_t LEN = 1024;
	Node __internal_array[LEN];

	/// keep track of the size of the linked list
	std::atomic<size_t> __size = 0;
	std::atomic<Node *> head;

public:
	constexpr inline Iterator begin() noexcept { return Iterator(head); }
	constexpr inline Iterator end() noexcept { return nullptr; }
	constexpr inline Iterator begin() const noexcept { return Iterator(head); }
	constexpr inline Iterator end() const noexcept { return nullptr; }

	///
	constexpr ConstFreeList() noexcept {
		/// create an empty element
		head.store(nullptr);
	}

	/// free everything
	/// NOTE: not thread save. Only call by a single thread
	constexpr ~ConstFreeList() noexcept {
		clear();
	}

	/// returns 1 if element is in list, 0 else
	constexpr int contains(const T &data) noexcept {
		Node *c = head.load();
		while (c != nullptr) {
			if (c->data == data) {
				return true;
			}

			c = c->next.load();
		}

		return false;
	}

	/// insert front, unsorted
	/// returns 0 on success. This function cannot fail
	constexpr int insert_front(const T &data) noexcept {
		Node *current_head = head.load();
		// TODO replace with allocator
		auto new_head = new Node(data);
		do {
			new_head->next.store(current_head);
		} while (!head.compare_exchange_weak(current_head,
		                                     new_head,
		                                     std::memory_order_release,
		                                     std::memory_order_relaxed));

		__size.fetch_add(1u);
		return 0;
	}

	/// insert unsorted
	/// returns zero on success, cannot fail
	constexpr int insert(const T &data) noexcept {
		return insert_front(data);
	}

	/// IMPORTANT: this function is not thread safe
	/// this clears the whole data struct
	constexpr void clear() noexcept {
		Node *c = head.load(), *next;
		while (c != nullptr) {
			next = c->next.load();
			delete c;
			c = next;
		}

		__size = 0;
	}

	constexpr size_t size() const noexcept { return __size; }
	constexpr void print() const noexcept {
		auto next = head.load();
		while (next != nullptr) {
			std::cout << next->data << "\n";
			next = next->next.load();
		}
	}
};

#endif//CRYPTANALYSISLIB_CONST_LINKEDLIST_H
