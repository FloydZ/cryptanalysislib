#ifndef CRYPTANALYSISLIB_CACHE_H
#define CRYPTANALYSISLIB_CACHE_H

#include "helper.h"
#include "container/linkedlist.h"
#include "popcount/popcount.h"

template<class T>
class CacheAllocator {
	constexpr static size_t bits = 64;
	std::atomic<uint32_t> ctr = 0;

	class Node {

		using Limb = LogTypeTemplate<bits>;
		using ALimb = std::atomic<Limb>;

		alignas(64) T data[bits];
		ALimb free = ALimb(-1);

	public:

		/// \param ptr
		/// \return
		constexpr inline bool owns(const T *ptr) {
			return ((((uintptr_t)(data + bits)) - ((uintptr_t)ptr)) / sizeof(T)) <= bits;
		}

		/// \param ptr
		/// \return
		constexpr inline void deallocate(const T *ptr) {
			uint32_t pos = bits - ((((uintptr_t)(data + bits)) - ((uintptr_t)ptr)) / sizeof(T));

			Limb d, nd;
			do {
				d = free.load();
				nd = d ^ (1ull << pos);
			} while(!free.compare_exchange_weak(d, nd));
		}

		/// could be named `is_slot_free`. If so it returns true
		/// \param ptr
		/// \return
		constexpr inline bool allocate(T **ptr) noexcept {
			uint32_t pos;
			Limb d, nd;
			do {
				d = free.load();
				if (d == 0) {
					return false;
				}

				pos = __builtin_ctzll(d);
				nd = d ^ (1ull << pos);
			} while(!free.compare_exchange_weak(d, nd));

			*ptr = data + pos;
			return true;
		}

		Node() noexcept {
			free.store(-1);
			memset((void *)data, 0, sizeof(T) * bits);
		}

		Node(const Node &t) noexcept {
			this->free.store(t.free.load());
			// TODO this looks wrong
			memcpy(data, data, sizeof(T) *bits);
		}

		bool operator==(const Node &b) const noexcept {
			return (uintptr_t)data == (uintptr_t)b.data;
		}
	};

	using LinkedList = ConstFreeList<Node>;
	LinkedList root{};

public:
	constexpr CacheAllocator() noexcept {
		root.insert(Node{});
		ctr.store(0);
	}

	constexpr inline uint32_t size() noexcept {
		return ctr.load();
	}
	///
	/// \return
	T* allocate() {
		ctr.fetch_add(1);

		while (true) {
			// first find a free entry in
			for (auto &i: root) {
				T *ptr = nullptr;
				if (i.allocate(&ptr)) {
					// this case we found an empty slot
					return ptr;
				}
			}

			// allocate a new node, ignore if multiple are added
			root.insert(Node{});
		}
	}

	///
	bool deallocate(T *ptr) {
		for (auto &i: root) {
			if (i.owns(ptr)) {
				i.deallocate(ptr);
				ctr.fetch_sub(1);
				return true;
			}
		}

		// if we reach here, the pointer was invalid
		return false;
	}
};

#endif//CRYPTANALYSISLIB_CACHE_H
