#ifndef CRYPTANALYSISLIB_CACHE_H
#define CRYPTANALYSISLIB_CACHE_H

#include "helper.h"
#include "container/linkedlist.h"
#include "memory/memory.h"


/// Configuration for the cache allocator
struct CacheAllocatorConfig {
	/// number of elements to store in a single bucket
	constexpr static size_t bits = 64;
};
constexpr static CacheAllocatorConfig cacheAllocatorConfig;

/// Floyds simple try of super simple allocator, which is made for
/// caches. It stores `bits` many `T` typed elements in single bucket, which
/// are extended via a linked list.
/// NOTE: Dont use it for anything useful
/// NOTE: Seams to be super slow. Lol.
/// \tparam T[in]: type to allocate
/// \tparam LinkedList[in]: linked list implementation to use
/// \tparam config[in]: configuration for the cache allocator
template<class T,
		 template<class L1, 
                  template<class >class L2=cryptanalysislib::allocator,
                  class L3=std::atomic<T>> class LinkedList = ConstFreeList,
		 const CacheAllocatorConfig &config=cacheAllocatorConfig>
class CacheAllocator {
	// number of elements to store in a single bucket
	constexpr static size_t bits = config.bits;
	std::atomic<uint32_t> ctr = 0;

	/// node in linked list
	class Node {
		using Limb = LogTypeTemplate<bits>;
		using ALimb = std::atomic<Limb>;

		alignas(64) T data[bits];
		ALimb free = ALimb(-1);

	public:

		/// Checks if this node owns the given pointer
		/// \param ptr[in]: pointer to check ownership for
		/// \return true if the pointer is within this node's memory range
		[[nodiscard]] constexpr inline bool owns(const T *ptr) noexcept {
			return ((((uintptr_t)(data + bits)) - ((uintptr_t)ptr)) / sizeof(T)) <= bits;
		}

		/// Deallocates a previously allocated pointer from this node
		/// \param ptr[in]: pointer to deallocate
		constexpr inline void deallocate(const T *ptr) noexcept {
			const uint32_t pos = bits - ((((uintptr_t)(data + bits)) - ((uintptr_t)ptr)) / sizeof(T));

			Limb d, nd;
			do {
				d = free.load();
				nd = d ^ (1ull << pos);
			} while(!free.compare_exchange_weak(d, nd));
		}

		/// Attempts to allocate a slot from this node
		/// \param ptr[out]: pointer to the pointer to allocate
		/// \return true if a slot was successfully allocated, false if no slots are available
		[[nodiscard]] constexpr inline bool allocate(T **ptr) noexcept {
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

		/// Default constructor - initializes a new node with all slots free
		constexpr Node() noexcept {
			free.store(-1);
			memset((void *)data, 0, sizeof(T) * bits);
		}

		/// Copy constructor - creates a copy of an existing node
		/// \param t[in]: node to copy from
		constexpr Node(const Node &t) noexcept {
			this->free.store(t.free.load());
			cryptanalysislib::memcpy(data, t.data, sizeof(T));
		}

		/// Equality operator - compares if two nodes have the same data pointer
		/// \param b[in]: node to compare with
		/// \return true if the nodes have the same data pointer
		[[nodiscard]] constexpr bool operator==(const Node &b) const noexcept {
			return (uintptr_t)data == (uintptr_t)b.data;
		}
	};

	LinkedList<Node> root{};

public:
	constexpr CacheAllocator() noexcept {
		root.insert(Node{});
		ctr.store(0);
	}

	constexpr ~CacheAllocator() noexcept {
		// NOTE: doesn't make much sense, as the datastructure
		// doesn't save any ptr data.
	}

	/// Returns the current number of allocated elements
	/// \return number of active allocations
	[[nodiscard]] constexpr inline uint32_t size() noexcept {
		return ctr.load();
	}

	/// Allocates memory for a new element
	/// \return pointer to the allocated memory
	[[nodiscard]] T* allocate() noexcept {
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

	/// Deallocates a previously allocated pointer
	/// \param ptr[in]: pointer to deallocate
	/// \return true if deallocation was successful, false if the pointer was invalid
	[[nodiscard]] bool deallocate(const T *ptr) noexcept {
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
