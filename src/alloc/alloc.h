#ifndef SMALLSECRETLWE_ALLOC_H
#define SMALLSECRETLWE_ALLOC_H

#include <cstdlib>
#include <iostream>
#include <limits>

#include "container/queue.h"
#include "helper.h"
#include "memory/memory.h"

///
/// \tparam alignment in bytes
/// \param n input to align up to a multiple of `alignment`
/// \return the up aligned value
template<const size_t alignment = 256>
constexpr size_t roundToAligned(const size_t n) noexcept {
	return ((n + alignment - 1) / alignment) * alignment;
}

namespace cryptanalysislib {
	void *aligned_alloc(std::size_t alignment, std::size_t size) {
		// just to please the compiler
		if (size < alignment) {
			size = alignment;
		}

		if ((size % alignment) != 0) {
			size = ((size + alignment - 1) / alignment) * alignment;
		}

#ifdef __APPLE__
		void *ret;
		if (posix_memalign(&ret, alignment, size))
			return nullptr;

		return ret;
		// const size_t off = alignment - 1;
		// void *mem = malloc(size + off);
		// void *ptr = (void *)(((uintptr_t)mem+off) & ~ (uintptr_t)off);
		// return ptr;
#else
		return std::aligned_alloc(alignment, size);
#endif
	}
}// namespace cryptanalysislib

/// replacement for *void
/// instead of just give a pointer, all allocators do return
/// a block `blk` of memory.
struct Blk {
public:
	void *ptr;
	size_t len;

	constexpr Blk() noexcept : ptr(nullptr), len(0) {}
	constexpr Blk(void *ptr, size_t len) noexcept : ptr(ptr), len(len) {}

	/// checks whether the Blk of memory is valid or not
	/// \returns false if either ptr == nullptr or the length is zero.
	constexpr bool valid() const noexcept {
		return (ptr != nullptr) && (len != 0);
	}

	friend std::ostream &operator<<(std::ostream &os, Blk const &tc) noexcept {
		return os << tc.ptr << ":" << tc.len;
	}
};


///
struct AllocatorConfig {
	/// the base pointer to the internal data struct are always to 16bytes aligned
	constexpr static size_t base_alignment = 16;

	/// all pointers (Blks) returned do have this alignment
	constexpr static size_t alignment = 1;

	/// if set, all allocs are callocs
	constexpr static bool calloc = true;

	/// if set, after memory was free, it will be zerod
	constexpr static bool zero_after_free = true;

	/// enforce that every allocator obeys a given hint.
	constexpr static bool obey_hint = false;
} allocatorConfig;

/// concept of an allocator
template<class T>
concept Allocator = requires(T a, Blk b, size_t n) {
	{ a.allocate(n) } -> std::convertible_to<Blk>;
	a.deallocate(b);
	a.deallocateAll();
	a.owns(b);
};

/// Simple Stack Allocator
/// \tparam s  allocates `s` bytes on the stack
/// \tparam allocatorConfig
template<size_t s,
         const struct AllocatorConfig &allocatorConfig = allocatorConfig>
class StackAllocator {
	/// minimal datatype = 1 byte
	using T = uint8_t;


	/// data storage, good old stack
	alignas(allocatorConfig.base_alignment) T _d[s];
	/// pointer to the currently free=non-allocated memory
	T *_p;

public:
	constexpr StackAllocator() : _p(_d) {}

	/// \param n allocate n bytes
	/// \return
	/// 	success: a Blk of memory of size n bytes.
	/// 	error:   a Blk containing {nullptr, 0}
	constexpr Blk allocate(const size_t n) noexcept {
		const size_t n1 = roundToAligned<allocatorConfig.alignment>(n);
		if (n1 > (uintptr_t) (_d + s) - (uintptr_t) _p) {
			return {nullptr, 0};
		}

		Blk result = {_p, n};
		if constexpr (allocatorConfig.calloc) {
			cryptanalysislib::template memset<T>(_p, T(0), n);
		}

		_p += n1;
		return result;
	}

	///
	/// \param b
	/// \return
	constexpr void deallocate(Blk b) noexcept {
		// a little stupid. But the allocator is only to deallocate something
		// if it's the last element in the stack
		const size_t bla = roundToAligned<allocatorConfig.alignment>(b.len);
		if ((T *) ((size_t) b.ptr + bla) == _p) {
			if constexpr (allocatorConfig.zero_after_free) {
				cryptanalysislib::memset(_p, T(0), (size_t) _p - (size_t) _d);
			}
			_p = (T *) b.ptr;
		}
	}

	///
	/// \return
	constexpr void deallocateAll() noexcept {
		if constexpr (allocatorConfig.zero_after_free) {
			cryptanalysislib::memset(_d, T(0), _p - _d);
		}

		_p = _d;
	}

	///
	/// \param b
	/// \return
	constexpr bool owns(Blk b) noexcept {
		return b.ptr >= _d && b.ptr < _p;
	}
};

/// FreeList = makes use of Freeing memory previously
/// allocated by `parent`
///
/// \tparam Parent allocator for each node
/// \tparam s exact size of the allocator
template<Allocator Parent, const size_t size>
class FreeListAllocator {
	struct Node {
		Node *next;
	};

	Parent _parent;
	Node *_root = nullptr;

public:
	///
	/// \param n
	/// \return
	constexpr Blk allocate(const size_t n) noexcept {
		if (n == size && (_root != nullptr)) {
			Blk b = {_root, n};
			_root = _root->next;
			return b;
		}

		return _parent.allocate(n);
	}

	///
	/// \param b
	/// \return
	constexpr void deallocate(const Blk &b) {
		if (b.len != size) {
			return _parent.deallocate(b);
		}

		auto p = (Node *) b.ptr;
		p->next = _root;
		_root = p;
	}

	/// iterate through the list and deallocate through
	/// the parent allocator
	/// \return
	constexpr void deallocateAll() noexcept {
		const Node *c = _root;
		while (c != nullptr) {
			const Node *next = c->next;
			_parent.deallocate({(void *) c, size});
			c = next;
		}

		_parent.deallocateAll();
	}

	/// \param b memory blk
	/// \returns true if the memory blk is owned by this allocator
	constexpr bool owns(const Blk &b) {
		return (b.len == size) || _parent.owns(b);
	}
};

///
/// \tparam Primary
/// \tparam Fallback
template<class Primary, class Fallback>
class FallbackAllocator : private Primary, private Fallback {
public:
	constexpr Blk allocate(const size_t n) {
		Blk r = Primary::allocate(n);
		if (r.ptr == nullptr) {
			r = Fallback::allocate(n);
		}

		return r;
	}

	constexpr void deallocate(Blk b) {
		if (Primary::owns(b)) {
			Primary::deallocate(b);
		} else {
			Fallback::deallocate(b);
		}
	}

	constexpr bool owns(const Blk b) {
		return Primary::owns(b) || Fallback::owns(b);
	}
};

/// Special Allocator, which does not allocate anything but adds
/// debug information, stats and very importantly it allocates
/// a predix and a suffix around the underlying memory allocation.
/// \tparam A base allocator
/// \tparam Prefix type to allocate before the memory allocation
/// \tparam Suffix type to allocate after the memory allocation
template<Allocator A,
         class Prefix,
         class Suffix = void>
class AffixAllocator {
	constexpr static size_t compute() {
		if constexpr (std::is_void_v<Suffix>) {
			return 0;
		} else {
			return sizeof(Suffix);
		}
	}
	// sizes if bytes
	constexpr static size_t prefix_bytes = sizeof(Prefix);
	constexpr static size_t suffix_bytes = compute();
	A allocator;

	// some stats
	size_t nr_allocations = 0;
	size_t nr_deallocations = 0;
	size_t nr_own = 0;

public:
	///
	/// \param n
	/// \return
	constexpr Blk allocate(const size_t n) {
		Blk b = allocator.allocate(n + prefix_bytes + suffix_bytes);
		if (!b.valid()) {
			return b;
		}

		nr_allocations += 1;
		std::cout << "Allocated: " << b << ", nr_allocations: " << nr_allocations << std::endl;

		return {(void *) ((uintptr_t) b.ptr + prefix_bytes), n};
	}

	///
	/// \param b
	/// \return
	constexpr void deallocate(Blk b) {
		nr_deallocations += 1;
		const Blk bprime = {(void *) ((uintptr_t) b.ptr - prefix_bytes), b.len + prefix_bytes + suffix_bytes};
		std::cout << "DeAllocated: " << bprime << ", nr_deallocations: " << nr_deallocations << std::endl;
		if (allocator.owns(bprime)) {
			allocator.deallocate(bprime);
		}
	}

	///
	/// \return
	constexpr void deallocateAll() {
		allocator.deallocateAll();
	}

	///
	/// \param b
	/// \return
	constexpr bool owns(const Blk b) {
		nr_own += 1;
		const Blk bprime = {(void *) ((uintptr_t) b.ptr - prefix_bytes), b.len + prefix_bytes + suffix_bytes};
		std::cout << "owns: " << bprime << ", nr_own: " << nr_own << std::endl;
		return allocator.owns(bprime);
	}
};

///
/// \tparam SmallAllocator
/// \tparam LargeAllocator
/// \tparam Threshold
template<class SmallAllocator,
         class LargeAllocator,
         const size_t Threshold>
class Segregator {
	SmallAllocator smallAllocator;
	LargeAllocator largeAllocator;

public:
	///
	/// \param n
	/// \return
	constexpr Blk allocate(const size_t n) {
		if (n >= Threshold) {
			return largeAllocator.allocate(n);
		}

		return smallAllocator.allocate(n);
	}

	///
	/// \param b
	/// \return
	constexpr void deallocate(const Blk &b) {
		if (b.len >= Threshold) {
			return largeAllocator.deallocate(b);
		}

		return smallAllocator.deallocate(b);
	}

	///
	/// \return
	constexpr void deallocateAll() {
		largeAllocator.deallocateAll();
		smallAllocator.deallocateAll();
	}

	///
	/// \param b
	/// \return
	constexpr bool owns(const Blk &b) {
		return largeAllocator.owns(b) || smallAllocator.owns(b);
	}
};

/// simple page allocator. It can only allocate a single page
template<const size_t page_alignment = 1u << 12u,
         const size_t page_size = 1u << 12u>
class PageMallocator {
	constexpr static uintptr_t MASK = ~(page_size - 1u);

public:
	///
	/// \param n
	/// \return
	constexpr Blk allocate() noexcept {
		void *ptr = cryptanalysislib::aligned_alloc(page_alignment, page_size);
		return {ptr, ptr == nullptr ? 0 : page_size};
	}

	///
	/// \param b
	/// \return
	constexpr void deallocate(const Blk &b) noexcept {
		if (owns(b)) {
			std::free(b.ptr);
		}
	}

	///
	/// \return
	constexpr void deallocateAll() noexcept {
		/// well nothing
	}

	///
	/// \param b
	/// \return
	constexpr bool owns(const Blk &b) noexcept {
		return ((uintptr_t) b.ptr) & MASK;
	}
};


/// taken from:https://raw.githubusercontent.com/codecryptanalysis/mccl/main/mccl/core/collection.hpp
/// - modified to not use exceptions
/// - modified to use the new allocation interface
/// memory allocator pool for fixed size pages
/// do not use page_allocator before static members have been initialized
/// freeing pages after end of main (i.e. during static deconstructors) leads to undefined behaviour
template<const size_t _page_alignment = 1u << 12u,
         const size_t _page_size = 1u << 12u,
         typename PAllocator =
                 PageMallocator<_page_alignment, _page_size>>
class FreeListPageMallocator {
public:
	typedef concurrent_queue<Blk> queue_type;
	static constexpr std::size_t page_size = _page_size;
	static constexpr std::size_t page_alignment() noexcept { return _page_alignment; }
	constexpr static uintptr_t MASK = ~(page_size - 1u);

private:
	// freed pages are not returned to heap
	// but stored in queue for future page allocations instead
	// only at program end all pages are freed
	struct _static_helper {
		// concurrent queue to store freed pages
		queue_type _queue;
		std::size_t _alignment = _page_alignment;
		std::mutex _mutex;

		_static_helper() noexcept {}

		// free queue at program end
		~_static_helper() noexcept {
			Blk p;
			while (_queue.try_pop_front(p)) {
				allocator.deallocate(p);
			}

			ASSERT(_queue.size() == 0);
		}
	};
	static inline PAllocator allocator;
	static inline _static_helper _helper{};

public:
	///
	/// \param n
	/// \return
	constexpr Blk allocate() noexcept {
		return allocator.allocate();
	}

	///
	/// \param b
	/// \return
	constexpr void deallocate(const Blk &b) noexcept {
		if (owns(b)) {
			_helper._queue.push_back(b);
		}
	}

	///
	/// \return
	constexpr void deallocateAll() noexcept {
		/// well nothing
	}

	///
	/// \param b
	/// \return
	constexpr bool owns(const Blk &b) noexcept {
		return allocator.owns(b);
	}
};


/// wrapper class to expose a interface for algorithms in the std
/// \tparam T Base type to allocate
/// \tparam Allocator allocator type
template<typename T,
         typename Allocator>
class STDAllocatorWrapper {
public:
	using inner_allocator = Allocator;
	typedef STDAllocatorWrapper<T, Allocator> allocator_type;
	typedef STDAllocatorWrapper<T, Allocator> Alloc;
	typedef T value_type;
	typedef T *pointer;
	typedef const T *const_pointer;
	typedef void *void_pointer;
	typedef const void *const_void_pointer;
	typedef size_t size_type;

	inner_allocator allocator;
	///
	/// \param a
	/// \param n
	/// \return
	[[nodiscard]] static constexpr pointer allocate(allocator_type &a, size_type n) {
		Blk b = a.allocator.allocate(n);
		return (pointer) b.ptr;
	}

	/// currently ignoring the hint
	/// \param a
	/// \param n
	/// \param hint
	/// \return
	[[nodiscard]] static constexpr pointer allocate(allocator_type &a,
	                                                size_type n,
	                                                const_void_pointer hint) {
		(void) hint;
		return allocate(a, n);
	}

	// C++23 feature
	// [[nodiscard]] static constexpr std::allocation_result<pointer, size_type>
	//     allocate_at_least( Alloc& a, size_type n ) {
	//
	// }

	static constexpr void deallocate(Alloc &a, pointer p, size_type n) {
		const Blk b((void *) p, n);
		a.allocator.deAllocate(b);
	}

	/// TODO
	/// \tparam TT
	/// \tparam Args
	/// \param a
	/// \param p
	/// \param args
	/// \return
	// template<class TT, class... Args>
	// static constexpr void construct(Alloc &a, TT *p, Args &&...args) {
	// 	(void) a;
	// 	(void) p;
	// }

	/// TODO
	/// \tparam TT
	/// \param a
	/// \param p
	/// \return
	// template<class TT>
	// static constexpr void destroy(Alloc &a, TT *p) {
	// 	(void) a;
	// 	(void) p;
	// }

	static constexpr size_type max_size(const Alloc &a) noexcept {
		(void) a;
		return std::numeric_limits<size_t>::max();
	}

	static constexpr Alloc select_on_container_copy_construction(const Alloc &a) {
		(void) a;
	}
};
#endif//CRYPTANALYSISLIB_ALLOC_H
