#ifndef CRYPTANALYSISLIB_ALLOC_H
#define CRYPTANALYSISLIB_ALLOC_H

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <mutex>

#include "container/queue.h"
#include "helper.h"
#include "memory/memory.h"

/// Rounds up a value to the next multiple of alignment
/// \tparam alignment[in]: in bytes
/// \param n[in]: input to align up to a multiple of `alignment`
/// \return the up aligned value
template<const size_t alignment = 256>
constexpr size_t roundToAligned(const size_t n) noexcept {
	return ((n + alignment - 1) / alignment) * alignment;
}

namespace cryptanalysislib {

	/// Allocates memory with specified alignment
	/// \param alignment[in]: number of bytes to align the pointer to
	/// \param size[in]: number of bytes to allocate
	/// \return pointer to the aligned data or nullptr
	static inline void *aligned_alloc(const std::size_t alignment,
	                                  const std::size_t size) noexcept {
        void *p = malloc(size + sizeof(void *) + alignment - 1);
        if (!p) [[unlikely]] {
            return p;
	    }
        void **ap = (void **)(((uint64_t)p + sizeof(void *) + alignment - 1) & ~(alignment - 1));
        ap[-1] = p;
        return ap;
	}

    /// Frees memory allocated by aligned_alloc
    /// NOTE: will fail if the ptr was not returned by `aligned_alloc`
    /// \param p[in]: pointer to free
    static inline void aligned_free(void *p) noexcept {
        if (nullptr != p) [[likely]] { 
            free(((void **)p)[-1]);
    	}
    }
}// namespace cryptanalysislib

#ifdef __unix__
#include <errno.h>
#include <fcntl.h>
#include <linux/kernel-page-flags.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

// See <https://www.kernel.org/doc/Documentation/vm/pagemap.txt> for
// format which these bitmasks refer to
#define PAGEMAP_PRESENT(ent) (((ent) & (1ull << 63)) != 0)
#define PAGEMAP_PFN(ent) ((ent) & ((1ull << 55) - 1))

// Checks if the page pointed at by `ptr` is huge. Assumes that `ptr` has already
// been allocated.
static void check_huge_page(void *ptr) {
	const uint64_t CUSTOM_PAGE_SIZE = 1u<<13; // TODO dont know if this is correct
	int pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
	if (pagemap_fd < 0) {
		std::cout << "could not open /proc/self/pagemap: " << strerror(errno) << "\n";
	}
	int kpageflags_fd = open("/proc/kpageflags", O_RDONLY);
	if (kpageflags_fd < 0) {
		std::cout << "could not open /proc/kpageflags: " << strerror(errno) << "\n";
	}

	// each entry is 8 bytes long
	uint64_t ent;
	if (pread(pagemap_fd, &ent, sizeof(ent), ((uintptr_t) ptr) / CUSTOM_PAGE_SIZE * 8) != sizeof(ent)) {
		std::cout << "could not read from pagemap\n";
	}

	if (!PAGEMAP_PRESENT(ent)) {
		std::cout << "page not present in /proc/self/pagemap, did you allocate it?\n";
	}
	if (!PAGEMAP_PFN(ent)) {
		std::cout << "page frame number not present, run this program as root\n";
	}

	uint64_t flags;
	if (pread(kpageflags_fd, &flags, sizeof(flags), PAGEMAP_PFN(ent) << 3) != sizeof(flags)) {
		std::cout << "could not read from kpageflags\n";
	}

	if (!(flags & (1ull << KPF_THP))) {
		std::cout << "could not allocate huge page\n";
	}

	if (close(pagemap_fd) < 0) {
		std::cout << "could not close /proc/self/pagemap: " << strerror(errno) << "\n";
	}
	if (close(kpageflags_fd) < 0) {
		std::cout << "could not close /proc/kpageflags: " << strerror(errno) << "\n";
	}
}

/// Tries to allocate a huge page
/// \param size[in]: number of bytes to allocate
/// \return pointer to the allocated huge page or nullptr
static 
void *cryptanalysislib_hugepage_malloc(const size_t size) {
	const uint64_t HPAGE_SIZE = 1u<<13; // TODO dont know if this is correct
	const size_t nr_pages = (size + HPAGE_SIZE - 1) / HPAGE_SIZE;
	const size_t alloc_size = nr_pages * HPAGE_SIZE;
	void *ret = cryptanalysislib::aligned_alloc(HPAGE_SIZE, alloc_size);
	if (ret == nullptr) {
		std::cout << "error alloc\n";
		return nullptr;
	}

	madvise(ret, size, MADV_HUGEPAGE);

	size_t buf = (size_t) ret;
	for (size_t end = buf + size; buf < end; buf += HPAGE_SIZE) {
		// allocate page
		memset((void *) buf, 0, 1);
		// check the page is indeed huge
		check_huge_page((void *) buf);
	}

	return ret;
}
#endif

/// Replacement for *void
/// Instead of just giving a pointer, all allocators return
/// a block `blk` of memory.
struct Blk {
public:
	void *ptr;
	size_t len;

	constexpr Blk() noexcept : ptr(nullptr), len(0) {}
	constexpr Blk(void *ptr, size_t len) noexcept : ptr(ptr), len(len) {}

	/// Checks whether the Blk of memory is valid or not
	/// \return false if either ptr == nullptr or the length is zero
	constexpr inline bool valid() const noexcept {
		return (ptr != nullptr) && (len != 0);
	}

	/// Stream output operator for Blk to simplify debugging
	/// \param os[in]: output stream
	/// \param tc[in]: Blk object to output
	/// \return the modified output stream
	friend std::ostream &operator<<(std::ostream &os,
	                                Blk const &tc) noexcept {
		return os << tc.ptr << ":" << tc.len;
	}
};

/// Configuration settings for allocators
struct AllocatorConfig : public AlignmentConfig {
	/// the base pointer to the internal data struct are always to 16bytes aligned
	constexpr static size_t base_alignment = 16;

	/// all pointers (Blks) returned do have this alignment
	// constexpr static size_t alignment = 1;

	/// if set, all allocations are zero allocations
	constexpr static bool calloc = true;

	/// if set, after memory was free, it will be overwritten with zero
	constexpr static bool zero_after_free = true;

	/// enforce that every allocator obeys a given hint.
	constexpr static bool obey_hint = false;
};
constexpr static AllocatorConfig allocatorConfig;

/// Concept definition for an allocator
/// Requires allocate, deallocate, deallocateAll, and owns methods
template<class T>
concept Allocator = requires(T a, Blk b, size_t n) {
	{ a.allocate(n) } -> std::convertible_to<Blk>;
	a.deallocate(b);
	a.deallocateAll();
	a.owns(b);
};

/// Simple Stack Allocator
/// \tparam s[in]: allocates `s` bytes on the stack
/// \tparam allocatorConfig[in]: configuration for the allocator
template<const size_t s,
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

	/// Allocates memory from the stack allocator
	/// \param n[in]: number of bytes to allocate
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

	/// Deallocates a block of memory, but only if it's the last allocated block
	/// \param b[in]: Block to deallocate
	constexpr void deallocate(const Blk b) noexcept {
		// a little stupid. But the allocator is only to deallocate something
		// if it's the last element in the stack
		const size_t bla = roundToAligned<allocatorConfig.alignment>(b.len);
		if ((T *) ((size_t) b.ptr + bla) == _p) {
			if constexpr (allocatorConfig.zero_after_free) {
				cryptanalysislib::memset(_p, T(0), ((uintptr_t) _p - (uintptr_t) _d)/sizeof(T));
			}
			_p = (T *) b.ptr;
		}
	}

	/// Deallocates all allocations and resets the allocator to its initial state
	constexpr void deallocateAll() noexcept {
		if constexpr (allocatorConfig.zero_after_free) {
			cryptanalysislib::memset(_d, T(0), ((uintptr_t)_p - (uintptr_t)_d)/sizeof(T));
		}

		_p = _d;
	}

	/// Checks if a memory block is owned by this allocator
	/// \param b[in]: Block to check ownership for
	/// \return true if the block is within the allocator's memory range
	constexpr bool owns(Blk b) noexcept {
		return b.ptr >= _d && b.ptr < _p;
	}
};

/// FreeList = makes use of Freeing memory previously
/// allocated by `parent`
///
/// \tparam Parent[in]: allocator for each node
/// \tparam size[in]: exact size of the allocator
template<Allocator Parent, const size_t size>
class FreeListAllocator {
	struct Node {
		Node *next;
	};

	Parent _parent;
	Node *_root = nullptr;

public:
	/// Allocates a block of memory of specified size
	/// \param n[in]: Number of bytes to allocate
	/// \return A valid block if allocation succeeded, or {nullptr, 0} otherwise
	constexpr Blk allocate(const size_t n) noexcept {
		if (n == size && (_root != nullptr)) {
			Blk b = {_root, n};
			_root = _root->next;
			return b;
		}

		return _parent.allocate(n);
	}

	/// Deallocates a memory block, returning it to the free list or parent allocator
	/// \param b[in]: memory block to deallocate
	constexpr void deallocate(const Blk &b) {
		if (b.len != size) {
			return _parent.deallocate(b);
		}

		auto p = (Node *) b.ptr;
		p->next = _root;
		_root = p;
	}

	/// Iterates through the free list and deallocates all nodes through
	/// the parent allocator
	constexpr void deallocateAll() noexcept {
		const Node *c = _root;
		while (c != nullptr) {
			const Node *next = c->next;
			_parent.deallocate({(void *) c, size});
			c = next;
		}

		_parent.deallocateAll();
	}

	/// Checks if this allocator owns the given memory block
	/// \param b[in]: memory block to check
	/// \return true if the memory block is owned by this allocator
	constexpr bool owns(const Blk &b) {
		return (b.len == size) || _parent.owns(b);
	}
};

/// Fallback allocator that tries Primary first, then Fallback
/// \tparam Primary[in]: primary allocator to try first
/// \tparam Fallback[in]: fallback allocator to use if Primary fails
template<class Primary, class Fallback>
class FallbackAllocator : private Primary, private Fallback {
public:
	/// Allocates memory using Primary allocator, falls back to Fallback if Primary fails
	/// \param n[in]: number of bytes to allocate
	/// \return allocated memory block
	constexpr Blk allocate(const size_t n) {
		Blk r = Primary::allocate(n);
		if (r.ptr == nullptr) {
			r = Fallback::allocate(n);
		}

		return r;
	}

	/// Deallocates memory using the appropriate allocator
	/// \param b[in]: memory block to deallocate
	constexpr void deallocate(Blk b) {
		if (Primary::owns(b)) {
			Primary::deallocate(b);
		} else {
			Fallback::deallocate(b);
		}
	}

	/// Checks if this allocator owns the given memory block
	/// \param b[in]: memory block to check
	/// \return true if either Primary or Fallback allocator owns the block
	constexpr bool owns(const Blk b) {
		return Primary::owns(b) || Fallback::owns(b);
	}
};

/// Special Allocator, which does not allocate anything but adds
/// debug information, stats and very importantly it allocates
/// a predix and a suffix around the underlying memory allocation.
/// \tparam A[in]: base allocator
/// \tparam Prefix[in]: type to allocate before the memory allocation
/// \tparam Suffix[in]: type to allocate after the memory allocation
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
	/// Allocates memory with prefix and suffix regions
	/// \param n[in]: number of bytes to allocate
	/// \return memory block with adjusted pointer and size
	constexpr Blk allocate(const size_t n) {
		Blk b = allocator.allocate(n + prefix_bytes + suffix_bytes);
		if (!b.valid()) {
			return b;
		}

		nr_allocations += 1;
		std::cout << "Allocated: " << b << ", nr_allocations: " << nr_allocations << std::endl;

		return {(void *) ((uintptr_t) b.ptr + prefix_bytes), n};
	}

	/// Deallocates memory, accounting for prefix and suffix regions
	/// \param b[in]: memory block to deallocate
	constexpr void deallocate(Blk b) {
		nr_deallocations += 1;
		const Blk bprime = {(void *) ((uintptr_t) b.ptr - prefix_bytes), b.len + prefix_bytes + suffix_bytes};
		std::cout << "DeAllocated: " << bprime << ", nr_deallocations: " << nr_deallocations << std::endl;
		if (allocator.owns(bprime)) {
			allocator.deallocate(bprime);
		}
	}

	/// Deallocates all memory from the underlying allocator
	constexpr void deallocateAll() {
		allocator.deallocateAll();
	}

	/// Checks if this allocator owns the given memory block
	/// \param b[in]: memory block to check
	/// \return true if the allocator owns the block
	constexpr bool owns(const Blk b) {
		nr_own += 1;
		const Blk bprime = {(void *) ((uintptr_t) b.ptr - prefix_bytes), b.len + prefix_bytes + suffix_bytes};
		std::cout << "owns: " << bprime << ", nr_own: " << nr_own << std::endl;
		return allocator.owns(bprime);
	}
};

/// Segregator allocator that uses different allocators based on allocation size
/// \tparam SmallAllocator[in]: allocator for small allocations
/// \tparam LargeAllocator[in]: allocator for large allocations
/// \tparam Threshold[in]: size threshold to determine small vs large
template<class SmallAllocator,
         class LargeAllocator,
         const size_t Threshold>
class Segregator {
	SmallAllocator smallAllocator;
	LargeAllocator largeAllocator;

public:
	/// Allocates memory using the appropriate allocator based on size
	/// \param n[in]: number of bytes to allocate
	/// \return allocated memory block
	constexpr Blk allocate(const size_t n) {
		if (n >= Threshold) {
			return largeAllocator.allocate(n);
		}

		return smallAllocator.allocate(n);
	}

	/// Deallocates memory using the appropriate allocator based on size
	/// \param b[in]: memory block to deallocate
	constexpr void deallocate(const Blk &b) {
		if (b.len >= Threshold) {
			return largeAllocator.deallocate(b);
		}

		return smallAllocator.deallocate(b);
	}

	/// Deallocates all memory from both allocators
	constexpr void deallocateAll() {
		largeAllocator.deallocateAll();
		smallAllocator.deallocateAll();
	}

	/// Checks if this allocator owns the given memory block
	/// \param b[in]: memory block to check
	/// \return true if either allocator owns the block
	constexpr bool owns(const Blk &b) {
		return largeAllocator.owns(b) || smallAllocator.owns(b);
	}
};

/// Simple page allocator that can only allocate a single page
/// \tparam page_alignment[in]: alignment of the page in bytes
/// \tparam page_size[in]: size of the page in bytes
template<const size_t page_alignment = 1u << 12u,
         const size_t page_size = 1u << 12u>
class PageMallocator {
	constexpr static uintptr_t MASK = ~(page_size - 1u);

public:
	/// Allocates a single page of memory
	/// \return memory block containing a page
	constexpr Blk allocate() noexcept {
		void *ptr = cryptanalysislib::aligned_alloc(page_alignment, page_size);
		return {ptr, ptr == nullptr ? 0 : page_size};
	}

	/// Deallocates a page of memory
	/// \param b[in]: memory block to deallocate
	constexpr void deallocate(const Blk &b) noexcept {
		if (owns(b)) {
			cryptanalysislib::aligned_free(b.ptr);
			//std::free(b.ptr);
		}
	}

	/// Deallocates all memory (does nothing for this allocator)
	constexpr void deallocateAll() noexcept {
		/// well nothing
	}

	/// Checks if this allocator owns the given memory block
	/// \param b[in]: memory block to check
	/// \return true if the block is a page owned by this allocator
	constexpr bool owns(const Blk &b) noexcept {
		return ((uintptr_t) b.ptr) & MASK;
	}
};

/// Memory allocator pool for fixed size pages
/// Taken from: https://raw.githubusercontent.com/codecryptanalysis/mccl/main/mccl/core/collection.hpp
/// - Modified to not use exceptions
/// - Modified to use the new allocation interface
/// Do not use page_allocator before static members have been initialized
/// Freeing pages after end of main (i.e. during static deconstructors) leads to undefined behaviour
/// \tparam _page_alignment[in]: alignment of the page in bytes
/// \tparam _page_size[in]: size of the page in bytes
/// \tparam PAllocator[in]: underlying page allocator
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
		std::mutex _mutex; // TODO make template argument

		_static_helper() noexcept {}

		// free queue at program end
		~_static_helper() noexcept {
			Blk p;
			while (_queue.try_pop_front(p)) {
				allocator.deallocate(p);
			}

			assert(_queue.size() == 0);
		}
	};
	static inline PAllocator allocator;
	static inline _static_helper _helper{};

public:
	/// Allocates a memory page from the page allocator
	/// \return memory block containing a page
	constexpr Blk allocate() noexcept {
		return allocator.allocate();
	}

	/// Adds the page to the free list queue instead of deallocating it
	/// \param b[in]: memory block to deallocate
	constexpr void deallocate(const Blk &b) noexcept {
		if (owns(b)) {
			_helper._queue.push_back(b);
		}
	}

	/// Deallocates all memory (does nothing for this allocator)
	constexpr void deallocateAll() noexcept {
		/// well nothing
	}

	/// Checks if this allocator owns the given memory block
	/// \param b[in]: memory block to check
	/// \return true if the block is owned by this allocator
	constexpr bool owns(const Blk &b) noexcept {
		return allocator.owns(b);
	}
};

/// Wrapper class to expose an interface for algorithms in the STL
/// \tparam T[in]: Base type to allocate
/// \tparam Allocator[in]: allocator type
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

	static inline inner_allocator sallocator{};
	inner_allocator allocator;

	/// Allocates memory for n elements
	/// \param n[in]: number of elements to allocate
	/// \return pointer to allocated memory or nullptr
	[[nodiscard]] static constexpr inline pointer allocate(const size_type n) noexcept {
		Blk b = sallocator.allocate(n);
		return (pointer) b.ptr;
	}

	/// Allocates memory for n elements using the provided allocator
	/// \param a[in]: allocator to use
	/// \param n[in]: number of elements to allocate
	/// \return pointer to allocated memory or nullptr
	[[nodiscard]] static constexpr inline pointer allocate(allocator_type &a,
	                                                       const size_type n) noexcept {
		Blk b = a.allocator.allocate(n);
		return (pointer) b.ptr;
	}

	/// Allocates memory for n elements with a hint (currently ignored)
	/// \param a[in]: allocator to use
	/// \param n[in]: number of elements to allocate
	/// \param hint[in]: allocation hint (ignored)
	/// \return pointer to allocated memory or nullptr
	[[nodiscard]] static constexpr inline pointer allocate(allocator_type &a,
	                                                const size_type n,
	                                                const const_void_pointer hint) noexcept {
		(void) hint;
		return allocate(a, n);
	}

	// C++23 feature
	// [[nodiscard]] static constexpr std::allocation_result<pointer, size_type>
	//     allocate_at_least( Alloc& a, size_type n ) {
	//
	// }

	/// Deallocates memory
	/// \param p[in]: pointer to memory to deallocate
	/// \param n[in]: number of elements
	static constexpr inline void deallocate(const pointer p,
											const size_type n) noexcept {
		const Blk b((void *) p, n);
		sallocator.deallocate(b);
	}

	/// Deallocates memory using the provided allocator
	/// \param a[in]: allocator to use
	/// \param p[in]: pointer to memory to deallocate
	/// \param n[in]: number of elements
	static constexpr inline void deallocate(allocator_type &a,
	                                        const pointer p,
	                                        const size_type n) noexcept {
		const Blk b((void *) p, n);
		a.allocator.deallocate(b);
	}

	/// removed in C++20
	// template<class TT, class... Args>
	// static constexpr void construct(Alloc &a, TT *p, Args &&...args) {
	// 	(void) a;
	// 	(void) p;
	// }

	/// removed in C++20
	// template<class TT>
	// static constexpr void destroy(Alloc &a, TT *p) {
	// 	(void) a;
	// 	(void) p;
	// }

	// not really implemented, as the base allocators are not that good
	/// removed in C++20
	//static constexpr inline size_type max_size(const Alloc &a) noexcept {
	//	(void) a;
	//	return std::numeric_limits<size_t>::max();
	//}
};

/// C++ wrapper around `aligned_alloc` and `aligned_free`
/// \tparam T[in]: type to allocate
/// \tparam alignment[in]: in bytes
template<typename T,
		 const size_t alignment = 1024>
class AlignmentMallocator {
public:
	typedef AlignmentMallocator<T, alignment> allocator_type;
	typedef AlignmentMallocator<T, alignment> Alloc;
	typedef T value_type;
	typedef T *pointer;
	typedef const T *const_pointer;
	typedef void *void_pointer;
	typedef const void *const_void_pointer;
	typedef size_t size_type;

	/// Allocates aligned memory
	/// \param n[in]: number of bytes to allocate
	/// \return pointer to aligned memory or nullptr
	[[nodiscard]] static constexpr inline pointer allocate(const size_type n) noexcept {
		return static_cast<pointer>(cryptanalysislib::aligned_alloc(alignment, n));
	}

	/// Deallocates aligned memory
	/// \param p[in]: pointer to memory to deallocate
	/// \param n[in]: number of bytes (unused)
	static constexpr inline void deallocate(const pointer p,
											const size_type n) noexcept {
        (void) n;
        cryptanalysislib::aligned_free(p);
	}
};

#ifdef USE_TRACY

template<typename T,
		 const size_t alignment = 1024>
class TracyAllocator {
public:
	typedef TracyAllocator<T, alignment> allocator_type;
	typedef TracyAllocator<T, alignment> Alloc;
	typedef T value_type;
	typedef T *pointer;
	typedef const T *const_pointer;
	typedef void *void_pointer;
	typedef const void *const_void_pointer;
	typedef size_t size_type;

	const char *pool_name = "tracy_allocator";

	/// Allocates memory with Tracy profiling
	/// \param n[in]: number of elements to allocate
	/// \return pointer to allocated memory or nullptr
	[[nodiscard]] static constexpr inline pointer allocate(const size_type n) noexcept {
		T *p = nullptr;
		TracyCAllocN(p, sizeof(T) * n, pool_name);
		return p;
	}

	/// Deallocates memory with Tracy profiling
	/// \param p[in]: pointer to memory to deallocate
	/// \param n[in]: number of elements
	static constexpr inline void deallocate(const pointer p,
											const size_type n) noexcept {
		TracyCFreeN(p, sizeof(T) * n);
	}
};
#endif

namespace cryptanalysislib {
	// define a standard allocator
    template<typename T>
	using allocator = std::allocator<T>; // PageMallocator<1u<<12u, 1u<<12u>;

	template <typename T>
	using alignment_allocator = AlignmentMallocator<T>;
}
#endif //CRYPTANALYSISLIB_ALLOC_H