#ifndef SMALLSECRETLWE_ALLOC_H
#define SMALLSECRETLWE_ALLOC_H

#include <iostream>
#include "mem/memset.h"

/// replacement for *void
/// instead of just give a pointer, all allocators do return
/// a block `blk` of memory.
struct Blk {
public:
	void *ptr;
	size_t len;
};

/// helper function. That's only useful for debugging
std::ostream& operator<< (std::ostream &out, const Blk &obj) {
	std::cout << obj.ptr << " " << obj.len << "\n";
	return out;
}

// TODO move to helper.h und template it
constexpr size_t roundToAligned(const size_t n) noexcept {
	constexpr size_t alignment = 16;
	return ((n + alignment - 1)/alignment)*alignment;
}


struct AllocatorConfig {
	/// the base pointer to the internal data struct are always to 16bytes aligned
	constexpr static size_t base_alignment = 16;

	/// all pointers (Blks) returned do have this alignment
	constexpr static size_t alignment = 1;

	/// if set, all allocs are callocs
	constexpr static bool calloc = true;

	/// if set, after memory was free, it will be zerod
	constexpr static bool zero_after_free = true;
} allocatorConfig;

/// concept of an allocator
template<class T>
concept Allocator = requires(T a, Blk b, size_t n) {
	{ a.allocate(n) }   -> std::convertible_to<Blk>;
	a.deallocate(b);
	a.deallocateAll();
	a.owns(b);
};

/// Simple Stack Allocator
/// allocates `s` bytes on the stack
template<size_t s, const struct AllocatorConfig &allocatorConfig=allocatorConfig>
class StackAllocator {
	using T = uint8_t;

	alignas(allocatorConfig.alignment) T _d[s];
	T *_p;

public:
	StackAllocator() : _p(_d) {}

	///
	/// \param n
	/// \return
	constexpr Blk allocate(const size_t n) noexcept {
		auto n1 = roundToAligned(n);
		if (n1 > (_d + s) - _p) {
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
		if ( (T *)((size_t)b.ptr + roundToAligned(b.len)) == _p) {
			if constexpr (allocatorConfig.zero_after_free) {
				cryptanalysislib::memset(_p, T(0), (size_t)_p - (size_t)_d);
			}
			_p = (T *)b.ptr;
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

///
/// \tparam Parent
/// \tparam s
template <class Parent, const size_t s>
class FreeListAllocator {
	struct Node {
		Node *next;
	};

	Parent _parent;
	Node _root;

public:
	constexpr Blk allocate(const size_t n) noexcept {
		if (n == s && _root) {
			Blk b = {_root, n};
			_root = *_root.next;
			return b;
		}

		return _parent.allocate(n);
	}

	constexpr void deallocate(Blk b) {
		if (b.len != s)
			return _parent.deallocate(b);

		auto p = (Node *)b.ptr;
		p->next = _root;
		_root *p;
	}

	constexpr bool owns(Blk b) {
		return b.len == s || _parent.owns(b);
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
		if (r.ptr == nullptr)
			r = Fallback::allocate(n);

		return r;
	}

	constexpr void deallocate(Blk b) {
		if (Primary::owns(b))
			Primary::deallocate(b);
		else
			Fallback::deallocate(b);
	}

	constexpr bool owns(const Blk b) {
		return Primary::owns(b) || Fallback::owns(b);
	}
};

///
/// \tparam Parent
/// \tparam Prefix
/// \tparam Suffix
template<class Parent, class Prefix, class Suffix = void>
class AffixAllocator {
	// TODO optional prefix and suffix, construct/destroy appropriate, debug, stats, info
};

///
/// \tparam SmallAllocator
/// \tparam LargeAllocator
/// \tparam Threshold
template<class SmallAllocator, class LargeAllocator, const size_t Threshold>
class Segregator {
	constexpr Blk allocate(const size_t n) {
		if (n >= Threshold) {
			return LargeAllocator::allocate(n);
		}

		return SmallAllocator::allocate(n);
	}

	constexpr void deallocate(Blk b) {
		if (b.len>= Threshold) {
			return LargeAllocator::deallocate(b);
		}

		return SmallAllocator::deallocate(b);
	}

	constexpr bool owns(const Blk b) {
		return LargeAllocator::owns(b) || SmallAllocator::owns(b);
	}
};

#endif //SMALLSECRETLWE_ALLOC_H
