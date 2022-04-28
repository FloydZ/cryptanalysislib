#ifndef SMALLSECRETLWE_ALLOC_H
#define SMALLSECRETLWE_ALLOC_H

struct Blk {
public:
	void *ptr;
	size_t len;
};

std::ostream& operator<< (std::ostream &out, const Blk &obj) {
	std::cout << obj.ptr << " " << obj.len << "\n";
	return out;
}

// TODO move to helper.h und template it
constexpr size_t roundToAligned(const size_t n) noexcept {
	constexpr size_t alignment = 16;
	return ((n + alignment - 1)/alignment)*alignment;
}


///
template<class T>
concept Allocator = requires(T a, Blk b, size_t n) {
	{ a.allocate(n) } -> std::convertible_to<Blk>;
	{ a.deallocate(b) } -> std::convertible_to<Blk>;
	{ a.owns(b) } -> std::convertible_to<Blk>;
};

template<size_t s>
class StackAllocator {
	uint8_t _d[s];
	uint8_t *_p;

public:
	StackAllocator() : _p(_d) {}

	constexpr Blk allocate(const size_t n) noexcept {
		auto n1 = roundToAligned(n);
		if (n1 > (_d + s) - _p) {
			return {nullptr, 0};
		}

		Blk result = {_p, n};
		_p += n1;
		return result;
	}

	constexpr void deallocate(Blk b) noexcept {
		// a little stupid. But the allocator is only to deallocate something
		// if its the last element in the stack
		if ( (uint8_t *)((size_t)b.ptr + roundToAligned(b.len)) == _p) {
			_p = (uint8_t *)b.ptr;
		}
	}

	constexpr bool owns(Blk b) noexcept {
		return b.ptr >= _d && b.ptr < _d + s;
	}

	constexpr void deallocateAll() noexcept {
		_p = _d;
	}
};

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

template<class Parent, class Prefix, class Suffix = void>
class AffixAllocator {
	// TODO optional prefix and suffix, construct/destroy appropriate, debug, stats, info
};

template<class SmallAllocator, class LargeAllocator, const size_t Threshold>
class Segregator {
	constexpr Blk allocate(const size_t n) {
		if (n >= Threshold)
			return LargeAllocator::allocate(n);

		return SmallAllocator::allocate(n);
	}

	constexpr void deallocate(Blk b) {
		if (b.len>= Threshold)
			return LargeAllocator::deallocate(b);

		return SmallAllocator::deallocate(b);
	}

	constexpr bool owns(const Blk b) {
		return LargeAllocator::owns(b) || SmallAllocator::owns(b);
	}
};

#endif //SMALLSECRETLWE_ALLOC_H
