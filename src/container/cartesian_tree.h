#ifndef CRYPTANALYSISLIB_CONTAINER_CARTESIAN_TREE_H
#define CRYPTANALYSISLIB_CONTAINER_CARTESIAN_TREE_H

#include <cstddef>
#include <cstdint>
#include <utility>

namespace cryptanalysislib {

/// Cartesian tree / treap: a binary search tree on the keys which is a heap
/// on random priorities, i.e. balanced in expectation. Supports insert,
/// erase, contains and order statistics (`kth`) in expected O(log n).
/// \tparam T key type (totally ordered via `<`)
template<typename T = int>
class cartesian_tree {
	struct node {
		T x;          // key (search tree order)
		uint64_t y;   // priority (max heap order)
		size_t sz;    // size of the subtree
		node *l, *r;
		constexpr node(const T &_x, const uint64_t _y) noexcept :
		    x(_x), y(_y), sz(1), l(nullptr), r(nullptr) {}
	};

	node *root = nullptr;
	/// xorshift64 state for the priorities
	uint64_t state = 0x9e3779b97f4a7c15ull;

	constexpr uint64_t next_priority() noexcept {
		state ^= state << 13u;
		state ^= state >> 7u;
		state ^= state << 17u;
		return state;
	}

	constexpr static size_t tsize(const node *t) noexcept {
		return t ? t->sz : 0;
	}

	constexpr static void augment(node *t) noexcept {
		t->sz = 1 + tsize(t->l) + tsize(t->r);
	}

	/// splits `t` into the keys < x and the keys >= x
	constexpr static std::pair<node *, node *> split(node *t, const T &x) noexcept {
		if (!t) { return {nullptr, nullptr}; }
		if (t->x < x) {
			const std::pair<node *, node *> res = split(t->r, x);
			t->r = res.first;
			augment(t);
			return {t, res.second};
		}
		const std::pair<node *, node *> res = split(t->l, x);
		t->l = res.second;
		augment(t);
		return {res.first, t};
	}

	/// merges two trees, all keys of `l` must be smaller than all of `r`
	constexpr static node *merge(node *l, node *r) noexcept {
		if (!l) { return r; }
		if (!r) { return l; }
		if (l->y > r->y) {
			l->r = merge(l->r, r);
			augment(l);
			return l;
		}
		r->l = merge(l, r->l);
		augment(r);
		return r;
	}

	constexpr static const node *find(const node *t, const T &x) noexcept {
		while (t) {
			if (x < t->x) {
				t = t->l;
			} else if (t->x < x) {
				t = t->r;
			} else {
				return t;
			}
		}
		return nullptr;
	}

	constexpr static node *erase(node *t, const T &x, bool &removed) noexcept {
		if (!t) { return nullptr; }
		if (t->x < x) {
			t->r = erase(t->r, x, removed);
		} else if (x < t->x) {
			t->l = erase(t->l, x, removed);
		} else {
			node *old = t;
			t = merge(t->l, t->r);
			delete old;
			removed = true;
		}
		if (t) { augment(t); }
		return t;
	}

	constexpr static void destroy(node *t) noexcept {
		if (!t) { return; }
		destroy(t->l);
		destroy(t->r);
		delete t;
	}

public:
	constexpr cartesian_tree() noexcept = default;
	cartesian_tree(const cartesian_tree &) = delete;
	cartesian_tree &operator=(const cartesian_tree &) = delete;
	constexpr ~cartesian_tree() noexcept { destroy(root); }

	/// \return number of keys
	[[nodiscard]] constexpr size_t size() const noexcept { return tsize(root); }

	/// \return true if there are no keys
	[[nodiscard]] constexpr bool empty() const noexcept { return root == nullptr; }

	/// \return true if `x` is in the tree
	[[nodiscard]] constexpr bool contains(const T &x) const noexcept {
		return find(root, x) != nullptr;
	}

	/// inserts `x` (with a random priority)
	/// \return false if `x` was already in the tree
	constexpr bool insert(const T &x) noexcept {
		if (find(root, x) != nullptr) { return false; }
		const std::pair<node *, node *> res = split(root, x);
		root = merge(res.first, merge(new node(x, next_priority()), res.second));
		return true;
	}

	/// removes `x`
	/// \return false if `x` was not in the tree
	constexpr bool erase(const T &x) noexcept {
		bool removed = false;
		root = erase(root, x, removed);
		return removed;
	}

	/// \param k[in]: rank, < size()
	/// \return the k-th smallest key (k = 0: the minimum)
	[[nodiscard]] constexpr const T &kth(size_t k) const noexcept {
		const node *t = root;
		while (true) {
			const size_t ls = tsize(t->l);
			if (k < ls) {
				t = t->l;
			} else if (k == ls) {
				return t->x;
			} else {
				k -= ls + 1;
				t = t->r;
			}
		}
	}

	/// \return number of keys < x
	[[nodiscard]] constexpr size_t rank(const T &x) const noexcept {
		size_t r = 0;
		const node *t = root;
		while (t) {
			if (t->x < x) {
				r += tsize(t->l) + 1;
				t = t->r;
			} else {
				t = t->l;
			}
		}
		return r;
	}
};
} // end namespace cryptanalysislib
#endif
