#ifndef CRYPTANALYSISLIB_AA_TREE_H
#define CRYPTANALYSISLIB_AA_TREE_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

namespace cryptanalysislib {
/// AA tree set (Arne Andersson, "Balanced search trees made simple", 1993):
/// a balanced binary search tree (a simplified red-black tree) storing
/// every value at most once. insert/erase/contains in O(log n).
/// Port of https://www.nayuki.io/res/aa-tree-set/aatreeset.rs
/// Invariants (checked by `check()`):
/// 	- leaves have level 1,
/// 	- level(left child) == level - 1,
/// 	- level(right child) is level or level - 1,
/// 	- level(right grandchild) < level.
/// \tparam T value type (copyable)
/// \tparam Less strict weak ordering
/// \tparam Allocator allocator, rebound to the node type
template<typename T,
         typename Less = std::less<T>,
         class Allocator = std::allocator<T>>
class AATreeSet {
private:
	struct Node {
		T value;
		Node *left = nullptr;
		Node *right = nullptr;
		uint32_t level = 1;
	};

	using NodeAllocator = typename std::allocator_traits<Allocator>::template rebind_alloc<Node>;
	using NodeTraits = std::allocator_traits<NodeAllocator>;

	Node *root = nullptr;
	size_t size_ = 0;
	NodeAllocator alloc{};
	Less less{};

	[[nodiscard]] static constexpr uint32_t level(const Node *n) noexcept {
		return n ? n->level : 0u;
	}

	Node *make_node(const T &v) noexcept {
		Node *n = NodeTraits::allocate(alloc, 1);
		NodeTraits::construct(alloc, n, Node{v, nullptr, nullptr, 1u});
		return n;
	}

	void destroy_node(Node *n) noexcept {
		NodeTraits::destroy(alloc, n);
		NodeTraits::deallocate(alloc, n, 1);
	}

	void destroy_tree(Node *n) noexcept {
		if (!n) { return; }
		destroy_tree(n->left);
		destroy_tree(n->right);
		destroy_node(n);
	}

	///       |          |
	///   A - B    ->    A - B
	///  / \   \        /   / \
	/// 0   1   2      0   1   2
	/// removes a left horizontal link
	[[nodiscard]] static constexpr Node *skew(Node *n) noexcept {
		if (n && n->left && (n->left->level == n->level)) {
			Node *l = n->left;
			n->left = l->right;
			l->right = n;
			return l;
		}
		return n;
	}

	///   |                      |
	///   |                    - B -
	///   |                   /     \
	///   A - B - C    ->    A       C
	///  /   /   / \        / \     / \
	/// 0   1   2   3      0   1   2   3
	/// removes two consecutive right horizontal links
	[[nodiscard]] static constexpr Node *split(Node *n) noexcept {
		if (n && n->right && n->right->right && (n->right->right->level == n->level)) {
			Node *r = n->right;
			n->right = r->left;
			r->left = n;
			r->level += 1;
			return r;
		}
		return n;
	}

	Node *insert(Node *n, const T &v, bool &inserted) noexcept {
		if (!n) {
			inserted = true;
			return make_node(v);
		}
		if (less(v, n->value)) {
			n->left = insert(n->left, v, inserted);
		} else if (less(n->value, v)) {
			n->right = insert(n->right, v, inserted);
		} else {
			return n; // already present
		}
		return split(skew(n));
	}

	Node *erase(Node *n, const T &v, bool &removed) noexcept {
		if (!n) { return nullptr; }
		if (less(v, n->value)) {
			n->left = erase(n->left, v, removed);
		} else if (less(n->value, v)) {
			n->right = erase(n->right, v, removed);
		} else {
			removed = true;
			if (!n->left && !n->right) {
				destroy_node(n);
				return nullptr;
			}
			bool dummy = false;
			if (!n->left) {
				// replace by the successor
				const Node *s = n->right;
				while (s->left) { s = s->left; }
				n->value = s->value;
				n->right = erase(n->right, n->value, dummy);
			} else {
				// replace by the predecessor
				const Node *p = n->left;
				while (p->right) { p = p->right; }
				n->value = p->value;
				n->left = erase(n->left, n->value, dummy);
			}
		}

		// rebalance: decrease the level, then skew and split the path
		const uint32_t lmin = level(n->left) < level(n->right) ? level(n->left) : level(n->right);
		const uint32_t should = lmin + 1u;
		if (should < n->level) {
			n->level = should;
			if (n->right && (should < n->right->level)) {
				n->right->level = should;
			}
		}
		n = skew(n);
		n->right = skew(n->right);
		if (n->right) { n->right->right = skew(n->right->right); }
		n = split(n);
		n->right = split(n->right);
		return n;
	}

	template<typename F>
	static void for_each(const Node *n, F &f) noexcept {
		if (!n) { return; }
		for_each(n->left, f);
		f(n->value);
		for_each(n->right, f);
	}

	/// \return number of nodes, or -1 if an invariant is violated
	int64_t check(const Node *n) const noexcept {
		if (!n) { return 0; }
		const bool leaf = !n->left && !n->right;
		if (leaf && (n->level != 1)) { return -1; }
		if (n->left && (n->left->level + 1 != n->level)) { return -1; }
		if (!n->left && !leaf && (n->level != 1)) { return -1; }
		if (n->right && !((n->right->level == n->level) || (n->right->level + 1 == n->level))) { return -1; }
		if (!n->right && (n->level != 1)) { return -1; }
		if (n->right && n->right->right && !(n->right->right->level < n->level)) { return -1; }
		if (n->left && !less(n->left->value, n->value)) { return -1; }
		if (n->right && !less(n->value, n->right->value)) { return -1; }
		const int64_t l = check(n->left), r = check(n->right);
		if ((l < 0) || (r < 0)) { return -1; }
		return l + r + 1;
	}

public:
	constexpr AATreeSet() noexcept = default;
	AATreeSet(const AATreeSet &) = delete;
	AATreeSet &operator=(const AATreeSet &) = delete;
	~AATreeSet() noexcept { clear(); }

	/// \return true if the set is empty
	[[nodiscard]] constexpr inline bool empty() const noexcept {
		return size_ == 0;
	}

	/// \return number of values in the set
	[[nodiscard]] constexpr inline size_t len() const noexcept {
		return size_;
	}

	/// removes all values
	inline void clear() noexcept {
		destroy_tree(root);
		root = nullptr;
		size_ = 0;
	}

	/// \return true if `val` is in the set
	[[nodiscard]] constexpr inline bool contains(const T &val) const noexcept {
		const Node *n = root;
		while (n) {
			if (less(val, n->value)) {
				n = n->left;
			} else if (less(n->value, val)) {
				n = n->right;
			} else {
				return true;
			}
		}
		return false;
	}

	/// inserts `val`
	/// \return true if it was not in the set before
	inline bool insert(const T &val) noexcept {
		bool inserted = false;
		root = insert(root, val, inserted);
		size_ += inserted;
		return inserted;
	}

	/// removes `val`
	/// \return true if it was in the set
	inline bool erase(const T &val) noexcept {
		bool removed = false;
		root = erase(root, val, removed);
		size_ -= removed;
		return removed;
	}

	/// calls `f(value)` for every value in ascending order
	template<typename F>
	inline void for_each(F &&f) const noexcept {
		for_each(root, f);
	}

	/// \return true if the AA tree invariants and the search order hold and
	/// 	the size is correct
	[[nodiscard]] inline bool check() const noexcept {
		return check(root) == int64_t(size_);
	}
};
} // end namespace cryptanalysislib
#endif
