#ifndef CRYPTANALYSISLIB_CONTAINER_BPLUS_TREE_H
#define CRYPTANALYSISLIB_CONTAINER_BPLUS_TREE_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>

namespace cryptanalysislib {
	/// In-memory B+ tree map: the key/value pairs are in the leaves, which
	/// are linked (in order), the inner nodes only contain separators:
	/// all keys of child `i+1` are >= `keys[i]`, all keys of child `i` are
	/// < `keys[i]`. Every node except the root has between `Order/2 - 1`
	/// (rounded down) and `Order - 1` keys, all leaves have the same depth.
	/// idea: http://www.amittai.com/prose/bplustree.html
	/// Usage:
	/// \code
	/// 	bplus_tree<uint64_t, uint32_t> t;
	/// 	t.insert(5, 1);
	/// 	if (auto *v = t.find(5)) { ... }
	/// 	for (auto [k, v] : t) { ... } // in order
	/// \endcode
	/// \tparam Key default constructible and copy assignable
	/// \tparam Value default constructible and copy assignable
	/// \tparam Order maximal number of children of a node (>= 4)
	/// \tparam Compare strict weak order on `Key`
	template<typename Key,
	         typename Value,
	         const uint32_t Order = 32,
	         class Compare = std::less<Key>>
	class bplus_tree {
		static_assert(Order >= 4);

		/// maximal/minimal number of keys of a node (except the root)
		constexpr static uint32_t max_keys = Order - 1;
		constexpr static uint32_t min_keys = max_keys / 2;

		struct leaf_node {
			uint32_t n = 0;
			Key keys[max_keys];
			Value values[max_keys];
			leaf_node *prev = nullptr;
			leaf_node *next = nullptr;
		};

		struct inner_node {
			uint32_t n = 0;
			Key keys[max_keys];
			/// `n+1` children: `inner_node` or `leaf_node` (the last level)
			void *children[max_keys + 1];
		};

		/// `leaf_node` if `height == 0`, else `inner_node`, `nullptr` if empty
		void *root = nullptr;
		/// number of inner levels
		uint32_t height = 0;
		size_t nr_elements = 0;
		Compare comp;

		[[nodiscard]] bool equal(const Key &a, const Key &b) const noexcept {
			return !comp(a, b) && !comp(b, a);
		}

		/// \return the first position `i` in `keys[0, n)` with `keys[i] >= key`
		[[nodiscard]] uint32_t lower(const Key *keys, const uint32_t n, const Key &key) const noexcept {
			uint32_t lo = 0, hi = n;
			while (lo < hi) {
				const uint32_t mid = (lo + hi) / 2;
				if (comp(keys[mid], key)) { lo = mid + 1; } else { hi = mid; }
			}
			return lo;
		}

		/// \return the first position `i` in `keys[0, n)` with `keys[i] > key`,
		/// 	i.e. the child of an inner node, which contains `key`
		[[nodiscard]] uint32_t upper(const Key *keys, const uint32_t n, const Key &key) const noexcept {
			uint32_t lo = 0, hi = n;
			while (lo < hi) {
				const uint32_t mid = (lo + hi) / 2;
				if (!comp(key, keys[mid])) { lo = mid + 1; } else { hi = mid; }
			}
			return lo;
		}

		/// \return the leaf, which contains `key` (if present)
		[[nodiscard]] leaf_node *find_leaf(const Key &key) const noexcept {
			void *v = root;
			for (uint32_t l = 0; l < height; ++l) {
				auto *in = (inner_node *)v;
				v = in->children[upper(in->keys, in->n, key)];
			}
			return (leaf_node *)v;
		}

		void free_rec(void *v, const uint32_t level) noexcept {
			if (level == 0) {
				delete (leaf_node *)v;
				return;
			}
			auto *in = (inner_node *)v;
			for (uint32_t i = 0; i <= in->n; ++i) {
				free_rec(in->children[i], level - 1);
			}
			delete in;
		}

		/// result of an insertion into a subtree
		struct split_result {
			/// the subtree was split: `right` is the new right sibling,
			/// whose keys are >= `key`
			void *right = nullptr;
			Key key{};
		};

		/// inserts (key, value) into the subtree `v`
		/// \param inserted[out]: false if `key` was present
		/// \param assign[in]: overwrite the value of a present key
		split_result insert_rec(void *v,
		                        const uint32_t level,
		                        const Key &key,
		                        const Value &value,
		                        bool &inserted,
		                        const bool assign) {
			if (level == 0) {
				auto *leaf = (leaf_node *)v;
				const uint32_t pos = lower(leaf->keys, leaf->n, key);
				if ((pos < leaf->n) && equal(leaf->keys[pos], key)) {
					inserted = false;
					if (assign) { leaf->values[pos] = value; }
					return {};
				}

				inserted = true;
				if (leaf->n < max_keys) {
					for (uint32_t i = leaf->n; i > pos; --i) {
						leaf->keys[i] = leaf->keys[i - 1];
						leaf->values[i] = leaf->values[i - 1];
					}
					leaf->keys[pos] = key;
					leaf->values[pos] = value;
					leaf->n += 1;
					return {};
				}

				// split: the lower half stays, the upper half moves right
				Key tk[max_keys + 1];
				Value tv[max_keys + 1];
				for (uint32_t i = 0, j = 0; i <= max_keys; ++i) {
					if (i == pos) {
						tk[i] = key;
						tv[i] = value;
					} else {
						tk[i] = leaf->keys[j];
						tv[i] = leaf->values[j];
						j += 1;
					}
				}

				auto *right = new leaf_node;
				const uint32_t left_n = (max_keys + 1) / 2;
				leaf->n = left_n;
				right->n = max_keys + 1 - left_n;
				for (uint32_t i = 0; i < left_n; ++i) {
					leaf->keys[i] = tk[i];
					leaf->values[i] = tv[i];
				}
				for (uint32_t i = 0; i < right->n; ++i) {
					right->keys[i] = tk[left_n + i];
					right->values[i] = tv[left_n + i];
				}

				right->next = leaf->next;
				right->prev = leaf;
				if (leaf->next) { leaf->next->prev = right; }
				leaf->next = right;
				return {right, right->keys[0]};
			}

			auto *in = (inner_node *)v;
			const uint32_t idx = upper(in->keys, in->n, key);
			const split_result r = insert_rec(in->children[idx], level - 1, key, value, inserted, assign);
			if (r.right == nullptr) {
				return {};
			}

			if (in->n < max_keys) {
				for (uint32_t i = in->n; i > idx; --i) {
					in->keys[i] = in->keys[i - 1];
					in->children[i + 1] = in->children[i];
				}
				in->keys[idx] = r.key;
				in->children[idx + 1] = r.right;
				in->n += 1;
				return {};
			}

			// split: the middle key moves up
			Key tk[max_keys + 1];
			void *tc[max_keys + 2];
			for (uint32_t i = 0, j = 0; i <= max_keys; ++i) {
				tk[i] = (i == idx) ? r.key : in->keys[j++];
			}
			for (uint32_t i = 0, j = 0; i <= max_keys + 1; ++i) {
				tc[i] = (i == idx + 1) ? r.right : in->children[j++];
			}

			auto *right = new inner_node;
			const uint32_t mid = (max_keys + 1) / 2;
			in->n = mid;
			right->n = max_keys - mid;
			for (uint32_t i = 0; i < mid; ++i) { in->keys[i] = tk[i]; }
			for (uint32_t i = 0; i <= mid; ++i) { in->children[i] = tc[i]; }
			for (uint32_t i = 0; i < right->n; ++i) { right->keys[i] = tk[mid + 1 + i]; }
			for (uint32_t i = 0; i <= right->n; ++i) { right->children[i] = tc[mid + 1 + i]; }
			return {right, tk[mid]};
		}

		/// removes the key `idx` and the child `idx + 1` of `in`
		static void remove_from_inner(inner_node *in, const uint32_t idx) noexcept {
			for (uint32_t i = idx; i + 1 < in->n; ++i) {
				in->keys[i] = in->keys[i + 1];
				in->children[i + 1] = in->children[i + 2];
			}
			in->n -= 1;
		}

		/// restores the minimal fill of the child `idx` of `parent`, whose
		/// children are leaves
		void fix_leaf(inner_node *parent, const uint32_t idx) noexcept {
			auto *c = (leaf_node *)parent->children[idx];
			auto *l = idx > 0 ? (leaf_node *)parent->children[idx - 1] : nullptr;
			auto *r = idx < parent->n ? (leaf_node *)parent->children[idx + 1] : nullptr;

			if (l && l->n > min_keys) {
				// borrow the last pair of the left sibling
				for (uint32_t i = c->n; i > 0; --i) {
					c->keys[i] = c->keys[i - 1];
					c->values[i] = c->values[i - 1];
				}
				c->keys[0] = l->keys[l->n - 1];
				c->values[0] = l->values[l->n - 1];
				c->n += 1;
				l->n -= 1;
				parent->keys[idx - 1] = c->keys[0];
			} else if (r && r->n > min_keys) {
				// borrow the first pair of the right sibling
				c->keys[c->n] = r->keys[0];
				c->values[c->n] = r->values[0];
				c->n += 1;
				for (uint32_t i = 0; i + 1 < r->n; ++i) {
					r->keys[i] = r->keys[i + 1];
					r->values[i] = r->values[i + 1];
				}
				r->n -= 1;
				parent->keys[idx] = r->keys[0];
			} else {
				// merge the right one of (l, c) or (c, r) into the left one
				leaf_node *a = l ? l : c, *b = l ? c : r;
				const uint32_t key_idx = l ? idx - 1 : idx;
				for (uint32_t i = 0; i < b->n; ++i) {
					a->keys[a->n + i] = b->keys[i];
					a->values[a->n + i] = b->values[i];
				}
				a->n += b->n;
				a->next = b->next;
				if (b->next) { b->next->prev = a; }
				delete b;
				remove_from_inner(parent, key_idx);
			}
		}

		/// restores the minimal fill of the child `idx` of `parent`, whose
		/// children are inner nodes
		void fix_inner(inner_node *parent, const uint32_t idx) noexcept {
			auto *c = (inner_node *)parent->children[idx];
			auto *l = idx > 0 ? (inner_node *)parent->children[idx - 1] : nullptr;
			auto *r = idx < parent->n ? (inner_node *)parent->children[idx + 1] : nullptr;

			if (l && l->n > min_keys) {
				// rotate right: separator down to c, last key of l up
				c->children[c->n + 1] = c->children[c->n];
				for (uint32_t i = c->n; i > 0; --i) {
					c->keys[i] = c->keys[i - 1];
					c->children[i] = c->children[i - 1];
				}
				c->keys[0] = parent->keys[idx - 1];
				c->children[0] = l->children[l->n];
				c->n += 1;
				parent->keys[idx - 1] = l->keys[l->n - 1];
				l->n -= 1;
			} else if (r && r->n > min_keys) {
				// rotate left: separator down to c, first key of r up
				c->keys[c->n] = parent->keys[idx];
				c->children[c->n + 1] = r->children[0];
				c->n += 1;
				parent->keys[idx] = r->keys[0];
				for (uint32_t i = 0; i + 1 < r->n; ++i) {
					r->keys[i] = r->keys[i + 1];
					r->children[i] = r->children[i + 1];
				}
				r->children[r->n - 1] = r->children[r->n];
				r->n -= 1;
			} else {
				// merge: a + separator + b
				inner_node *a = l ? l : c, *b = l ? c : r;
				const uint32_t key_idx = l ? idx - 1 : idx;
				a->keys[a->n] = parent->keys[key_idx];
				for (uint32_t i = 0; i < b->n; ++i) {
					a->keys[a->n + 1 + i] = b->keys[i];
				}
				for (uint32_t i = 0; i <= b->n; ++i) {
					a->children[a->n + 1 + i] = b->children[i];
				}
				a->n += 1 + b->n;
				delete b;
				remove_from_inner(parent, key_idx);
			}
		}

		/// removes `key` from the subtree `v`
		/// \return true if `key` was present
		bool erase_rec(void *v, const uint32_t level, const Key &key) noexcept {
			if (level == 0) {
				auto *leaf = (leaf_node *)v;
				const uint32_t pos = lower(leaf->keys, leaf->n, key);
				if ((pos >= leaf->n) || !equal(leaf->keys[pos], key)) {
					return false;
				}
				for (uint32_t i = pos; i + 1 < leaf->n; ++i) {
					leaf->keys[i] = leaf->keys[i + 1];
					leaf->values[i] = leaf->values[i + 1];
				}
				leaf->n -= 1;
				return true;
			}

			auto *in = (inner_node *)v;
			const uint32_t idx = upper(in->keys, in->n, key);
			if (!erase_rec(in->children[idx], level - 1, key)) {
				return false;
			}

			if (level == 1) {
				if (((leaf_node *)in->children[idx])->n < min_keys) { fix_leaf(in, idx); }
			} else {
				if (((inner_node *)in->children[idx])->n < min_keys) { fix_inner(in, idx); }
			}
			return true;
		}

		/// \return true if the subtree `v` is valid, all keys in [lo, hi)
		bool verify_rec(const void *v,
		                const uint32_t level,
		                const Key *lo,
		                const Key *hi,
		                const bool is_root,
		                size_t &count,
		                const leaf_node *&prev_leaf) const noexcept {
			if (level == 0) {
				const auto *leaf = (const leaf_node *)v;
				if (!is_root && (leaf->n < min_keys || leaf->n > max_keys)) { return false; }
				for (uint32_t i = 0; i < leaf->n; ++i) {
					if (i > 0 && !comp(leaf->keys[i - 1], leaf->keys[i])) { return false; }
					if (lo && comp(leaf->keys[i], *lo)) { return false; }
					if (hi && !comp(leaf->keys[i], *hi)) { return false; }
				}
				if (leaf->prev != prev_leaf) { return false; }
				if (prev_leaf && prev_leaf->next != leaf) { return false; }
				prev_leaf = leaf;
				count += leaf->n;
				return true;
			}

			const auto *in = (const inner_node *)v;
			if (in->n > max_keys || (is_root ? in->n < 1 : in->n < min_keys)) { return false; }
			for (uint32_t i = 0; i <= in->n; ++i) {
				if (i > 0 && i < in->n && !comp(in->keys[i - 1], in->keys[i])) { return false; }
				const Key *l = i == 0 ? lo : &in->keys[i - 1];
				const Key *h = i == in->n ? hi : &in->keys[i];
				if (!verify_rec(in->children[i], level - 1, l, h, false, count, prev_leaf)) { return false; }
			}
			return true;
		}

	public:
		/// forward iterator over the pairs, in order
		class iterator {
			friend class bplus_tree;
			leaf_node *leaf = nullptr;
			uint32_t pos = 0;
			iterator(leaf_node *l, const uint32_t p) noexcept : leaf(l), pos(p) {
				// an iterator at the end of a leaf points to the next leaf
				if (leaf && pos >= leaf->n) {
					leaf = leaf->next;
					pos = 0;
				}
			}

		public:
			/// `first`: key, `second`: value
			struct reference {
				const Key &first;
				Value &second;
			};

			iterator() noexcept = default;
			reference operator*() const noexcept { return {leaf->keys[pos], leaf->values[pos]}; }
			const Key &key() const noexcept { return leaf->keys[pos]; }
			Value &value() const noexcept { return leaf->values[pos]; }
			iterator &operator++() noexcept {
				pos += 1;
				if (pos >= leaf->n) {
					leaf = leaf->next;
					pos = 0;
				}
				return *this;
			}
			bool operator==(const iterator &o) const noexcept { return leaf == o.leaf && pos == o.pos; }
			bool operator!=(const iterator &o) const noexcept { return !(*this == o); }
		};

		bplus_tree() noexcept = default;
		bplus_tree(const bplus_tree &) = delete;
		bplus_tree &operator=(const bplus_tree &) = delete;
		bplus_tree(bplus_tree &&o) noexcept : root(o.root), height(o.height), nr_elements(o.nr_elements), comp(o.comp) {
			o.root = nullptr;
			o.height = 0;
			o.nr_elements = 0;
		}

		~bplus_tree() noexcept {
			clear();
		}

		/// removes all elements
		void clear() noexcept {
			if (root) { free_rec(root, height); }
			root = nullptr;
			height = 0;
			nr_elements = 0;
		}

		/// \return number of elements
		[[nodiscard]] size_t size() const noexcept { return nr_elements; }
		[[nodiscard]] bool empty() const noexcept { return nr_elements == 0; }
		/// \return number of inner levels (0: the root is a leaf)
		[[nodiscard]] uint32_t depth() const noexcept { return height; }

		/// inserts (key, value), if `key` is not present
		/// \return true if inserted, false if `key` was present (unchanged)
		bool insert(const Key &key, const Value &value) {
			return insert_impl(key, value, false);
		}

		/// inserts (key, value), or overwrites the value of a present `key`
		/// \return true if inserted, false if overwritten
		bool insert_or_assign(const Key &key, const Value &value) {
			return insert_impl(key, value, true);
		}

		/// \return the value of `key`, `nullptr` if not present
		[[nodiscard]] Value *find(const Key &key) const noexcept {
			if (!root) { return nullptr; }
			leaf_node *leaf = find_leaf(key);
			const uint32_t pos = lower(leaf->keys, leaf->n, key);
			if ((pos < leaf->n) && equal(leaf->keys[pos], key)) {
				return &leaf->values[pos];
			}
			return nullptr;
		}

		[[nodiscard]] bool contains(const Key &key) const noexcept {
			return find(key) != nullptr;
		}

		/// removes `key`
		/// \return true if `key` was present
		bool erase(const Key &key) noexcept {
			if (!root || !erase_rec(root, height, key)) {
				return false;
			}

			nr_elements -= 1;
			if (height > 0) {
				auto *in = (inner_node *)root;
				if (in->n == 0) {
					root = in->children[0];
					height -= 1;
					delete in;
				}
			} else if (((leaf_node *)root)->n == 0) {
				delete (leaf_node *)root;
				root = nullptr;
			}
			return true;
		}

		/// \return iterator to the first element with a key >= `key`
		[[nodiscard]] iterator lower_bound(const Key &key) const noexcept {
			if (!root) { return end(); }
			leaf_node *leaf = find_leaf(key);
			return iterator(leaf, lower(leaf->keys, leaf->n, key));
		}

		[[nodiscard]] iterator begin() const noexcept {
			if (!root) { return end(); }
			void *v = root;
			for (uint32_t l = 0; l < height; ++l) { v = ((inner_node *)v)->children[0]; }
			return iterator((leaf_node *)v, 0);
		}

		[[nodiscard]] iterator end() const noexcept {
			return iterator();
		}

		/// checks all invariants (order, fill, separators, leaf links, size)
		[[nodiscard]] bool verify() const noexcept {
			if (!root) { return nr_elements == 0 && height == 0; }
			size_t count = 0;
			const leaf_node *prev = nullptr;
			if (!verify_rec(root, height, nullptr, nullptr, true, count, prev)) { return false; }
			return (prev->next == nullptr) && (count == nr_elements);
		}

	private:
		bool insert_impl(const Key &key, const Value &value, const bool assign) {
			if (!root) {
				auto *leaf = new leaf_node;
				leaf->keys[0] = key;
				leaf->values[0] = value;
				leaf->n = 1;
				root = leaf;
				nr_elements = 1;
				return true;
			}

			bool inserted = false;
			const split_result r = insert_rec(root, height, key, value, inserted, assign);
			if (r.right) {
				auto *in = new inner_node;
				in->n = 1;
				in->keys[0] = r.key;
				in->children[0] = root;
				in->children[1] = r.right;
				root = in;
				height += 1;
			}

			nr_elements += inserted;
			return inserted;
		}
	};
} // end namespace cryptanalysislib
#endif
