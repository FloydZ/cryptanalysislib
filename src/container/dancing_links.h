#pragma once

#include <memory>
#include <vector>
#include "alloc/alloc.h"


/// TODO iterator
/// \tparam T
template <class T,
		  template<class> class Allocator=cryptanalysislib::allocator>
struct dancing_links {
public:
	struct node {
		T item;
		node *l, *r;

		/// \param _item
		/// \param _l
		/// \param _r
		node(const T &_item,
			 node *_l=nullptr,
			 node *_r=nullptr)
		  : item(_item), l(_l), r(_r) {
			if (l != nullptr) { l->r = this; }
			if (r != nullptr) { r->l = this; }
		}
	};

private:
	node *front = nullptr,
		 *back = nullptr;
	size_t size_ = 0;
	// NOTE: every node ever allocated. `erase` only unlinks a node (so that
	// 	`restore` can relink it), thus the list itself does not know all nodes.
	std::vector<node *> nodes_;
public:
	///
	dancing_links() noexcept { }

	/// NOTE: before, no node was ever freed
	~dancing_links() noexcept {
		for (node *n : nodes_) { delete n; }
	}

	// NOTE: the nodes are owned by the container
	dancing_links(const dancing_links &) = delete;
	dancing_links &operator=(const dancing_links &) = delete;

	/// NOTE: linear search, as the container is not sorted
	/// \param item
	/// \return
	node *find_front(const T &item) const noexcept {
		node *t = front;
		while (t != nullptr) {
			if (t->item == item) {
				return t;
			}
			t = t->r;
		}

		return nullptr;
	}

	/// NOTE: linear search, as the container is not sorted
	/// \param item
	/// \return
	node *find_back(const T &item) const noexcept {
		node *t = back;
		while (t != nullptr) {
			if (t->item == item) {
				return t;
			}
			t = t->l;
		}

		return nullptr;
	}

	/// \param item
	/// \return
	node *push_back(const T &item) noexcept {
		back = new node(item, back, nullptr);
		nodes_.push_back(back);
		if (!front) {
			front = back;
		}
		size_ += 1;
		return back;
	}

	/// \param item
	/// \return
	node *push_front(const T &item) noexcept {
		front = new node(item, nullptr, front);
		nodes_.push_back(front);
		if (!back) {
			back = front;
		}
		size_ += 1;
		return front;
	}

	/// \param n
	void erase(node *n) noexcept {
		if (!n->l) { front = n->r; } else { n->l->r = n->r; }
		if (!n->r) { back = n->l; } else { n->r->l = n->l; }
		size_ -= 1;
	}

	/// \param n
	void restore(node *n) noexcept {
		if (!n->l) { front = n; } else { n->l->r = n; }
		if (!n->r) { back = n; } else { n->r->l = n; }
		size_ += 1;
	}

	/// \return size
	constexpr inline size_t size() const noexcept {
		return size_;
	}

	/// \param reverse
	void print(const bool reverse=false) const noexcept {
		// NOTE: was inverted, `reverse=false` printed back to front
		if (!reverse) {
			node *t = front;
			while (t != nullptr) {
				std::cout << t->item << " ";
				t = t->r;
			}

		} else {
			node *t = back;
			while (t != nullptr) {
				std::cout << t->item << " ";
				t = t->l;
			}
		}
		std::cout << std::endl;
	}
};
