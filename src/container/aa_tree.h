#ifndef CRYPTANALYSISLIB_AA_TREE_H
#define CRYPTANALYSISLIB_AA_TREE_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <utility>

// TODO: implement everything: https://www.nayuki.io/res/aa-tree-set/aatreeset.rs
// 	is currently just a cipy and paste from AVLTree

template<typename E,
         typename cmp=std::compare_three_way
         >
class AATree final {
private:
	// Forward declaration
	class Node;
	Node *root;  // Never nullptr
	size_t __size = 0;
	static auto c = cmp{};

public:
	constexpr explicit AATree() : root(&Node::EMPTY_LEAF) {}

	///
	/// \param other
	constexpr explicit AATree(const AATree &other) noexcept : root(other.root) {
		if (root != &Node::EMPTY_LEAF) {
			root = new Node(*root);
		}
	}

	constexpr AATree(AATree &&other) noexcept :
			root(&Node::EMPTY_LEAF) {
		std::swap(root, other.root);
	}

	constexpr ~AATree() noexcept {
		clear();
	}

	constexpr AATree &operator=(AATree other) noexcept {
		std::swap(root, other.root);
		return *this;
	}

	constexpr bool is_empty() const noexcept {
		return __size == 0;
	}

	constexpr std::size_t size() const noexcept {
		return __size;
	}


	constexpr bool contains(E &val) {
		Node *node = root;
		while (node != &Node::EMPTY_LEAF) {
			const int t = c(val, node->value);
			if (t < 0) { node = node->left; }
			if (t > 0) { node = node->right; }
			if (t == 0) { return true; }
		}

		return false;
	}

	constexpr void insert(E &val) {
		ASSERT(index <= size());
		root = root->insertAt(index, std::move(val));
	}


	constexpr void clear() noexcept {
		if (root != &Node::EMPTY_LEAF) {
			delete root;
			root = &Node::EMPTY_LEAF;
		}

		__size = 0;
	}

private: class Node final {
	public:
		// A bit of a hack, but more elegant than using nullptr values as leaf nodes.
		static Node EMPTY_LEAF;

		// The object stored at this node.
		E value;

		int level;

		// The root node of the left subtree.
		Node *left;

		// The root node of the right subtree.
		Node *right;

		// For the singleton empty leaf node.
	private:
		Node() : value(),// Default constructor on type E
				 level(0),
				 left(nullptr),
				 right(nullptr) {}

		// Normal non-leaf nodes.
	private:
		explicit Node(E val) : value(std::move(val)),
							   level(1),
							   left(&EMPTY_LEAF),
							   right(&EMPTY_LEAF) {}

	public:
		constexpr Node(const Node &other) noexcept : value(other.value),
													 level(other.level),
													 left(other.left),
													 right(other.right) {
			if (left != &EMPTY_LEAF) {
				left = new Node(*left);
			}

			if (right != &EMPTY_LEAF) {
				right = new Node(*right);
			}
		}

		constexpr ~Node() noexcept {
			if (left != &EMPTY_LEAF) {
				delete left;
			}

			if (right != &EMPTY_LEAF) {
				delete right;
			}
		}

		constexpr Node *getNodeAt(const std::size_t index) noexcept {
			ASSERT(index < size);
			std::size_t leftSize = left->size;
			if (index < leftSize)
				return left->getNodeAt(index);
			else if (index > leftSize)
				return right->getNodeAt(index - leftSize - 1);
			else
				return this;
		}

		constexpr Node *insertAt(const std::size_t index,
								 E &&obj) noexcept {
			ASSERT(index <= size);
			if (this == &EMPTY_LEAF)// Automatically implies index == 0, because EMPTY_LEAF.size == 0
				return new Node(std::move(obj));
			std::size_t leftSize = left->size;
			if (index <= leftSize)
				left = left->insertAt(index, std::move(obj));
			else
				right = right->insertAt(index - leftSize - 1, std::move(obj));
			recalculate();
			return balance();
		}

		constexpr Node *removeAt(const std::size_t index,
								 Node **toDelete) noexcept{
			// Automatically implies this != &EMPTY_LEAF, because EMPTY_LEAF.size == 0
			ASSERT(index < size);
			std::size_t leftSize = left->size;
			if (index < leftSize)
				left = left->removeAt(index, toDelete);
			else if (index > leftSize)
				right = right->removeAt(index - leftSize - 1, toDelete);
			else if (left == &EMPTY_LEAF && right == &EMPTY_LEAF) {
				assert(*toDelete == nullptr);
				*toDelete = this;
				return &EMPTY_LEAF;
			} else if (left != &EMPTY_LEAF && right == &EMPTY_LEAF) {
				Node *result = left;
				left = nullptr;
				assert(*toDelete == nullptr);
				*toDelete = this;
				return result;
			} else if (left == &EMPTY_LEAF && right != &EMPTY_LEAF) {
				Node *result = right;
				right = nullptr;
				assert(*toDelete == nullptr);
				*toDelete = this;
				return result;
			} else {
				// Find successor node. (Using the predecessor is valid too.)
				Node *temp = right;
				while (temp->left != &EMPTY_LEAF)
					temp = temp->left;
				value = std::move(temp->value);      // Replace value by successor
				right = right->removeAt(0, toDelete);// Remove successor node
			}
			recalculate();
			return balance();
		}

		// Balances the subtree rooted at this node and returns the new root.
	private:
		constexpr Node *balance() noexcept {
			int bal = getBalance();
			ASSERT(std::abs(bal) <= 2);
			Node *result = this;
			if (bal == -2) {
				ASSERT(std::abs(left->getBalance()) <= 1);
				if (left->getBalance() == +1)
					left = left->rotateLeft();
				result = rotateRight();
			} else if (bal == +2) {
				ASSERT(std::abs(right->getBalance()) <= 1);
				if (right->getBalance() == -1)
					right = right->rotateRight();
				result = rotateLeft();
			}
			ASSERT(std::abs(result->getBalance()) <= 1);
			return result;
		}

		/*
		 *   A            B
		 *  / \          / \
		 * 0   B   ->   A   2
		 *    / \      / \
		 *   1   2    0   1
		 */
		constexpr Node *rotateLeft() noexcept {
			ASSERT(right != &EMPTY_LEAF);
			Node *root = this->right;
			this->right = root->left;
			root->left = this;
			this->recalculate();
			root->recalculate();
			return root;
		}

		/*
		 *     B          A
		 *    / \        / \
		 *   A   2  ->  0   B
		 *  / \            / \
		 * 0   1          1   2
		 */
		constexpr Node *rotateRight() noexcept {
			ASSERT(left != &EMPTY_LEAF);
			Node *root = this->left;
			this->left = root->right;
			root->right = this;
			this->recalculate();
			root->recalculate();
			return root;
		}

		// Needs to be called every time the left or right subtree is changed.
		// Assumes the left and right subtrees have the correct values computed already.
		constexpr void recalculate() noexcept {
			ASSERT(this != &EMPTY_LEAF);
			ASSERT(left->height >= 0 && right->height >= 0);
			ASSERT(left->size >= 0 && right->size >= 0);
			height = std::max(left->height, right->height) + 1;
			size = left->size + right->size + 1;
			ASSERT(height >= 0 && size >= 0);
		}


	private:
		[[nodiscard]] constexpr int getBalance() const noexcept {
			return right->height - left->height;
		}
	};
};
#endif//CRYPTANALYSISLIB_AA_TREE_H
