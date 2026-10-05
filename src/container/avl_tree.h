#ifndef CRYPTANALYSISLIB_CONTAINER_AVL_TREE_H
#define CRYPTANALYSISLIB_CONTAINER_AVL_TREE_H

/// original code from:
///     https://www.nayuki.io/page/avl-tree-list
/// heavily modified by Floyd

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <utility>

#include "reflection/reflection.h"
#include "alloc/alloc.h"


struct AvlTreeConfig : public AlignmentConfig {
};
constexpr static AvlTreeConfig avlTreeConfig;

/// TODO use allocator class 
template <typename E,
          template<class N> class Allocator = cryptanalysislib::allocator,
		  const AvlTreeConfig &config=avlTreeConfig>
class AvlTreeList final {
private:
	    // Forward declaration
	    class Node;
	    // Never nullptr
	    Node *root;

public:
    /// Default constructor
	/// Initializes an empty AVL tree with a pointer to the empty leaf node
	constexpr explicit AvlTreeList() : root(&Node::EMPTY_LEAF) {}

	/// Creates a non-deep copy of the provided AVL tree
	/// 
	/// \param other[in] The AVL tree to copy
	constexpr explicit AvlTreeList(const AvlTreeList &other) noexcept : root(other.root) {
		if (root != &Node::EMPTY_LEAF) {
			root = new Node(*root);
		}
	}

	/// base type
	using S = AvlTreeList<E>;

	/// Move constructor
	/// Takes ownership of the tree from another AvlTreeList object
	/// 
	/// \param other[in] The AVL tree to move from
	constexpr AvlTreeList(AvlTreeList &&other) noexcept :
	    root(&Node::EMPTY_LEAF) {
		std::swap(root, other.root);
	}

	/// Destructor
	/// Cleans up all nodes in the tree by calling clear()
	constexpr ~AvlTreeList() noexcept {
		clear();
	}

    /// Assignment operator using copy-and-swap idiom
	/// Takes a copy of the argument and swaps with it
	/// 
	/// \param other[in] The AVL tree to assign from
	/// \return Reference to this object after assignment
	constexpr AvlTreeList &operator=(AvlTreeList other) noexcept {
		std::swap(root, other.root);
		return *this;
	}

	/// Checks if the tree is empty
	/// 
	/// \return True if the tree is empty, false otherwise
	[[nodiscard]] constexpr inline bool empty() const noexcept {
		return root->size == 0;
	}

	/// Returns the total count of elements stored in the AVL tree
	/// 
	/// \return Number of elements in the tree
	[[nodiscard]] constexpr inline std::size_t size() const noexcept {
		return root->size;
	}

	/// Accesses the element at the specified position
	/// Provides non-const access to the element at the given index
	/// 
	/// \param index[in] Position of the element to return
	/// \return Reference to the element at the specified position
	[[nodiscard]] constexpr E &operator[](const std::size_t index) noexcept {
		assert(index < size());
		return root->getNodeAt(index)->value;
	}

	/// Provides const access to the element at the given index
	/// 
	/// \param index[in] Position of the element to return
	/// \return Const reference to the element at the specified position
	[[nodiscard]] constexpr const E &operator[](const std::size_t index) const noexcept {
		assert(index < size());
		return root->getNodeAt(index)->value;
	}

	/// Inserts the given value at the end of the AVL tree
	/// 
	/// \param val[in] Value to be appended
	constexpr void push_back(E val) noexcept {
		insert(size(), std::move(val));
	}

	/// Adds a new element at the specified position in the AVL tree
	/// 
	/// \param index[in] Position where the new element is inserted
	/// \param val[in] Element to insert
	constexpr void insert(std::size_t index, E val) noexcept {
		assert(index <= size());
		root = root->insertAt(index, std::move(val));
	}

	/// Erases the element at the given position in the AVL tree
	/// 
	/// \param index[in] Position of the element to remove
	constexpr void erase(const std::size_t index) noexcept {
		assert(index < size());
		Node *toDelete = nullptr;
		root = root->removeAt(index, &toDelete);
		delete toDelete;
	}

	/// Deletes all nodes in the tree and resets it to empty state
	constexpr void clear() noexcept {
		if (root != &Node::EMPTY_LEAF) {
			delete root;
			root = &Node::EMPTY_LEAF;
		}
	}

	/// Prints information about the tree type
	/// Outputs the class name in JSON format for debugging and reflection
	constexpr static void info() noexcept {
		std::cout << " { name: \"AvlTreeList\" }" << std::endl;
	}

	private: class Node final {
		public:
		// A bit of a hack, but more elegant than using nullptr values as leaf nodes.
		static Node EMPTY_LEAF;

		// The object stored at this node.
		E value;

		// The height of the tree rooted at this node. Empty nodes have height 0.
		// This node has height equal to max(left->height, right->height) + 1.
		int height;

		// The number of non-empty nodes in the tree rooted at this node, including this node.
		// Empty nodes have size 0. This node has size equal to left->size + right->size + 1.
		std::size_t size;

		// The root node of the left subtree.
		Node *left;

		// The root node of the right subtree.
		Node *right;

		// For the singleton empty leaf node.
	private:
		Node() : value(),// Default constructor on type E
		         height(0),
		         size(0),
		         left(nullptr),
		         right(nullptr) {}

		// Normal non-leaf nodes.
	private:
		explicit Node(E val) : value(std::move(val)),
		                       height(1),
		                       size(1),
		                       left(&EMPTY_LEAF),
		                       right(&EMPTY_LEAF) {}

	public:
		/// Copy constructor for Node
		/// Creates a deep copy of the node and its children
		/// 
		/// \param other[in] The node to copy
		constexpr Node(const Node &other) noexcept : value(other.value),
		                          height(other.height),
		                          size(other.size),
		                          left(other.left),
		                          right(other.right) {
			if (left != &EMPTY_LEAF) {
				left = new Node(*left);
			}

			if (right != &EMPTY_LEAF) {
				right = new Node(*right);
			}
		}

		/// Destructor
		/// Recursively deletes all child nodes
		constexpr ~Node() noexcept {
			if (left != &EMPTY_LEAF)
				delete left;
			if (right != &EMPTY_LEAF)
				delete right;
		}

		/// Retrieves the node at the specified position
		/// 
		/// \param index[in] Position of the node to retrieve
		/// \return Pointer to the node at the specified position
		constexpr Node *getNodeAt(const std::size_t index) noexcept {
			assert(index < size);
			std::size_t leftSize = left->size;
			if (index < leftSize)
				return left->getNodeAt(index);
			else if (index > leftSize)
				return right->getNodeAt(index - leftSize - 1);
			else
				return this;
		}

		/// Inserts a new element at the specified position in this subtree
		/// Recursively finds the insertion point and maintains tree balance
		/// 
		/// \param index[in] Position where the element is to be inserted
		/// \param obj[in] Value to insert at the specified position
		/// \return Pointer to the new root of the subtree after insertion
		constexpr Node *insertAt(const std::size_t index,
		                         E &&obj) noexcept {
			assert(index <= size);
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

		/// Removes an element at the specified position in this subtree
		/// Recursively finds the node to remove and maintains tree balance
		/// Handles various cases of node removal (leaf, one child, two children)
		/// 
		/// \param index[in] Position of the element to remove
		/// \param toDelete[out] Pointer to store the node that needs to be deleted
		/// \return Pointer to the new root of the subtree after removal
		constexpr Node *removeAt(const std::size_t index,
		                         Node **toDelete) noexcept{
			// Automatically implies this != &EMPTY_LEAF, because EMPTY_LEAF.size == 0
			assert(index < size);
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

		/// Balances the subtree rooted at this node
		/// Checks the balance factor and performs rotations as needed
		/// Implements the AVL tree balancing algorithm
		/// 
		/// \return Pointer to the new root of the balanced subtree
	private:
		constexpr Node *balance() noexcept {
			int bal = getBalance();
			assert(std::abs(bal) <= 2);
			Node *result = this;
			if (bal == -2) {
				assert(std::abs(left->getBalance()) <= 1);
				if (left->getBalance() == +1)
					left = left->rotateLeft();
				result = rotateRight();
			} else if (bal == +2) {
				assert(std::abs(right->getBalance()) <= 1);
				if (right->getBalance() == -1)
					right = right->rotateRight();
				result = rotateLeft();
			}
			assert(std::abs(result->getBalance()) <= 1);
			return result;
		}

		/*
		 *   A            B
		 *  / \          / \
		 * 0   B   ->   A   2
		 *    / \      / \
		 *   1   2    0   1
		 */
		/// Performs a left rotation on this node
		/// Used in AVL tree balancing when the right subtree is too tall
		/// 
		/// \return Pointer to the new root after rotation
		constexpr Node *rotateLeft() noexcept {
			assert(right != &EMPTY_LEAF);
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
		/// Performs a right rotation on this node
		/// Used in AVL tree balancing when the left subtree is too tall
		/// 
		/// \return Pointer to the new root after rotation
		constexpr Node *rotateRight() noexcept {
			assert(left != &EMPTY_LEAF);
			Node *root = this->left;
			this->left = root->right;
			root->right = this;
			this->recalculate();
			root->recalculate();
			return root;
		}

		/// Updates the height and size of this node
		/// Must be called after any change to the subtrees
		/// Assumes the left and right subtrees have the correct values computed already
		constexpr void recalculate() noexcept {
			assert(this != &EMPTY_LEAF);
			assert(left->height >= 0 && right->height >= 0);
			assert(left->size >= 0 && right->size >= 0);
			height = std::max(left->height, right->height) + 1;
			size = left->size + right->size + 1;
			assert(height >= 0 && size >= 0);
		}

	private:
		/// Calculates the balance factor of this node
		/// The balance factor is the height of the right subtree minus the height of the left subtree
		/// Used to determine if the tree needs rebalancing
		/// 
		/// \return Balance factor (should be between -1 and 1 for a balanced tree)
		[[nodiscard]] constexpr int getBalance() const noexcept {
			return right->height - left->height;
		}
	};
};


template <typename E, const AvlTreeConfig &config>
typename AvlTreeList<E, config>::Node AvlTreeList<E,config>::Node::EMPTY_LEAF;

#endif//CRYPTANALYSISLIB_AVL_TREE_H
