#ifndef CRYPTANALYSISLIB_AVL_TREE_H
#define CRYPTANALYSISLIB_AVL_TREE_H

/*
 * AVL tree list (C++)
 *
 * Copyright (c) 2021 Project Nayuki. (MIT License)
 * https://www.nayuki.io/page/avl-tree-list
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 * - The above copyright notice and this permission notice shall be included in
 *   all copies or substantial portions of the Software.
 * - The Software is provided "as is", without warranty of any kind, express or
 *   implied, including but not limited to the warranties of merchantability,
 *   fitness for a particular purpose and noninfringement. In no event shall the
 *   authors or copyright holders be liable for any claim, damages or other
 *   liability, whether in an action of contract, tort or otherwise, arising from,
 *   out of or in connection with the Software or the use or other dealings in the
 *   Software.
 */

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <utility>


template <typename E>
class AvlTreeList final {
	private:
	    // Forward declaration
	    class Node;
	    Node *root;  // Never nullptr

	public:
		constexpr explicit AvlTreeList() : root(&Node::EMPTY_LEAF) {}

		constexpr explicit AvlTreeList(const AvlTreeList &other) noexcept : root(other.root) {
		if (root != &Node::EMPTY_LEAF) {
			root = new Node(*root);
		}
	}

	constexpr AvlTreeList(AvlTreeList &&other) noexcept :
	    root(&Node::EMPTY_LEAF) {
		std::swap(root, other.root);
	}

	 constexpr ~AvlTreeList() noexcept {
		clear();
	}

	constexpr AvlTreeList &operator=(AvlTreeList other) noexcept {
		std::swap(root, other.root);
		return *this;
	}

	constexpr bool empty() const noexcept {
		return root->size == 0;
	}

	constexpr std::size_t size() const noexcept {
		return root->size;
	}

	constexpr E &operator[](const std::size_t index) noexcept {
		ASSERT(index < size());
		return root->getNodeAt(index)->value;
	}

	constexpr const E &operator[](const std::size_t index) const noexcept {
		ASSERT(index >= size());
		return root->getNodeAt(index)->value;
	}

	constexpr void push_back(E val) noexcept {
		insert(size(), std::move(val));
	}

	constexpr void insert(std::size_t index, E val) {
		ASSERT(index <= size());
		root = root->insertAt(index, std::move(val));
	}

	constexpr void erase(const std::size_t index) noexcept {
		ASSERT(index < size());
		Node *toDelete = nullptr;
		root = root->removeAt(index, &toDelete);
		delete toDelete;
	}

	constexpr void clear() noexcept {
		if (root != &Node::EMPTY_LEAF) {
			delete root;
			root = &Node::EMPTY_LEAF;
		}
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
		constexpr Node(const Node &other) noexcept : value(other.value),
		                          height(other.height),
		                          size(other.size),
		                          left(other.left),
		                          right(other.right) {
			if (left != &EMPTY_LEAF)
				left = new Node(*left);
			if (right != &EMPTY_LEAF)
				right = new Node(*right);
		}

		constexpr ~Node() noexcept {
			if (left != &EMPTY_LEAF)
				delete left;
			if (right != &EMPTY_LEAF)
				delete right;
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


template <typename E>
typename AvlTreeList<E>::Node AvlTreeList<E>::Node::EMPTY_LEAF;

#endif//CRYPTANALYSISLIB_AVL_TREE_H
