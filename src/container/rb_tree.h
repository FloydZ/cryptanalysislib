#ifndef CRYPTANALYSISLIB_RB_TREE_H
#define CRYPTANALYSISLIB_RB_TREE_H

#include <vector>
#include <cstdlib>
#include <cstddef>

#include "helper.h"

enum RB_Color {
	RED,
	BLACK,
	DOUBLE_BLACK,
};


template<typename V>
struct RB_Node {
	typedef RB_Node* ptr;
	typedef const RB_Node* const_ptr;
public:
	RB_Color 	c;
	ptr 		*left;
	ptr 		*parent;
	V val;
};


template<typename K, // key
		 typename V  // value
		 >
class RB_Tree {
private:
	typedef RB_Node<V>* 		node_ptr;
	typedef const RB_Node<V>* 	const_node_ptr;
	
	node_ptr root = nullptr;
public:
	typedef K key_type;
	typedef V value_type;
	typedef value_type* pointer;
	typedef const value_type* const_pointer;
	typedef value_type& reference;
	typedef const value_type& const_reference;
	typedef node_ptr _Link_type;
	typedef const_node_ptr _Const_Link_type;
	typedef size_t size_type;
	typedef ptrdiff_t difference_type;
	// TODO allocator typedef _Alloc allocator_type;
	
	constexpr inline RB_Color get_color(const node_ptr &ptr) {
		if (ptr == nullptr) { return BLACK; }
		return ptr->c;
	}

	constexpr inline void set_color(node_ptr &ptr, const RB_Color cc) {
		ASSERT(ptr);
		ptr->c = cc;
	}

	constexpr node_ptr insertBST(node_ptr &root, node_ptr &ptr) noexcept {
    if (root == nullptr)
        return ptr;

    if (ptr->data < root->data) {
        root->left = insertBST(root->left, ptr);
        root->left->parent = root;
    } else if (ptr->data > root->data) {
        root->right = insertBST(root->right, ptr);
        root->right->parent = root;
    }

    return root;
}


};

#endif
