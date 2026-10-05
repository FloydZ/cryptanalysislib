#ifndef CRYPTANALYSISLIB_CONTAINER_BK_TREE_H
#define CRYPTANALYSISLIB_CONTAINER_BK_TREE_H

#include <vector>

#include "alloc/alloc.h"

/// TODO add namespace
/// TODO add concept for T, as T needs a dist functions

/// Details: https://dl.acm.org/doi/pdf/10.1145/362003.362025
template<class T>
class BKTreeNode {
public:
	using node_type = BKTreeNode<T>;

	std::vector<node_type> children;
	std::vector<uint32_t> duv;
	T data;

	/// Default constructor is deleted to ensure nodes always have data
	BKTreeNode() = delete;
	
	/// Constructor that initializes a BK-tree node with the given data
	/// 
	/// \param data[in] The data to store in this node
	BKTreeNode(const T data) noexcept : data(data) {}
};

struct BKTreeConfig : public AlignmentConfig {
};
constexpr static BKTreeConfig bkTreeConfig;

/// Burkhard-Keller Tree (BK-tree) implementation for approximate string matching and similarity search
/// A BK-tree is a metric tree specifically adapted to discrete metric spaces
/// 
/// \tparam T Type of elements stored in the tree (must provide a static dist method)
/// \tparam Allocator Allocator used for memory management
/// \tparam config Configuration options for the BK-tree
template<class T,
         template<class N> class Allocator = cryptanalysislib::allocator,
		 const BKTreeConfig &config=bkTreeConfig>
class BKTree {
	using node_type = BKTreeNode<T>;
	node_type root = node_type(T());

	/// Prints information about the BK-tree structure
	/// Outputs the tree name and allocator information in JSON format
	void info() const noexcept {
		std::cout << " { name: \"BKTree\""
				  << " , \"allocator\": " << Allocator<node_type>::str()
				  // << " , \"config\": " << config
				  << " }" <<std::endl;
	}

	/// Calculates the distance between two elements
	/// Uses the static dist method provided by the template parameter type T
	/// This distance function is critical for the BK-tree's operation
	/// 
	/// \param a[in] First element to compare
	/// \param b[in] Second element to compare
	/// \return Distance value between the two elements
	constexpr static uint32_t d(const T &a,
	                            const T &b) noexcept {
		return T::dist(a, b);
	}

	/// Internal recursive method to insert an element into the BK-tree
	/// Navigates the tree structure based on distance values and adds new nodes as needed
	/// 
	/// \param a[in] Element to insert into the tree
	/// \param node[in,out] Current node being examined for insertion
	void _insert(const T&a, node_type &node) noexcept {
		const uint32_t k = d(a, node.data);
		if (k == 0) {
			// already inserted
			return;
		}

		for (uint32_t i = 0; i < node.children.size(); ++i) {
			if (node.duv[i] == k) {
				// simply take the first one
				_insert(a, node.children[i]);
				return;
			}
		}

		node.children.emplace_back(node_type(a));
		node.duv.emplace_back(k);
	}

public:
	/// Default constructor
	/// Initializes an empty BK-tree with a zero-initialized root node
	constexpr BKTree() noexcept {
		root.data.zero();
	}

	/// Inserts an element into the BK-tree
	/// Public interface that delegates to the private recursive implementation
	/// 
	/// \param a[in] Element to insert into the tree
	constexpr void insert(const T &a) noexcept {
		_insert(a, root);
	}

	/// Finds the closest element in the tree to the given query element
	/// Implements an iterative search algorithm using the triangle inequality property
	/// Returns 0 if the tree is empty
	/// 
	/// \param a[in] Query element to find closest match for
	/// \return Minimum distance found between query and any element in the tree
	constexpr uint32_t lookup(const T &a) const noexcept {
		if (root.children.size() == 0) { return 0; }

		std::vector<node_type> S;
		S.emplace_back(root);
		uint32_t d_best = uint32_t(-1);

		while (S.size() > 0) {
			const auto u = S[S.size() - 1];
			S.pop_back();
			const uint32_t du = d(a, u.data);
			if (du < d_best) { d_best = du; }

			for (size_t i = 0; i < u.children.size(); ++i) {
				if (std::abs((int32_t)u.duv[i] - (int32_t)du) < (int32_t)d_best) {
					S.emplace_back(u.children[i]);
				}
			}
		}

		return d_best;
	}
};
#endif//CRYPTANALYSISLIB_BK_TREE_H
