#ifndef CRYPTANALYSISLIB_CONTAINER_BK_TREE_H
#define CRYPTANALYSISLIB_CONTAINER_BK_TREE_H

#include <vector>


/// Details: https://dl.acm.org/doi/pdf/10.1145/362003.362025
template<class T>
class BKTreeNode {
public:
	using node_type = BKTreeNode<T>;

	std::vector<node_type> children;
	std::vector<uint32_t> duv;
	T data;

	BKTreeNode() = delete;
	BKTreeNode(const T data) noexcept : data(data) {}
};

struct BKTreeConfig : public AlignmentConfig {
};
constexpr static BKTreeConfig bkTreeConfig;

/// TODO iterator, allocator
template<class T,
		 const BKTreeConfig &config=bkTreeConfig>
class BKTree {
	using node_type = BKTreeNode<T>;
	node_type root = node_type(T());

	///
	constexpr void info() const noexcept {
		std::cout << " { name: \"BKTree\""
				  << " }" <<std::endl;
	}

	/// distance function
	/// \param a
	/// \param b
	/// \return
	constexpr static uint32_t d(const T &a,
	                            const T &b) noexcept {
		return T::dist(a, b);
	}

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
	///
	constexpr BKTree() noexcept {
		root.data.zero();
	}

	/// \param a
	/// \return
	constexpr void insert(const T &a) noexcept {
		_insert(a, root);
	}

	/// returns 0 on empty
	/// \param a
	/// \return
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
