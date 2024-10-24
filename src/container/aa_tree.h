#ifndef CRYPTANALYSISLIB_AA_TREE_H
#define CRYPTANALYSISLIB_AA_TREE_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <set>
#include <stdexcept>
#include <utility>

#include "memory/memory.h"

// TODO: implement everything: https://www.nayuki.io/res/aa-tree-set/aatreeset.rs
// 	is currently just a cipy and paste from AVLTree

struct AANodeConfig : public AlignmentConfig {
} aaNodeConfig;

// forward declaration
template<typename T, const AANodeConfig &config>
class AANode;

///
/// \tparam T
/// \tparam config
template <typename T,
          typename cmp = std::compare_three_way,
		  class Allocator = std::allocator<AANode<T, aaNodeConfig>>,
		  const AANodeConfig &config=aaNodeConfig>
class AAMaybeNode {
private:
	alignas(config.alignment) AANode<T, config> *ptr = nullptr;

	using allocator_type = Allocator;
	const Allocator &alloc = Allocator();
public:
	using node = AANode<T, config>;


	/// \return if the pointer is valid
	[[nodiscard]] constexpr inline bool exists() const noexcept {
		return ptr != 0;
	}

	/// \return
	[[nodiscard]] constexpr inline uint32_t level() const noexcept {
		if (!exists()) {
			return 0;
		}

		return ptr->level;
	}

	inline void pop() noexcept {
		// TODO free the stuff
	}

	[[nodiscard]] constexpr inline T& value() noexcept {
		ASSERT(!exists());
		return ptr->value;
	}

	/// \return
	[[nodiscard]] constexpr inline node& node_ref() noexcept {
		ASSERT(exists());
		return *ptr;
	}

	/// \return
	[[nodiscard]] constexpr inline node& node_ref() const noexcept {
		ASSERT(exists());
		return *ptr;
	}

	/// \param val
	/// \return
	[[nodiscard]] constexpr inline auto& insert(const T val) noexcept {
		if (!exists()) {
			ptr = alloc.allocate(sizeof(node));
			ptr->value = val;
			return *this;
		}

		// TODO what if full
	}

	///       |          |
	///   A - B    ->    A - B
	///  / \   \        /   / \
	/// 0   1   2      0   1   2
	/// \param val
	/// \return
	[[nodiscard]] constexpr inline auto& skew(const T val) noexcept {
		ASSERT(exists());
		auto selfnode = node_ref();
		if (selfnode.left.level() < level()) {
			return *this;
		}

		// TODO rest
	}
};


template <typename T,
          const AANodeConfig &config=aaNodeConfig>
class AANode {
private:
	alignas(config.alignment) T value;

	AANode *left = nullptr;
	AANode *right = nullptr;
	const uint32_t level = 1;
public:
};

template<typename T,
		 typename Cmp = std::compare_three_way,
		 class Allocator = std::allocator<AANode<T, aaNodeConfig>>,
		 const AANodeConfig &config=aaNodeConfig>
class AATreeSet {
private:
	using mnode = AAMaybeNode<T, Cmp, Allocator, config>;
	mnode root{};
	size_t size = 0;
public:
	constexpr AATreeSet() noexcept = default;

	/// \return
	[[nodiscard]] constexpr inline bool empty() const noexcept {
		return size == 0;
	}

	/// \return
	[[nodiscard]] constexpr inline size_t len() const noexcept {
		return size;
	}

	///
	inline void clear() noexcept {
		root.pop();
	}

	///
	/// \return
	[[nodiscard]] constexpr inline bool contains(const T &val) const noexcept {
		auto node = root;
		while (node.exists()) {
			const auto t = Cmp(val.value);
		}
		return false;
	}

};
#endif
