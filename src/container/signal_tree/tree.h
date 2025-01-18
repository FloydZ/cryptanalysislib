#pragma once

#include <cstdint>
#include <concepts>
#include <bit>
#include <array>
#include <atomic>
#include <functional>
#include <utility>

// TODO: type rich
using signal_index = std::uint64_t;
static signal_index constexpr invalid_signal_index{~0ull};


static thread_local std::uint64_t select_bias_hint = 0;


static inline constexpr auto minimum_bit_count(const std::unsigned_integral auto value ) {
	constexpr static size_t bits_per_byte = 8ull;
    return ((sizeof(value) * bits_per_byte) - std::countl_zero(value));
}



template <std::size_t, std::size_t>
struct sub_counter_arity;

//=============================================================================
template <std::size_t counter_capacity>
struct sub_counter_arity<counter_capacity, 0> {
    static auto constexpr value = 0;
};

//=============================================================================
template <std::size_t counter_capacity, std::size_t N>
struct sub_counter_arity {
    static auto constexpr bits_per_counter = (64ull - std::countl_zero(counter_capacity / N));
    static auto constexpr bits_required = (bits_per_counter * N);
    static auto constexpr value = (bits_required <= 64) ? N :
            sub_counter_arity<counter_capacity, N / 2>::value;
};

//=========================================================================
template <std::size_t counter_capacity>
static auto constexpr sub_counter_arity_v = sub_counter_arity<counter_capacity, 64>::value;

//=========================================================================
// these will be type rich types in the near future
using tree_index = std::uint64_t;

///
/// @tparam total_counters
/// @tparam bits_per_counter
/// @tparam bias_bit
template <std::uint64_t total_counters,
		 std::uint64_t bits_per_counter, std::uint64_t bias_bit = (1ull << 63)>
struct default_selector {
        // default select will select which ever child is non-zero, or if both
        // children are non-zero, prefer the child indicated by the bias flag.
    inline auto operator()(std::uint64_t biasFlags,
        				   std::uint64_t counters,
        				   std::uint64_t nextBias = 0) const noexcept -> signal_index{
        if constexpr (total_counters == 1) {
            select_bias_hint |= nextBias;
            return 0;
        } else {
            static auto constexpr counters_per_half = (total_counters / 2);
            static auto constexpr bits_per_half = (counters_per_half * bits_per_counter);
            static auto constexpr right_bit_mask = ((1ull << bits_per_half) - 1);
            static auto constexpr left_bit_mask = (right_bit_mask << bits_per_half);

            auto const rightCounters = (counters & right_bit_mask);
            auto const leftCounters = (counters & left_bit_mask);
            auto const biasRight = (biasFlags & bias_bit);
            auto chooseRight = ((biasRight && rightCounters) || (leftCounters == 0ull));
            nextBias <<= 1;
            nextBias |= (rightCounters != 0);
            counters >>= (chooseRight) ? 0 : bits_per_half;
            return ((chooseRight) ? counters_per_half : 0) + default_selector<counters_per_half, bits_per_counter, bias_bit / 2>()(biasFlags, counters & right_bit_mask, nextBias);
        }
    }
};


//=====================================================================
// TODO: hack.  write better when have time
template <std::uint64_t N = 64>
consteval static std::uint64_t select_tree_size (const std::uint64_t requested) {
    constexpr std::array<std::uint64_t, 16> valid {
            64,             // 2^6
            64 << 3,        // 2^9
            64 << 5,        // 2^11
            64 << 7,        // 2^13
            64 << 9,        // 2^15
            64 << 11,       // 2^17
            64 << 12,       // 2^18
            64 << 13,       // 2^19
            64 << 14,       // 2^20
            64 << 15,       // 2^21
            64 << 16,       // 2^22
            64 << 17,       // 2^23
            64 << 18,       // 2^24
            64 << 19,       // 2^25
            64 << 20,       // 2^26
            64 << 21        // 2^27
        };
    for (const auto v : valid) {
        if (v >= requested) {
            return v;
        }
    }

    return (0x800000000000ull >> std::countl_zero(requested));
}

//=============================================================================

/// \tparam N1
/// \tparam N2
template<std::size_t N1, std::size_t N2>
    requires(std::popcount(N1) == 1u)
struct node_traits {
	static auto constexpr tree_capacity = N2;
	static auto constexpr capacity = N1;
	static auto constexpr root_node = (tree_capacity == capacity);
	static auto constexpr number_of_counters = sub_counter_arity_v<capacity>;
	static auto constexpr counter_capacity = capacity / number_of_counters;
	static auto constexpr bits_per_counter = minimum_bit_count(counter_capacity);
};

template<typename T>
concept node_traits_concept = std::is_same_v<T, node_traits<T::capacity, T::tree_capacity>>;
template<typename T>
concept leaf_node_traits = ((node_traits_concept<T>) && (T::capacity == 64));
template<typename T>
concept non_leaf_node_traits = ((node_traits_concept<T>) && (!leaf_node_traits<T>) );
template<typename T>
concept root_node_traits = ((node_traits_concept<T>) && (T::root_node));


//=============================================================================
// non leaf nodes ...
// node is a 64-bit integer which represents two (or more) counters
template<node_traits_concept T>
class alignas(64) node final {
public:
	static auto constexpr capacity = T::capacity;
	static auto constexpr tree_capacity = T::tree_capacity;
	static auto constexpr number_of_counters = T::number_of_counters;
	static auto constexpr counter_capacity = T::counter_capacity;
	static auto constexpr bits_per_counter = T::bits_per_counter;
	static auto constexpr counter_mask = (1ull << bits_per_counter) - 1;

	using child_type = node<node_traits<capacity / number_of_counters, tree_capacity>>;
	using bias_flags = std::uint64_t;
	using value_type = std::uint64_t;

	/// \param signalIndex
	/// \return
	inline std::pair<bool, bool> set(const std::uint64_t signalIndex) noexcept {
	    static auto constexpr set_successful = true;
	    auto counterIndex = signalIndex / counter_capacity;

	    if constexpr (root_node_traits<T>) {
	        // root node. return true if setting signal transitioned tree
	        // from empty to non-empty
	        if constexpr (non_leaf_node_traits<T>) {
	            return {(value_.fetch_add(addend_[counterIndex]) == 0ull), set_successful};
	        } else {
	            auto bit = 0x8000000000000000ull >> counterIndex;
	            auto prev = value_.fetch_or(bit);
	            return {prev == 0, (prev & bit) == 0ull};
	        }
	    } else {
	        // not a root node
	        if constexpr (non_leaf_node_traits<T>) {
	            // non-leaf node.  increment correct sub counter
	            value_.fetch_add(addend_[counterIndex]);
	            return {true, set_successful};
	        } else {
	            // leaf node. counters are 1 bit in size
	            // set correct counter bit and return true if not already set
	            auto bit = 0x8000000000000000ull >> counterIndex;
	            return {false, (value_.fetch_or(bit) & bit) == 0ull};
	        }
	    }
	}

	///
	/// \return
	[[nodiscard]] constexpr inline bool empty() const noexcept {
		return (value_ == 0);
	}

	/// \tparam selector
	/// \param biasFlags
	/// \return
	template <template <std::uint64_t, std::uint64_t> class selector>
	inline auto select( const bias_flags biasFlags) noexcept -> std::pair<signal_index, bool> {
	    auto expected = value_.load();
	    while (expected) {
	        auto counterIndex = selector<number_of_counters, bits_per_counter>()(biasFlags, expected);
	        if constexpr (non_leaf_node_traits<T>) {
	            auto desired = expected - addend_[counterIndex];
	            if (value_.compare_exchange_strong(expected, desired))
	                return {counterIndex, (desired == 0)};
	        }else {
	            auto bit = 0x8000000000000000ull >> counterIndex;
	            if (expected = value_.fetch_and(~bit); ((expected & bit) == bit))
	                return {counterIndex, (expected == bit)};
	        }
	    }
	    return {invalid_signal_index, false};
	}
protected:
	std::atomic<value_type> value_{0};

	static std::array<std::uint64_t, number_of_counters> constexpr addend_{
	        []<std::size_t... N>(std::index_sequence<N...>) -> std::array<std::uint64_t, number_of_counters> {
		        return {(1ull << ((number_of_counters - N - 1) * bits_per_counter))...};
	        }(std::make_index_sequence<number_of_counters>())};
};


//=============================================================================

/// \tparam N0
/// \tparam N1
template<std::size_t N0, std::size_t N1>
    requires((std::popcount(N0) == 1) && (std::popcount(N1) == 1))
struct level_traits {
	static auto constexpr number_of_nodes = N0;
	static auto constexpr node_capacity = N1;
	static auto constexpr tree_capacity = (node_capacity * number_of_nodes);
};


template<typename T>
concept level_traits_concept = std::is_same_v<T, level_traits<T::number_of_nodes, T::node_capacity>>;
template<typename T>
concept root_level_traits = ((level_traits_concept<T>) && (T::number_of_nodes == 1));
template<typename T>
concept leaf_level_traits = ((level_traits_concept<T>) && (T::node_capacity == 64));
template<typename T>
concept non_leaf_level_traits = ((level_traits_concept<T>) && (!leaf_level_traits<T>) );


template<level_traits_concept T>
struct level;


template<level_traits_concept T>
struct child_level {
	using type = struct {};
};


template<non_leaf_level_traits T>
struct child_level<T> {
	static auto constexpr number_of_child_nodes = T::number_of_nodes * node<node_traits<T::node_capacity, T::tree_capacity>>::number_of_counters;
	using type = level<level_traits<number_of_child_nodes, node<node_traits<T::node_capacity, T::tree_capacity>>::child_type::capacity>>;
};


//=============================================================================
template<level_traits_concept T>
class alignas(64) level final {
public:
	using node_type = node<node_traits<T::node_capacity, T::tree_capacity>>;
	using child_level_type = child_level<T>::type;
	using bias_flags = std::uint64_t;
	using node_index = std::uint64_t;

	/// \return
	[[nodiscard]] constexpr bool empty() const noexcept
	    requires(root_level_traits<T>) {
		return nodes_[0].empty();
	}

	/// increment the counter associated with the specified index (id of a leaf node)
	/// return true if this set caused the level to move from empty to non-empty
	/// return false otherwise.
	std::pair<bool, bool> set(signal_index signalIndex) noexcept {
	    if constexpr (non_leaf_level_traits<T>){
	        if (auto [_, success] = childLevel_.set(signalIndex); !success)
	            return {false, false};
	    }
	    return nodes_[signalIndex / node_capacity].set(signalIndex % node_capacity);
	}

	/// return the index of a counter which is not zero (indicates that one of the leaf nodes
	/// represented by this counter is set - a non-zero value
	/// \tparam select_function
	/// \param biasFlags
	/// \return
	template<template<std::uint64_t, std::uint64_t> class select_function>
	std::pair<signal_index, bool> select(bias_flags biasFlags) noexcept
	    requires(root_level_traits<T>) {
	    return select<select_function>(biasFlags, 0);
	}

protected:
	static auto constexpr node_capacity = T::node_capacity;
	static auto constexpr node_count = T::number_of_nodes;
	static auto constexpr counters_per_node = node_type::number_of_counters;
	static auto constexpr bits_per_counter = node_type::bits_per_counter;
	static auto constexpr counter_capacity = (node_capacity / counters_per_node);

	template<level_traits_concept>
	friend struct level;

	/// \tparam select_function
	/// \param biasFlags
	/// \param nodeIndex
	/// \return
	template<template<std::uint64_t, std::uint64_t> class select_function>
	std::pair<signal_index, bool> select(bias_flags biasFlags,
										 node_index nodeIndex) noexcept {
	    static auto constexpr bias_bits_consumed_to_select_counter = minimum_bit_count(counters_per_node) - 1;  // was node_count

	    auto [selectedCounter, nodeIsZero] = nodes_[nodeIndex]. template select<select_function>(biasFlags);
	    biasFlags <<= bias_bits_consumed_to_select_counter;

	    if constexpr (root_level_traits<T>) {
	        if (selectedCounter == invalid_signal_index)
	            return {invalid_signal_index, false};
	    }

	    if constexpr (leaf_level_traits<T>) {
		    return {selectedCounter, nodeIsZero};
	    } else {
	        static auto constexpr bias_bits_consumed_to_select_child_counter = minimum_bit_count(child_level_type::counters_per_node) - 1;
	        select_bias_hint <<= bias_bits_consumed_to_select_child_counter;
	        auto [childSelectedCounter, _] = childLevel_. template select<select_function>(biasFlags, (nodeIndex * counters_per_node) + selectedCounter);
	        selectedCounter *= counter_capacity;
	        return {selectedCounter | childSelectedCounter, nodeIsZero};
	    }
	}

	using node_array = std::array<node_type, node_count>;
	using iterator = node_array::iterator;

	node_array nodes_;

	child_level_type childLevel_;
};


//=============================================================================
template <std::uint64_t N>
class tree final
{
private:
    using root_level_traits = level_traits<1, N>;
    using root_level = level<root_level_traits>;

public:

    static auto constexpr capacity = N;
    static_assert(select_tree_size(capacity) == capacity,
                    "invalid signal_tree capacity");

    // set the leaf node associated with the index to 1
    inline std::pair<bool, bool> set(const signal_index signalIndex) noexcept {
        return rootLevel_.set(signalIndex);
    }

	/// \return
	[[nodiscard]] constexpr inline bool empty() const noexcept {
        return rootLevel_.empty();
    }

	/// \tparam select_function
	/// \param bias
	/// \return
	template <template <std::uint64_t, std::uint64_t> class select_function = default_selector>
    std::pair<signal_index, bool> select (std::uint64_t bias) noexcept {
        static auto constexpr number_of_bias_bits = (65 - minimum_bit_count(capacity));
        bias <<= number_of_bias_bits;
        return rootLevel_. template select<select_function>(bias);
    }

private:

    root_level rootLevel_;
}; // class tree

//
template <std::size_t N>
using signal_tree = tree<select_tree_size(N)>;
