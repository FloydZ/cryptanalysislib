#pragma once

#include <type_traits>
#include <concepts>
#include <array>

 template <std::size_t N0, std::size_t N1>
    requires ((std::popcount(N0) == 1) && (std::popcount(N1) == 1))
    struct level_traits
    {
        static auto constexpr number_of_nodes = N0;
        static auto constexpr node_capacity = N1;
        static auto constexpr tree_capacity = (node_capacity * number_of_nodes);
    };


    template <typename T> concept level_traits_concept = std::is_same_v<T, level_traits<T::number_of_nodes, T::node_capacity>>;
    template <typename T> concept root_level_traits = ((level_traits_concept<T>) && (T::number_of_nodes == 1));
    template <typename T> concept leaf_level_traits = ((level_traits_concept<T>) && (T::node_capacity == 64));
    template <typename T> concept non_leaf_level_traits = ((level_traits_concept<T>) && (!leaf_level_traits<T>));


    template <level_traits_concept T>
    struct level;


    template <level_traits_concept T>
    struct child_level{using type = struct{};};


    template <non_leaf_level_traits T>
    struct child_level<T>
    {
        static auto constexpr number_of_child_nodes = T::number_of_nodes * node<node_traits<T::node_capacity, T::tree_capacity>>::number_of_counters;
        using type = level<level_traits<number_of_child_nodes, node<node_traits<T::node_capacity, T::tree_capacity>>::child_type::capacity>>;
    };


    //=============================================================================
    template <level_traits_concept T>
    class alignas(64) level final
    {
    public:

        using node_type = node<node_traits<T::node_capacity, T::tree_capacity>>;
        using child_level_type = child_level<T>::type;
        using bias_flags = std::uint64_t;
        using node_index = std::uint64_t;

        bool empty() const noexcept requires (root_level_traits<T>);

        std::pair<bool, bool> set
        (
            signal_index
        ) noexcept;

        template <template <std::uint64_t, std::uint64_t> class>
        std::pair<signal_index, bool> select
        (
            bias_flags
        ) noexcept requires (root_level_traits<T>);

    protected:

        static auto constexpr node_capacity = T::node_capacity;
        static auto constexpr node_count = T::number_of_nodes;
        static auto constexpr counters_per_node = node_type::number_of_counters;
        static auto constexpr bits_per_counter = node_type::bits_per_counter;
        static auto constexpr counter_capacity = (node_capacity / counters_per_node);

        template <level_traits_concept> friend struct level;

        template <template <std::uint64_t, std::uint64_t> class>
        std::pair<signal_index, bool> select
        (
            bias_flags,
            node_index
        ) noexcept;

        using node_array = std::array<node_type, node_count>;
        using iterator = node_array::iterator;

        node_array          nodes_;

        child_level_type    childLevel_;
    };