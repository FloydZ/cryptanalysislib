#ifndef CRYPTANALYSISLIB_KHASH_H
#define CRYPTANALYSISLIB_KHASH_H

#if !defined(CRYPTANALYSISLIB_HASHMAP_H)
#error "Do not include this file directly. Use: `#include <container/hashmap.h>`"
#endif

#include <array>
#include <concepts>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

namespace cryptanalysislib {

template<typename T>
concept Hashable = requires(T t) {
    { std::hash<T>{}(t) } -> std::convertible_to<std::size_t>;
};

template<typename K, typename V>
concept KeyValuePair = requires {
    typename K;
    typename V;
} && Hashable<K>;

enum class BucketState : std::uint8_t {
    Empty = 0,
    Occupied = 1,
    Deleted = 2
};

template<Hashable Key, typename Value, typename Hash = std::hash<Key>, typename Equal = std::equal_to<Key>>
class KHash {
public:
    using key_type = Key;
    using mapped_type = Value;
    using value_type = std::pair<const Key, Value>;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;
    using hasher = Hash;
    using key_equal = Equal;
    using reference = value_type&;
    using const_reference = const value_type&;
    using pointer = value_type*;
    using const_pointer = const value_type*;

private:
    static constexpr double max_load_factor_ = 0.77;
    static constexpr size_type min_buckets = 4;

    size_type n_buckets_;
    size_type size_;
    size_type n_occupied_;
    size_type upper_bound_;
    
    std::unique_ptr<BucketState[]> flags_;
    std::unique_ptr<Key[]> keys_;
    std::unique_ptr<Value[]> vals_;
    
    [[no_unique_address]] Hash hasher_;
    [[no_unique_address]] Equal equal_;

    [[nodiscard]] constexpr bool is_either(size_type i) const noexcept {
        return flags_[i] != BucketState::Occupied;
    }
    
    [[nodiscard]] constexpr bool is_empty(size_type i) const noexcept {
        return flags_[i] == BucketState::Empty;
    }
    
    [[nodiscard]] constexpr bool is_del(size_type i) const noexcept {
        return flags_[i] == BucketState::Deleted;
    }

    constexpr size_type hash_func(const Key& key) const noexcept {
        return hasher_(key) & (n_buckets_ - 1);
    }

    constexpr void resize(size_type new_n_buckets) {
        auto old_flags = std::move(flags_);
        auto old_keys = std::move(keys_);
        auto old_vals = std::move(vals_);
        size_type old_n_buckets = n_buckets_;

        n_buckets_ = new_n_buckets;
        upper_bound_ = static_cast<size_type>(n_buckets_ * max_load_factor_);
        n_occupied_ = size_ = 0;

        flags_ = std::make_unique<BucketState[]>(n_buckets_);
        keys_ = std::make_unique<Key[]>(n_buckets_);
        vals_ = std::make_unique<Value[]>(n_buckets_);

        std::fill_n(flags_.get(), n_buckets_, BucketState::Empty);

        if (old_n_buckets == 0) return;

        for (size_type j = 0; j != old_n_buckets; ++j) {
            if (old_flags[j] == BucketState::Occupied) {
                auto [pos, ret] = put_internal(std::move(old_keys[j]));
                if (ret) {
                    vals_[pos] = std::move(old_vals[j]);
                }
            }
        }
    }

    constexpr std::pair<size_type, bool> put_internal(Key&& key) {
        if (n_occupied_ >= upper_bound_) {
            if (n_buckets_ > (size_ << 1)) {
                resize(n_buckets_ - 1);
            } else {
                resize(n_buckets_ ? n_buckets_ << 1 : min_buckets);
            }
        }

        size_type mask = n_buckets_ - 1;
        size_type i = hash_func(key);
        size_type last = i;
        
        while (!is_empty(i) && 
               (is_del(i) || !equal_(keys_[i], key))) {
            i = (i + 1) & mask;
            if (i == last) {
                resize(n_buckets_ << 1);
                return put_internal(std::move(key));
            }
        }

        bool is_bucket_empty = is_empty(i);
        bool is_bucket_del = is_del(i);
        
        if (is_bucket_empty || is_bucket_del) {
            keys_[i] = std::move(key);
            flags_[i] = BucketState::Occupied;
            if (is_bucket_empty) n_occupied_++;
            size_++;
            return {i, true};
        }
        
        return {i, false};
    }

public:
    class iterator {
    private:
        KHash* map_;
        size_type pos_;

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = KHash::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;

        constexpr iterator(KHash* map, size_type pos) noexcept 
            : map_(map), pos_(pos) {
            while (pos_ < map_->n_buckets_ && map_->flags_[pos_] != BucketState::Occupied) {
                ++pos_;
            }
        }

        constexpr value_type operator*() const noexcept {
            return {map_->keys_[pos_], map_->vals_[pos_]};
        }

        constexpr iterator& operator++() noexcept {
            do {
                ++pos_;
            } while (pos_ < map_->n_buckets_ && map_->flags_[pos_] != BucketState::Occupied);
            return *this;
        }

        constexpr iterator operator++(int) noexcept {
            iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        constexpr bool operator==(const iterator& other) const noexcept = default;
    };

    class const_iterator {
    private:
        const KHash* map_;
        size_type pos_;

    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = KHash::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = const value_type*;
        using reference = const value_type&;

        constexpr const_iterator(const KHash* map, size_type pos) noexcept 
            : map_(map), pos_(pos) {
            while (pos_ < map_->n_buckets_ && map_->flags_[pos_] != BucketState::Occupied) {
                ++pos_;
            }
        }

        constexpr value_type operator*() const noexcept {
            return {map_->keys_[pos_], map_->vals_[pos_]};
        }

        constexpr const_iterator& operator++() noexcept {
            do {
                ++pos_;
            } while (pos_ < map_->n_buckets_ && map_->flags_[pos_] != BucketState::Occupied);
            return *this;
        }

        constexpr const_iterator operator++(int) noexcept {
            const_iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        constexpr bool operator==(const const_iterator& other) const noexcept = default;
    };

    constexpr KHash() noexcept 
        : n_buckets_(0), size_(0), n_occupied_(0), upper_bound_(0) {}

    explicit constexpr KHash(size_type bucket_count, 
                            const Hash& hash = Hash{}, 
                            const Equal& equal = Equal{}) 
        : hasher_(hash), equal_(equal) {
        size_type new_size = min_buckets;
        while (new_size < bucket_count) new_size <<= 1;
        resize(new_size);
    }

    constexpr ~KHash() = default;
    constexpr KHash(const KHash& other) = delete;
    constexpr KHash& operator=(const KHash& other) = delete;
    
    constexpr KHash(KHash&& other) noexcept 
        : n_buckets_(std::exchange(other.n_buckets_, 0))
        , size_(std::exchange(other.size_, 0))
        , n_occupied_(std::exchange(other.n_occupied_, 0))
        , upper_bound_(std::exchange(other.upper_bound_, 0))
        , flags_(std::move(other.flags_))
        , keys_(std::move(other.keys_))
        , vals_(std::move(other.vals_))
        , hasher_(std::move(other.hasher_))
        , equal_(std::move(other.equal_)) {}

    constexpr KHash& operator=(KHash&& other) noexcept {
        if (this != &other) {
            n_buckets_ = std::exchange(other.n_buckets_, 0);
            size_ = std::exchange(other.size_, 0);
            n_occupied_ = std::exchange(other.n_occupied_, 0);
            upper_bound_ = std::exchange(other.upper_bound_, 0);
            flags_ = std::move(other.flags_);
            keys_ = std::move(other.keys_);
            vals_ = std::move(other.vals_);
            hasher_ = std::move(other.hasher_);
            equal_ = std::move(other.equal_);
        }
        return *this;
    }

    [[nodiscard]] constexpr iterator begin() noexcept {
        return iterator(this, 0);
    }

    [[nodiscard]] constexpr const_iterator begin() const noexcept {
        return const_iterator(this, 0);
    }

    [[nodiscard]] constexpr const_iterator cbegin() const noexcept {
        return begin();
    }

    [[nodiscard]] constexpr iterator end() noexcept {
        return iterator(this, n_buckets_);
    }

    [[nodiscard]] constexpr const_iterator end() const noexcept {
        return const_iterator(this, n_buckets_);
    }

    [[nodiscard]] constexpr const_iterator cend() const noexcept {
        return end();
    }

    [[nodiscard]] constexpr bool empty() const noexcept {
        return size_ == 0;
    }

    [[nodiscard]] constexpr size_type size() const noexcept {
        return size_;
    }

    [[nodiscard]] constexpr size_type bucket_count() const noexcept {
        return n_buckets_;
    }

    [[nodiscard]] constexpr double load_factor() const noexcept {
        return n_buckets_ ? static_cast<double>(size_) / n_buckets_ : 0.0;
    }

    [[nodiscard]] constexpr double max_load_factor() const noexcept {
        return max_load_factor_;
    }

    constexpr void clear() noexcept {
        if (n_buckets_) {
            std::fill_n(flags_.get(), n_buckets_, BucketState::Empty);
            size_ = n_occupied_ = 0;
        }
    }

    constexpr size_type get(const Key& key) const noexcept {
        if (n_buckets_ == 0) return n_buckets_;
        
        size_type mask = n_buckets_ - 1;
        size_type i = hash_func(key);
        size_type last = i;
        
        while (!is_empty(i) && 
               (is_del(i) || !equal_(keys_[i], key))) {
            i = (i + 1) & mask;
            if (i == last) return n_buckets_;
        }
        
        return is_either(i) ? n_buckets_ : i;
    }

    constexpr std::pair<size_type, bool> put(const Key& key) {
        return put_internal(Key{key});
    }

    constexpr std::pair<size_type, bool> put(Key&& key) {
        return put_internal(std::move(key));
    }

    constexpr void del(size_type pos) noexcept {
        if (pos != n_buckets_ && flags_[pos] == BucketState::Occupied) {
            flags_[pos] = BucketState::Deleted;
            --size_;
        }
    }

    constexpr std::pair<iterator, bool> insert(const value_type& value) {
        auto [pos, inserted] = put(value.first);
        if (inserted) {
            vals_[pos] = value.second;
        }
        return {iterator(this, pos), inserted};
    }

    constexpr std::pair<iterator, bool> insert(value_type&& value) {
        auto [pos, inserted] = put(std::move(const_cast<Key&>(value.first)));
        if (inserted) {
            vals_[pos] = std::move(value.second);
        }
        return {iterator(this, pos), inserted};
    }

    template<typename... Args>
    constexpr std::pair<iterator, bool> emplace(Args&&... args) {
        value_type value(std::forward<Args>(args)...);
        return insert(std::move(value));
    }

    constexpr iterator find(const Key& key) noexcept {
        size_type pos = get(key);
        return iterator(this, pos);
    }

    constexpr const_iterator find(const Key& key) const noexcept {
        size_type pos = get(key);
        return const_iterator(this, pos);
    }

    constexpr size_type count(const Key& key) const noexcept {
        return find(key) != end() ? 1 : 0;
    }

    constexpr bool contains(const Key& key) const noexcept {
        return find(key) != end();
    }

    constexpr iterator erase(const_iterator pos) {
        size_type bucket_pos = pos.pos_;
        del(bucket_pos);
        return iterator(this, bucket_pos);
    }

    constexpr size_type erase(const Key& key) {
        auto it = find(key);
        if (it != end()) {
            erase(it);
            return 1;
        }
        return 0;
    }

    constexpr Value& operator[](const Key& key) {
        auto [pos, inserted] = put(key);
        if (inserted) {
            vals_[pos] = Value{};
        }
        return vals_[pos];
    }

    constexpr Value& operator[](Key&& key) {
        auto [pos, inserted] = put(std::move(key));
        if (inserted) {
            vals_[pos] = Value{};
        }
        return vals_[pos];
    }

    constexpr Value& at(const Key& key) {
        size_type pos = get(key);
        if (pos == n_buckets_) {
            throw std::out_of_range("KHash::at: key not found");
        }
        return vals_[pos];
    }

    constexpr const Value& at(const Key& key) const {
        size_type pos = get(key);
        if (pos == n_buckets_) {
            throw std::out_of_range("KHash::at: key not found");
        }
        return vals_[pos];
    }

    constexpr void reserve(size_type count) {
        size_type new_buckets = min_buckets;
        while (new_buckets < count / max_load_factor_) {
            new_buckets <<= 1;
        }
        if (new_buckets > n_buckets_) {
            resize(new_buckets);
        }
    }

    constexpr void rehash(size_type bucket_count) {
        size_type new_buckets = min_buckets;
        while (new_buckets < bucket_count) {
            new_buckets <<= 1;
        }
        resize(new_buckets);
    }

    friend constexpr bool operator==(const KHash& lhs, const KHash& rhs) noexcept {
        if (lhs.size() != rhs.size()) return false;
        
        for (auto it = lhs.begin(); it != lhs.end(); ++it) {
            auto rhs_it = rhs.find(it->first);
            if (rhs_it == rhs.end() || rhs_it->second != it->second) {
                return false;
            }
        }
        return true;
    }
};

template<Hashable Key, typename Hash = std::hash<Key>, typename Equal = std::equal_to<Key>>
using KHashSet = KHash<Key, std::monostate, Hash, Equal>;

template<Hashable Key, typename Value, typename Hash = std::hash<Key>, typename Equal = std::equal_to<Key>>
using KHashMap = KHash<Key, Value, Hash, Equal>;

namespace detail {
    constexpr std::uint32_t kh_int_hash(std::uint32_t key) noexcept {
        key += ~(key << 15);
        key ^= (key >> 10);
        key += (key << 3);
        key ^= (key >> 6);
        key += ~(key << 11);
        key ^= (key >> 16);
        return key;
    }

    constexpr std::uint32_t kh_int64_hash(std::uint64_t key) noexcept {
        key = (~key) + (key << 18);
        key = key ^ (key >> 31);
        key = key * 21;
        key = key ^ (key >> 11);
        key = key + (key << 6);
        key = key ^ (key >> 22);
        return static_cast<std::uint32_t>(key);
    }

    constexpr std::uint32_t kh_str_hash(const char* s) noexcept {
        std::uint32_t h = 0;
        while (*s) {
            h = (h << 5) - h + static_cast<std::uint8_t>(*s++);
        }
        return h;
    }
}

struct KHashIntHash {
    constexpr std::size_t operator()(std::uint32_t key) const noexcept {
        return detail::kh_int_hash(key);
    }
    
    constexpr std::size_t operator()(std::uint64_t key) const noexcept {
        return detail::kh_int64_hash(key);
    }
};

struct KHashStrHash {
    constexpr std::size_t operator()(const char* s) const noexcept {
        return detail::kh_str_hash(s);
    }
    
    constexpr std::size_t operator()(const std::string& s) const noexcept {
        return detail::kh_str_hash(s.c_str());
    }
};

using khash_int_t = KHashMap<std::uint32_t, std::uint32_t, KHashIntHash>;
using khash_int64_t = KHashMap<std::uint64_t, std::uint64_t, KHashIntHash>;
using khash_str_t = KHashMap<std::string, std::string, KHashStrHash>;

using khash_set_int_t = KHashSet<std::uint32_t, KHashIntHash>;
using khash_set_int64_t = KHashSet<std::uint64_t, KHashIntHash>;
using khash_set_str_t = KHashSet<std::string, KHashStrHash>;

} // namespace cryptanalysislib

#endif // CRYPTANALYSISLIB_KHASH_H