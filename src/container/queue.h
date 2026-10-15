#ifndef CRYPTANALYSISLIB_CONTAINER_QUEUE_H
#define CRYPTANALYSISLIB_CONTAINER_QUEUE_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <limits>
#include <queue>
#include <mutex>
#include <optional>

// NOTE: cannot import this, as the allocator depends on it xD
// #include "alloc/alloc.h"
#include "atomic/atomic_primitives.h"


/// taken from: https://github.com/codecryptanalysis/mccl/blob/main/mccl/core/collection.hpp
/// multi consumer multi producer unbounded queue
/// implemented as simple wrapper around std::deque
/// \tparam T
/// \tparam Mutex
template<typename T,
         typename Mutex = std::mutex,
         typename Allocator = std::allocator<T>>
class concurrent_queue {
public:
	typedef Mutex mutex_type;
	typedef std::lock_guard<mutex_type> lock_type;
	typedef std::deque<T, Allocator> queue_type;

	typedef T value_type;

	constexpr concurrent_queue() noexcept {}
	constexpr ~concurrent_queue() noexcept {}

	///
	constexpr inline std::size_t size() noexcept {
		lock_type lock(_mutex);
		return _queue.size();
	}

	///
	constexpr inline bool empty() noexcept {
		lock_type lock(_mutex);
		return _queue.empty();
	}

	///
	constexpr inline void push_back(const value_type &v) noexcept {
		_emplace_back(v);
	}

	///
	constexpr inline void push_back(value_type &&v) noexcept {
		_emplace_back(std::move(v));
	}

	///
	template<typename... Args>
	constexpr inline void emplace_back(Args &&...args) noexcept {
		_emplace_back(std::forward<Args>(args)...);
	}

	constexpr inline bool try_pop_front(value_type &v) noexcept {
		lock_type lock(_mutex);
		if (_queue.empty()) {
			return false;
		}

		v = std::move(_queue.front());
		_queue.pop_front();
		return true;
	}

private:
	template<typename... Args>
	constexpr inline void _emplace_back(Args &&...args) noexcept {
		lock_type lock(_mutex);
		_queue.emplace_back(std::forward<Args>(args)...);
	}

	mutex_type _mutex;
	queue_type _queue;
};


/// \tparam T
/// \tparam Lock
template <typename T,
          typename Lock = std::mutex>
    requires is_lockable<Lock>
class thread_safe_queue {
public:
	using value_type = T;
	using size_type = typename std::deque<T>::size_type;

	thread_safe_queue() = default;

    ///
	void push_back(T&& value) noexcept {
		std::scoped_lock lock(mutex_);
		data_.push_back(std::forward<T>(value));
	}

    ///
	void push_front(T&& value) noexcept {
		std::scoped_lock lock(mutex_);
		data_.push_front(std::forward<T>(value));
	}

    ///
	[[nodiscard]] bool empty() const noexcept {
		std::scoped_lock lock(mutex_);
		return data_.empty();
	}

    ///
	size_type clear() noexcept {
		std::scoped_lock lock(mutex_);
		auto size = data_.size();
		data_.clear();

		return size;
	}

    ///
	[[nodiscard]] std::optional<T> pop_front() noexcept {
		std::scoped_lock lock(mutex_);
		if (data_.empty()) { return std::nullopt; }

		auto front = std::move(data_.front());
		data_.pop_front();
		return front;
	}

    ///
	[[nodiscard]] std::optional<T> pop_back() noexcept {
		std::scoped_lock lock(mutex_);
		if (data_.empty()) { return std::nullopt; }

		auto back = std::move(data_.back());
		data_.pop_back();
		return back;
	}

    ///
	[[nodiscard]] std::optional<T> steal() noexcept {
		std::scoped_lock lock(mutex_);
		if (data_.empty()) { return std::nullopt; }

		auto back = std::move(data_.back());
		data_.pop_back();
		return back;
	}
    
    /// 
	void rotate_to_front(const T& item) noexcept {
		std::scoped_lock lock(mutex_);
		auto iter = std::find(data_.begin(), data_.end(), item);

		if (iter != data_.end()) {
			std::ignore = data_.erase(iter);
		}

		data_.push_front(item);
	}

    ///
	[[nodiscard]] std::optional<T> copy_front_and_rotate_to_back() noexcept {
		std::scoped_lock lock(mutex_);

		if (data_.empty()) return std::nullopt;

		auto front = data_.front();
		data_.pop_front();
		data_.push_back(front);
		return front;
	}

private:
	std::deque<T> data_{};
	mutable Lock mutex_{};
};

namespace cryptanalysislib {
/// FIFO ring buffer queue (from the fxtbook), optionally growing by `gq`
/// elements when full.
/// \tparam Type trivially copyable element type (the buffer holds no
/// 	constructed objects)
/// \tparam Allocator allocator of the buffer (STL interface, assumed
/// 	stateless). The default `std::allocator` is `cryptanalysislib::allocator`,
/// 	which cannot be named here, as `alloc/alloc.h` includes this header.
template <typename Type,
          template<class> class Allocator = std::allocator>
class queue {
	static_assert(std::is_trivially_copyable_v<Type>);
	using allocator_type = Allocator<Type>;
	using traits = std::allocator_traits<allocator_type>;
public:
    Type *x_;   // pointer to data
    uint64_t s_;   // allocated size (# of elements)
    uint64_t n_;   // current number of entries in buffer
    uint64_t wpos_;  // next position to write in buffer
    uint64_t rpos_;  // next position to read in buffer
    uint64_t gq_;  // grow gq elements if necessary, 0 for "never grow"

    queue(const queue&) = delete;
    queue & operator = (const queue&) = delete;

public:
    /// \param n initial capacity (> 0)
    /// \param growq number of elements to grow by if full, 0: never grow
    explicit queue(uint64_t n, uint64_t growq=0) noexcept {
        allocator_type a{};
        s_ = n;
        x_ = traits::allocate(a, s_);
        n_ = 0;
        wpos_ = 0;
        rpos_ = 0;
        gq_ = growq;
    }

    ~queue() noexcept {
        allocator_type a{};
        traits::deallocate(a, x_, s_);
    }

    /// \return number of entries
    uint64_t num() const noexcept { return n_; }

    /// appends `z`
    /// \return number of entries, 0 on failure (full and 0 == gq_)
    uint64_t push(const Type &z) noexcept {
        if ( n_ >= s_ ) {
            if ( 0==gq_ )  return 0;  // growing disabled
            if (!grow()) { return 0; }
        }

        x_[wpos_] = z;
        ++wpos_;
        if ( wpos_>=s_ )  wpos_ = 0;

        ++n_;
        return n_;
    }

    /// reads the oldest entry without removing it
    /// \return number of entries; if zero, the value of z is undefined
    uint64_t peek(Type &z) const noexcept {
        if (n_) { z = x_[rpos_]; }
        return n_;
    }

    /// removes the oldest entry
    /// \return number of entries before the pop, i.e. zero if the queue
    /// 	was empty (then `z` is unchanged)
    uint64_t pop(Type &z) noexcept {
        uint64_t ret = n_;
        if ( 0!=n_ ) {
            z = x_[rpos_];
            ++rpos_;
            if ( rpos_ >= s_ )  rpos_ = 0;
            --n_;
        }

        return ret;
    }

private:
    /// \return false if the allocation failed (the queue is unchanged)
    bool grow() noexcept {
        const uint64_t ns = s_ + gq_;  // new size
        allocator_type a{};
        Type *nx = traits::allocate(a, ns);
        if (nx == nullptr) { return false; }
        // the queue is full here, i.e. the entries are x_[rpos_..s_)
        // followed by x_[0..rpos_): copy them in this order to the front
        for (uint64_t i = 0; i < s_; ++i) {
            const uint64_t j = rpos_ + i;
            nx[i] = x_[(j < s_) ? j : (j - s_)];
        }
        traits::deallocate(a, x_, s_);
        x_ = nx;
        wpos_ = s_;
        rpos_ = 0;
        s_ = ns;
        return true;
    }
};
} // end namespace cryptanalysislib
#endif//CRYPTANALYSISLIB_QUEUE_H
