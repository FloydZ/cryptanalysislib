#ifndef CRYPTANALYSISLIB_CONTAINER_QUEUE_H
#define CRYPTANALYSISLIB_CONTAINER_QUEUE_H

#include <atomic>
#include <cstdint>
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

// TODO
template <typename Type>
class queue
// Implementation of a queue
{
public:
    Type *x_;   // pointer to data
    ulong s_;   // allocated size (# of elements)
    ulong n_;   // current number of entries in buffer
    ulong wpos_;  // next position to write in buffer
    ulong rpos_;  // next position to read in buffer
    ulong gq_;  // grow gq elements if necessary, 0 for "never grow"

    queue(const queue&) = delete;
    queue & operator = (const queue&) = delete;

public:
    explicit queue(ulong n, ulong growq=0)
    {
        s_ = n;
//        x_ = new Type[s_];
        x_ = (Type *)std::malloc( s_ * sizeof(Type) );
        n_ = 0;
        wpos_ = 0;
        rpos_ = 0;
        gq_ = growq;
    }

    ~queue()  { std::free( x_ ); }

    ulong num()  const  { return n_; }

    ulong push(const Type &z)
    // Return number of entries.
    // Zero is returned on failure
    //   (i.e. space exhausted and 0==gq_)
    {
        if ( n_ >= s_ )
        {
            if ( 0==gq_ )  return 0;  // growing disabled
            grow();
        }

        x_[wpos_] = z;
        ++wpos_;
        if ( wpos_>=s_ )  wpos_ = 0;

        ++n_;
        return n_;
    }

    ulong peek(Type &z)
    // Return number of entries.
    // if zero is returned the value of z is undefined.
    {
        z = x_[rpos_];
        return n_;
    }

    ulong pop(Type &z)
    // Return number of entries before pop
    // i.e. zero is returned if queue was empty.
    // If zero is returned the value of z is undefined.
    {
        ulong ret = n_;
        if ( 0!=n_ )
        {
            z = x_[rpos_];
            ++rpos_;
            if ( rpos_ >= s_ )  rpos_ = 0;
            --n_;
        }

        return ret;
    }

private:
    void grow()
    {
        ulong ns = s_ + gq_;  // new size
        // move read-position to zero:
        rotate_left(x_, s_, rpos_);
        x_ = ReAlloc<Type>(x_, ns, s_);
        wpos_ = s_;
        rpos_ = 0;
        s_ = ns;
    }
};
#endif//CRYPTANALYSISLIB_QUEUE_H
