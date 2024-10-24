#ifndef CRYPTANALYSISLIB_QUEUE_H
#define CRYPTANALYSISLIB_QUEUE_H

#include <atomic>
#include <cstdint>
#include <limits>
#include <queue>
#include <mutex>
#include <optional>

#include "atomic/atomic_primitives.h"

/// taken from: https://github.com/codecryptanalysis/mccl/blob/main/mccl/core/collection.hpp
/// multi consumer multi producer unbounded queue
/// implemented as simple wrapper around std::deque
/// \tparam T
/// \tparam Mutex
template<typename T,
         typename Mutex = std::mutex>
class concurrent_queue {
public:
	typedef Mutex mutex_type;
	typedef std::lock_guard<mutex_type> lock_type;
	typedef std::deque<T> queue_type;

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


template <typename T, typename Lock = std::mutex>
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
#endif//CRYPTANALYSISLIB_QUEUE_H
