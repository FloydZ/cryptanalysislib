#ifndef CRYPTANALYSISLIB_QUEUE_H
#define CRYPTANALYSISLIB_QUEUE_H

#include <atomic>
#include <limits>
#include <queue>

/// taken from: https://github.com/codecryptanalysis/mccl/blob/main/mccl/core/collection.hpp
/// multi consumer multi producer unbounded queue
/// implemented as simple wrapper around std::deque
/// \tparam T
/// \tparam Mutex
template<typename T, typename Mutex = std::mutex>
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
#endif//CRYPTANALYSISLIB_QUEUE_H
