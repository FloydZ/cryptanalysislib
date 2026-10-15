#pragma once

#include <atomic>
#include <concepts>
#include <cstddef>
#include <vector>

#include "traits.h"
#include "alloc/alloc.h"

/// \tparam T
template<typename T,
          template<class N> class Allocator = cryptanalysislib::allocator>
class spsc_fixed_queue : non_copyable {
private:
	using type = T;
	using value_type = T;

	/// NOTE: `front_` and `back_` grow monotonically, so the position in
	///		the ring buffer is `index & capacityMask_`
	template<class Q, class R>
	struct Iterator {
		using iterator_category = std::forward_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using value_type = T;
		using reference = R;

		constexpr Iterator(Q *q, const std::size_t i) noexcept : q_(q), i_(i) {}
		constexpr reference operator*() const noexcept { return q_->queue_[i_ & q_->capacityMask_]; }
		constexpr Iterator &operator++() noexcept { i_ += 1; return *this; }
		constexpr Iterator operator++(int) noexcept { Iterator t = *this; i_ += 1; return t; }
		constexpr friend bool operator==(const Iterator &a, const Iterator &b) noexcept { return a.i_ == b.i_; }
		constexpr friend bool operator!=(const Iterator &a, const Iterator &b) noexcept { return a.i_ != b.i_; }

	private:
		Q *q_;
		std::size_t i_;
	};

public:
	[[nodiscard]] constexpr inline auto begin() noexcept { return Iterator<spsc_fixed_queue, T &>(this, front_.load(std::memory_order_acquire)); }
	[[nodiscard]] constexpr inline auto end() noexcept { return Iterator<spsc_fixed_queue, T &>(this, back_.load(std::memory_order_acquire)); }
	[[nodiscard]] constexpr inline auto begin() const noexcept { return Iterator<const spsc_fixed_queue, const T &>(this, front_.load(std::memory_order_acquire)); }
	[[nodiscard]] constexpr inline auto end() const noexcept { return Iterator<const spsc_fixed_queue, const T &>(this, back_.load(std::memory_order_acquire)); }

	/// NOTE: always rounds up to the next power of two.
	/// \param capacity
	constexpr explicit spsc_fixed_queue(const std::size_t capacity) noexcept :
	    front_(0), back_(0){
	    capacity_ = 1;
	    while (capacity_ < capacity) {
		    capacity_ <<= 1;
	    }

	    capacityMask_ = capacity_ - 1;
	    queue_.resize(capacity_);
	}

	spsc_fixed_queue(spsc_fixed_queue &&) = default;
	spsc_fixed_queue &operator=(spsc_fixed_queue &&) = default;
	~spsc_fixed_queue() = default;

	///
	/// \return
	constexpr inline auto pop() noexcept -> type {
	    std::size_t front = front_.load(std::memory_order_relaxed);
	    type ret = static_cast<type &&>(queue_[front++ & capacityMask_]);
	    front_.store(front, std::memory_order_release);
	    return ret;
	}

	/// /param value
	/// /return
	constexpr inline std::size_t pop(type & value) noexcept {
		auto front = front_.load(std::memory_order_relaxed);
		auto size = (back_.load(std::memory_order_acquire) - front);
		value = static_cast<type &&>(queue_[front++ & capacityMask_]);
		front_.store(front, std::memory_order_release);
		return size;
	}

	/// \param value
	/// \return
	constexpr inline std::size_t try_pop(type & value) noexcept {
		// NOTE: acquire `back_`, so the element written by the producer is visible
		const std::size_t front = front_.load(std::memory_order_relaxed);
		const std::size_t size = back_.load(std::memory_order_acquire) - front;
		if (size > 0) {
			value = static_cast<type &&>(queue_[front & capacityMask_]);
			front_.store(front + 1, std::memory_order_release);
			return size;
		}
		return 0;
	}

	/// \tparam T_
	/// \param value
	/// \return
	template <typename T_>
	constexpr inline bool push(T_ && value) noexcept {
		// NOTE: release `back_`, so the consumer sees the written element
		if (std::size_t back = back_.load(std::memory_order_relaxed); (back - front_.load(std::memory_order_acquire)) < capacity_){
			queue_[back++ & capacityMask_] = static_cast<T_ &&>(value);
			back_.store(back, std::memory_order_release);
			return true;
		}
		return false;
	}

	/// \tparam Ts
	/// \param args
	/// \return
	template <typename ... Ts>
	constexpr inline bool emplace(Ts && ... args) noexcept {
		if (std::size_t back = back_.load(std::memory_order_relaxed); (back - front_.load(std::memory_order_acquire)) < capacity_) {
			queue_[back++ & capacityMask_] = T(static_cast<Ts &&>(args) ...);
			back_.store(back, std::memory_order_release);
			return true;
		}
		return false;
	}

	/// \return
	constexpr inline T const &front() const noexcept{
		return queue_[front_.load(std::memory_order_relaxed) & capacityMask_];
	}

	/// \return
	constexpr inline T &front() noexcept {
		return queue_[front_.load(std::memory_order_relaxed) & capacityMask_];
	}

	/// \return
	[[nodiscard]] constexpr inline bool empty() const noexcept {
		return back_.load(std::memory_order_acquire) == front_.load(std::memory_order_acquire);
	}

	/// \return
	[[nodiscard]] constexpr inline std::size_t capacity() const noexcept {
		return capacity_;
	}

	/// \return
	[[nodiscard]] std::size_t size() const noexcept {
		return back_.load(std::memory_order_acquire) - front_.load(std::memory_order_acquire);
	}

	/// \return
	constexpr inline std::size_t discard() noexcept {
		const std::size_t front = front_.load(std::memory_order_relaxed);
		queue_[front & capacityMask_] = {};
		front_.store(front + 1, std::memory_order_release);
		return (back_.load(std::memory_order_acquire) - (front + 1));
	}


private:
	// NOTE: atomics instead of `volatile`: `volatile` does not order the
	// 	element accesses on weakly ordered CPUs (e.g. ARM)
	std::atomic<std::size_t> front_;
	std::atomic<std::size_t> back_;

	std::size_t capacity_;
	std::size_t capacityMask_;

	std::vector<type, Allocator<type>> queue_;
};
