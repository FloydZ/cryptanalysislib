#pragma once

#include <concepts>
#include <cstddef>
#include <vector>

#include "traits.h"

/// TODO allocator + iterator
/// @tparam T
template<typename T>
class spsc_fixed_queue : non_copyable {
public:
	using type = T;
	using value_type = T;

	[[nodiscard]] constexpr inline auto begin() noexcept { return queue_.begin() + front_; }
	[[nodiscard]] constexpr inline auto end() noexcept { return queue_.begin() + back_; }
	[[nodiscard]] constexpr inline auto begin() const noexcept { return queue_.begin() + front_; }
	[[nodiscard]] constexpr inline auto end() const noexcept { return queue_.begin() + back_; }

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
	    std::size_t front = front_;
	    type ret = std::move(queue_[front++ & capacityMask_]);
	    front_ = front;
	    return ret;
	}

	/// /param value
	/// /return
	constexpr inline std::size_t pop(type & value) noexcept {
		auto front = front_;
		auto size = (back_ - front);
		value = std::move(queue_[front++ & capacityMask_]);
		front_ = front;
		return size;
	}

	/// \param value
	/// \return
	constexpr inline std::size_t try_pop(type & value) noexcept {
		if (auto front = front_, size = (back_ - front); size > 0){
			value = std::move(queue_[front++ & capacityMask_]);
			front_ = front;
			return size;
		}
		return 0;
	}

	/// \tparam T_
	/// \param value
	/// \return
	template <typename T_>
	constexpr inline bool push(T_ && value) noexcept {
		if (std::size_t back = back_; (back - front_) < capacity_){
			queue_[back++ & capacityMask_] = std::forward<T_>(value);
			back_ = back;
			return true;
		}
		return false;
	}

	/// \tparam Ts
	/// \param args
	/// \return
	template <typename ... Ts>
	constexpr inline bool emplace(Ts && ... args) noexcept {
		if (std::size_t back = back_; (back - front_) < capacity_) {
			queue_[back++ & capacityMask_] = T(std::forward<Ts>(args) ...);
			back_ = back;
			return true;
		}
		return false;
	}

	/// \return
	constexpr inline T const &front() const noexcept{
		return queue_[front_ & capacityMask_];
	}

	/// \return
	constexpr inline T &front() noexcept {
		return queue_[front_ & capacityMask_];
	}

	/// \return
	[[nodiscard]] constexpr inline bool empty() const noexcept {
		return back_ == front_;
	}

	/// \return
	[[nodiscard]] constexpr inline std::size_t capacity() const noexcept {
		return capacity_;
	}

	/// \return
	[[nodiscard]] std::size_t size() const noexcept {
		return back_ - front_;
	}

	/// \return
	constexpr inline std::size_t discard() noexcept {
		queue_[front_ & capacityMask_] = {};
		front_ = front_ + 1;
		return (back_ - front_);
	}


private:
	std::size_t volatile front_;
	std::size_t volatile back_;

	std::size_t capacity_;
	std::size_t capacityMask_;

	std::vector<type> queue_;
};
