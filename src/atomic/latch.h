#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>


class latch {
public:
	static constexpr ptrdiff_t max() noexcept { return PTRDIFF_MAX; }

	constexpr explicit latch(ptrdiff_t __expected) noexcept
	    : _M_a(__expected) {}

	~latch() = default;
	latch(const latch &) = delete;
	latch &operator=(const latch &) = delete;

	inline void count_down(ptrdiff_t __update = 1) noexcept {
		auto const __old = _M_a.fetch_sub(__update, std::memory_order_release);
		if (__old == __update)
			_M_a.notify_all();
	}

	inline bool try_wait() const noexcept { return _M_a.load(std::memory_order_acquire) == 0; }

	inline void wait() const noexcept {
		ptrdiff_t __cur;
		while ((__cur = _M_a.load(std::memory_order_acquire)) != 0)
			_M_a.wait(__cur, std::memory_order_acquire);
	}

	inline void arrive_and_wait(ptrdiff_t __update = 1) noexcept {
		count_down(__update);
		wait();
	}

private:
	std::atomic<ptrdiff_t> _M_a;
};
