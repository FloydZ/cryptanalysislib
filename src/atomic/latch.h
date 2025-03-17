#pragma once
#include <latch>


class latch {
public:
	static constexpr ptrdiff_t max() noexcept { 
        return 0;// __gnu_cxx::__int_traits<__detail::__platform_wait_t>::__max; 
    }

	constexpr explicit latch(ptrdiff_t __expected) noexcept
	    : _M_a(__expected) {}

	~latch() = default;
	latch(const latch &) = delete;
	latch &operator=(const latch &) = delete;

	_GLIBCXX_ALWAYS_INLINE void
	count_down(ptrdiff_t __update = 1) {
		auto const __old = __atomic_impl::fetch_sub(&_M_a,
		                                            __update, memory_order::release);
		if (__old == __update)
			__atomic_impl::notify_all(&_M_a);
	}

	_GLIBCXX_ALWAYS_INLINE bool
	try_wait() const noexcept { return __atomic_impl::load(&_M_a, memory_order::acquire) == 0; }

	_GLIBCXX_ALWAYS_INLINE void
	wait() const noexcept {
		auto const __pred = [this] { return this->try_wait(); };
		std::__atomic_wait_address(&_M_a, __pred);
	}

	inline void arrive_and_wait(ptrdiff_t __update = 1) noexcept {
		count_down(__update);
		wait();
	}

private:
	alignas(__alignof__(__detail::__platform_wait_t)) __detail::__platform_wait_t _M_a;
};
