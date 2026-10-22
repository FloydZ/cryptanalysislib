#ifndef CRYPTANALYSISLIB_ATOMIC_SEMAPHORE_H
#define CRYPTANALYSISLIB_ATOMIC_SEMAPHORE_H

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#else
#include <semaphore.h>
#endif
#include <cassert>
#include <cerrno>

#include "helper.h"

namespace cryptanalysislib {
/// just a wrapper around the POSIX semaphore (`sem_t`, unnamed, not shared
/// between processes), initialized with 0
/// NOTE: macOS does not implement unnamed POSIX semaphores (`sem_init` fails
/// 	with ENOSYS), so a libdispatch semaphore is used there instead.
struct semaphore_wrapper {
private:
#ifdef __APPLE__
	dispatch_semaphore_t sem;

	inline void create() noexcept {
		sem = dispatch_semaphore_create(0);
		assert(sem != nullptr);
	}

	inline void close() noexcept {
		dispatch_release(sem);
	}
#else
	sem_t sem;

	inline void create() noexcept {
		int err = sem_init(&sem, 0, 0);
		assert(err == 0);
		(void)err;
	}

	inline void close() noexcept {
		sem_destroy(&sem);
	}
#endif

public:
	/// creates the semaphore with the value 0
	semaphore_wrapper() noexcept {
		create();
	}

	semaphore_wrapper(const semaphore_wrapper &) = delete;
	semaphore_wrapper &operator=(const semaphore_wrapper &) = delete;

	/// destroys the semaphore
	~semaphore_wrapper() noexcept {
		close();
	}

	/// blocks until the value is > 0, then decrements it
	inline void wait() noexcept {
#ifdef __APPLE__
		dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
#else
		int err;
		// NOTE: `sem_wait` returns early with EINTR if a signal arrives
		do { err = sem_wait(&sem); } while ((err != 0) && (errno == EINTR));
		assert(err == 0);
		(void)err;
#endif
	}

	/// increments the value `cnt` times, i.e. wakes up `cnt` waiters
	inline void signal(int cnt) noexcept {
		while (cnt-- > 0) {
#ifdef __APPLE__
			dispatch_semaphore_signal(sem);
#else
			sem_post(&sem);
#endif
		}
	}
};
} // end namespace cryptanalysislib

#include <atomic>
#include <coroutine>
#include <cstddef>
#include <mutex>

namespace cryptanalysislib::atomic {
/// counting semaphore for coroutines, taken from libcoro (`coro::semaphore`).
/// `co_await s.acquire()` suspends the coroutine until a resource is free.
/// `release()` hands its resource directly to a waiting coroutine, which is
/// resumed inside `release()` (i.e. in the releasing thread), or increments
/// the counter if nobody waits.
/// NOTE: the waiters are resumed in LIFO order, the semaphore is not fair.
/// \tparam Atomic atomic type of the counter and the stop flag
template<template<class> class Atomic = std::atomic>
class semaphore {
public:
	enum class acquire_result {
		acquired,         ///< the resource was acquired
		semaphore_stopped ///< `notify_waiters` was called, nothing was acquired
	};

	[[nodiscard]] constexpr static const char *to_string(const acquire_result ar) noexcept {
		switch (ar) {
			case acquire_result::acquired: return "acquired";
			case acquire_result::semaphore_stopped: return "semaphore_stopped";
		}
		return "unknown";
	}

	/// awaitable returned by `acquire()`
	class acquire_operation {
	public:
		explicit acquire_operation(semaphore &s) noexcept : m_semaphore(s) {}

		[[nodiscard]] bool await_ready() const noexcept {
			if (m_semaphore.m_notify_all_set.load(std::memory_order_relaxed)) {
				return true;
			}
			return m_semaphore.try_acquire();
		}

		bool await_suspend(std::coroutine_handle<> awaiting_coroutine) noexcept {
			std::unique_lock lk{m_semaphore.m_waiter_mutex};
			if (m_semaphore.m_notify_all_set.load(std::memory_order_relaxed)) {
				return false;
			}

			if (m_semaphore.try_acquire()) {
				return false;
			}

			// LIFO: this becomes the new head of the waiter list
			m_next = m_semaphore.m_acquire_waiters;
			m_semaphore.m_acquire_waiters = this;
			m_awaiting_coroutine = awaiting_coroutine;
			return true;
		}

		[[nodiscard]] acquire_result await_resume() const noexcept {
			if (m_semaphore.m_notify_all_set.load(std::memory_order_relaxed)) {
				return acquire_result::semaphore_stopped;
			}
			return acquire_result::acquired;
		}

	private:
		friend semaphore;

		semaphore &m_semaphore;
		std::coroutine_handle<> m_awaiting_coroutine;
		acquire_operation *m_next{nullptr};
	};

	/// \param least_max_value_and_starting_value[in]: maximal and initial number of resources
	explicit semaphore(const std::ptrdiff_t least_max_value_and_starting_value) noexcept
	    : semaphore(least_max_value_and_starting_value, least_max_value_and_starting_value) {}

	/// \param least_max_value[in]: maximal number of resources
	/// \param starting_value[in]: initial number of resources (at most `least_max_value`)
	semaphore(const std::ptrdiff_t least_max_value,
	          const std::ptrdiff_t starting_value) noexcept
	    : m_least_max_value(least_max_value),
	      m_counter(starting_value <= least_max_value ? starting_value : least_max_value) {}

	/// resumes all waiters with `semaphore_stopped`
	~semaphore() noexcept {
		notify_waiters();
	}

	semaphore(const semaphore &) = delete;
	semaphore(semaphore &&) = delete;
	semaphore &operator=(const semaphore &) = delete;
	semaphore &operator=(semaphore &&) = delete;

	/// \return awaitable: `co_await s.acquire()` returns an `acquire_result`
	[[nodiscard]] acquire_operation acquire() noexcept {
		return acquire_operation{*this};
	}

	/// releases a resource: either directly to a waiter, which is resumed
	/// in this call, or back into the counter
	void release() noexcept {
		// NOTE: the waiters are checked first, incrementing the counter and
		// 	resuming a waiter afterwards would race with new acquirers.
		std::unique_lock lk{m_waiter_mutex};
		if (m_acquire_waiters != nullptr) {
			acquire_operation *to_resume = m_acquire_waiters;
			m_acquire_waiters = m_acquire_waiters->m_next;
			lk.unlock();

			// transfers the ownership of the resource to the resumed waiter
			to_resume->m_awaiting_coroutine.resume();
		} else {
			// relaxed is enough within the lock
			m_counter.fetch_add(1, std::memory_order_relaxed);
		}
	}

	/// \return true if a resource was acquired, never suspends
	[[nodiscard]] bool try_acquire() noexcept {
		// optimistically grab the resource
		const auto previous = m_counter.fetch_sub(1, std::memory_order_acq_rel);
		if (previous <= 0) {
			// not available: undo the acquisition
			m_counter.fetch_add(1, std::memory_order_release);
			return false;
		}
		return true;
	}

	/// \return the maximal number of resources
	[[nodiscard]] constexpr std::ptrdiff_t max() const noexcept {
		return m_least_max_value;
	}

	/// \return the number of free resources
	[[nodiscard]] std::ptrdiff_t value() const noexcept {
		return m_counter.load(std::memory_order_relaxed);
	}

	/// stops the semaphore: all current and future `acquire` return
	/// `semaphore_stopped`, the current waiters are resumed in this call
	void notify_waiters() noexcept {
		m_notify_all_set.exchange(true, std::memory_order_release);
		while (true) {
			std::unique_lock lk{m_waiter_mutex};
			if (m_acquire_waiters != nullptr) {
				acquire_operation *to_resume = m_acquire_waiters;
				m_acquire_waiters = m_acquire_waiters->m_next;
				lk.unlock();

				to_resume->m_awaiting_coroutine.resume();
			} else {
				break;
			}
		}
	}

private:
	friend class acquire_operation;

	const std::ptrdiff_t m_least_max_value;
	Atomic<std::ptrdiff_t> m_counter;

	std::mutex m_waiter_mutex{};
	acquire_operation *m_acquire_waiters{nullptr};

	Atomic<bool> m_notify_all_set{false};
};
} // end namespace cryptanalysislib::atomic

#endif
