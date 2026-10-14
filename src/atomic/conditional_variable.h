#ifndef CRYPTANALYSISLIB_ATOMIC_CONDITIONAL_VARIABLE_H
#define CRYPTANALYSISLIB_ATOMIC_CONDITIONAL_VARIABLE_H

#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <thread>

#ifdef __linux__
#include <linux/futex.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#endif

namespace cryptanalysislib {
	/// Condition variable for any lock with `lock()`/`unlock()` (like
	/// `std::condition_variable_any`).
	/// A waiter reads the sequence number `seq` before it releases the lock,
	/// and sleeps only while `seq` is unchanged; every notification increments
	/// `seq`. Hence no notification after the release of the lock is lost.
	/// On Linux the waiters sleep on `seq` via `futex` (also with timeouts),
	/// otherwise via `std::atomic::wait` (timeouts by polling).
	/// Spurious wakeups are possible, as for `std::condition_variable`.
	/// Usage:
	/// \code
	/// 	std::unique_lock l(m);
	/// 	cv.wait(l, [&]{ return ready; });
	/// 	// other thread:
	/// 	{ std::lock_guard g(m); ready = true; }
	/// 	cv.notify_one();
	/// \endcode
	class condition_variable {
	private:
		/// incremented by every notification
		std::atomic<uint32_t> seq{0};
		/// number of sleeping (or about to sleep) waiters
		std::atomic<uint32_t> waiters{0};

#ifdef __linux__
		/// \param rel[in]: relative timeout, nullptr: none
		void sleep(const uint32_t s,
		           const struct timespec *rel) noexcept {
			syscall(SYS_futex, (uint32_t *)&seq, FUTEX_WAIT_PRIVATE, s, rel, nullptr, 0);
		}

		void wake(const int n) noexcept {
			syscall(SYS_futex, (uint32_t *)&seq, FUTEX_WAKE_PRIVATE, n, nullptr, nullptr, 0);
		}
#endif

		inline void notify(const int n) noexcept {
			// seq_cst: either the waiter sees the new `seq`, or this sees the
			// waiter in `waiters`
			seq.fetch_add(1, std::memory_order_seq_cst);
			if (waiters.load(std::memory_order_seq_cst) == 0) {
				return;
			}
#ifdef __linux__
			wake(n);
#else
			if (n == 1) { seq.notify_one(); } else { seq.notify_all(); }
#endif
		}

		/// releases `lock`, sleeps until notified or `deadline`, locks `lock`
		/// \return false if the deadline passed
		template<class Lock, class Clock, class Duration>
		bool wait_impl(Lock &lock,
		               const std::chrono::time_point<Clock, Duration> *deadline) noexcept {
			waiters.fetch_add(1, std::memory_order_seq_cst);
			const uint32_t s = seq.load(std::memory_order_seq_cst);
			lock.unlock();

			bool in_time = true;
			if (deadline == nullptr) {
#ifdef __linux__
				sleep(s, nullptr);
#else
				seq.wait(s, std::memory_order_acquire);
#endif
			} else {
				const auto now = Clock::now();
				if (now >= *deadline) {
					in_time = false;
				} else {
#ifdef __linux__
					const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(*deadline - now).count();
					struct timespec rel;
					rel.tv_sec = time_t(ns / 1000000000ll);
					rel.tv_nsec = long(ns % 1000000000ll);
					sleep(s, &rel);
#else
					while ((seq.load(std::memory_order_acquire) == s) && (Clock::now() < *deadline)) {
						std::this_thread::yield();
					}
#endif
					in_time = Clock::now() < *deadline;
				}
			}

			waiters.fetch_sub(1, std::memory_order_relaxed);
			lock.lock();
			return in_time;
		}

	public:
		constexpr condition_variable() noexcept = default;
		condition_variable(const condition_variable &) = delete;
		condition_variable &operator=(const condition_variable &) = delete;

		/// wakes (at least) one waiting thread
		inline void notify_one() noexcept {
			notify(1);
		}

		/// wakes all waiting threads
		inline void notify_all() noexcept {
			notify(INT_MAX);
		}

		/// releases `lock`, sleeps until notified (or spuriously woken),
		/// locks `lock` again
		/// \param lock[in]: locked by the calling thread
		template<class Lock>
		inline void wait(Lock &lock) noexcept {
			wait_impl<Lock, std::chrono::steady_clock, std::chrono::steady_clock::duration>(lock, nullptr);
		}

		/// waits until `pred()` is true
		/// \param lock[in]: locked by the calling thread, protects `pred`
		template<class Lock, class Predicate>
		inline void wait(Lock &lock, Predicate pred) noexcept {
			while (!pred()) {
				wait(lock);
			}
		}

		/// waits until `pred()` is true or the deadline passed
		/// \return `pred()`
		template<class Lock, class Clock, class Duration, class Predicate>
		inline bool wait_until(Lock &lock,
		                       const std::chrono::time_point<Clock, Duration> &deadline,
		                       Predicate pred) noexcept {
			while (!pred()) {
				if (!wait_impl(lock, &deadline)) {
					return pred();
				}
			}
			return true;
		}

		/// waits until `pred()` is true or `duration` passed
		/// \return `pred()`
		template<class Lock, class Rep, class Period, class Predicate>
		inline bool wait_for(Lock &lock,
		                     const std::chrono::duration<Rep, Period> &duration,
		                     Predicate pred) noexcept {
			return wait_until(lock, std::chrono::steady_clock::now() + duration, pred);
		}
	};
} // end namespace cryptanalysislib
#endif
