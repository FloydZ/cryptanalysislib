#ifndef CRYPTANALYSISLIB_FUTEX_H
#define CRYPTANALYSISLIB_FUTEX_H

#ifndef __APPLE__
#include <errno.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <cstdint>
#include <type_traits>

#include "atomic_primitives.h"


# define compare_and_swap(mem, newval, oldval)			 \
__arch_compare_and_exchange_val_64_acq(mem, newval, oldval)


namespace cryptanalysislib::atomic {
	// very simple binary semaphore based on linux futex
	struct futex {
	private:
		int count = 0;
		constexpr static int FUTEX_PASSED = (-(1024 * 1024 * 1024));

		constexpr futex() noexcept : count(1) {}
	public:

		/// \param val
		constexpr futex(const int val) noexcept :
			count(val) {}

	private:

		/// Atomic dec: return new value.
		/// \param counter
		/// \return
		inline int __futex_down(int *counter) noexcept {
			const int val = ACQUIRE(counter);

			// Don't decrement if already negative.
			if (val < 0) [[unlikely]] {
				return val;
			}

			const int nval = val - 1;
			// NOTE: value-returning CAS(ptr, expected, desired); the previous
			// call had expected/desired swapped, so the counter never changed
			const int oval = CASnp(counter, val, nval);
			if (oval == val) {
				return nval;
			}

			// Otherwise, we have no way of knowing value.  Guess -1 (if
			// we're wrong we'll spin).
			return -1;
		}

		/* Atomic inc: return 1 if counter incremented from 0 to 1. */
		static __inline__ int __futex_up(int *counter) noexcept {
			const int val = ACQUIRE(counter);
			const int nval = val + 1;
			// NOTE: see `__futex_down`
			const int oval = CASnp(counter, val, nval);
			return (oval == val && oval == 0);
		}

		///
		/// \param uaddr
		/// \param op
		/// \param val
		/// \param timeout
		/// \return
		static inline int sys_futex(int *uaddr,
									int op,
									int val,
									const struct timespec *timeout) noexcept {
			return syscall(SYS_futex, uaddr, op, val, timeout, nullptr, 0);
		}

		/// Returns -1 on fail, 0 on wakeup, 1 on pass, 2 on didn't sleep
		/// \param val
		/// \param rel
		/// \return
		int __futex_down_slow(int val,
							  struct timespec *rel) noexcept {
			if (sys_futex(&count, FUTEX_WAIT, val, rel) == 0) {
				// <= in case someone else decremented it
				if (ACQUIRE(&count) <= FUTEX_PASSED) {
					STORE(&count, -1);
					return 1;
				}
				return 0;
			}
			// EWOULDBLOCK just means value changed before we slept: loop
			if (errno == EWOULDBLOCK) {
				return 2;
			}
			return -1;
		}

		///
		/// \return
		inline int __futex_up_slow() noexcept {
			// NOTE: release, so the writes of the critical section are visible
			RELEASE(&count, 1);
			return sys_futex(&count, FUTEX_WAKE, 1, NULL);
		}

	public:
		/// \param signal
		/// \return
		int futex_await(int signal) {
			return sys_futex(&count, FUTEX_FD, signal, NULL);
		}

		/// \param rel
		/// \return
		inline int futex_down_timeout(struct timespec *rel) noexcept {
			int val, woken = 0;

			/* Returns new value */
			while ((val = __futex_down(&count)) != 0) {
				const auto v = __futex_down_slow(val, rel);
				switch (v) {
					case -1:
						// error
						return -1;
					case 1:
						// passed
						return 0;
					case 2:
						// NOTE: EWOULDBLOCK: the value changed before we
						// slept, try again (this is *not* an acquisition)
						break;
					case 0:
						woken = 1;
						// slept
						break;
				}
			}

			// If we were woken, someone else might be sleeping too: set to -1
			if (woken) {
				STORE(&count, -1);
			}

			return 0;
		}

		/// If __futex_down decrements from 1 to 0, we have it.
		/// Otherwise, sleep.
		/// \return
		inline int down() noexcept {
			return futex_down_timeout(nullptr);
		}

		/// returns 0 on success else -1
		inline int trydown() noexcept {
			return (__futex_down(&count) == 0 ? 0: -1);
		}

		// If __futex_up increments count from 0 -> 1, none was waiting.
		// Otherwise, set to 1 and tell kernel to wake them up.
		// returns 0 on success
		inline int up() noexcept {
			if (!__futex_up(&count)) {
				return __futex_up_slow();
			}

			return 0;
		}

		///
		inline int up_fair() noexcept {
			// Someone waiting?
			if (!__futex_up(&count)) {
				RELEASE(&count, FUTEX_PASSED);
				// If we wake one, they'll see it's a direct pass.
				if (sys_futex(&count, FUTEX_WAKE, 1, nullptr) == 1) {
					return 0;
				}

				// Otherwise do normal slow case
				return __futex_up_slow();
			}
			return 0;
		}

        constexpr inline void set(const uint32_t c) noexcept {
            STORE(&count, c);
        }
        constexpr inline int get() const noexcept {
            return ACQUIRE(&count);
        }
	};

	/// futex2 syscalls (https://docs.kernel.org/userspace-api/futex2.html):
	/// wait/wake with a bit mask (Linux 6.7+), and waiting on up to 128
	/// futexes at once (`waitv`, Linux 5.16+). All functions return -1 and
	/// set `errno` on failure, `ENOSYS` if the kernel does not support futex2.
	/// NOTE: the API has futex words of 8, 16, 32 and 64 bits, but the kernel
	/// 	only implements 32 bits so far (EINVAL for the other sizes).
	/// NOTE: only defined with Linux 6.7+ headers (`FUTEX2_SIZE_U8`)
#ifdef FUTEX2_SIZE_U8
	namespace futex2 {
		/// \return the FUTEX2 size flag of `T`
		template<typename T>
		constexpr static uint32_t size_flag() noexcept {
			static_assert(std::is_integral_v<T> && (sizeof(T) <= 8));
			if constexpr (sizeof(T) == 1) { return FUTEX2_SIZE_U8; }
			else if constexpr (sizeof(T) == 2) { return FUTEX2_SIZE_U16; }
			else if constexpr (sizeof(T) == 4) { return FUTEX2_SIZE_U32; }
			else { return FUTEX2_SIZE_U64; }
		}

		/// \return `mask` restricted to the bits of `T` (the kernel rejects
		/// 	masks and values wider than the futex word)
		template<typename T>
		constexpr static uint64_t word_mask(const uint64_t mask) noexcept {
			if constexpr (sizeof(T) == 8) {
				return mask;
			} else {
				return mask & ((1ull << (8u * sizeof(T))) - 1ull);
			}
		}

		/// \return the flags of a futex over `T`
		/// \param shared[in]: false: only shared between the threads of
		/// 	this process (faster)
		template<typename T>
		constexpr static uint32_t flags(const bool shared = false) noexcept {
			return size_flag<T>() | (shared ? 0u : uint32_t(FUTEX2_PRIVATE));
		}

		/// sleeps while `*addr == val`
		/// \param addr[in]: futex word, aligned to `sizeof(T)`
		/// \param val[in]: expected value
		/// \param mask[in]: only woken by a `wake` with a mask sharing a bit
		/// \param timeout[in]: absolute timeout w.r.t. `clock`, `nullptr`: none
		/// \param clock[in]: CLOCK_MONOTONIC or CLOCK_REALTIME
		/// \param shared[in]: see `flags`
		/// \return 0 if woken, -1 with errno EAGAIN (`*addr != val`),
		/// 	ETIMEDOUT, EINTR, ...
		template<typename T>
		inline int wait(T *addr,
		                const T val,
		                const uint64_t mask = ~0ull,
		                const struct timespec *timeout = nullptr,
		                const clockid_t clock = CLOCK_MONOTONIC,
		                const bool shared = false) noexcept {
#ifdef SYS_futex_wait
			return int(syscall(SYS_futex_wait, addr, word_mask<T>(uint64_t(val)), word_mask<T>(mask),
			                   flags<T>(shared), timeout, clock));
#else
			(void)addr; (void)val; (void)mask; (void)timeout; (void)clock; (void)shared;
			errno = ENOSYS;
			return -1;
#endif
		}

		/// wakes up to `nr` threads waiting on `addr` with a mask sharing a
		/// bit with `mask`
		/// \return number of woken threads, -1 on error
		template<typename T>
		inline int wake(T *addr,
		                const int nr = INT32_MAX,
		                const uint64_t mask = ~0ull,
		                const bool shared = false) noexcept {
#ifdef SYS_futex_wake
			return int(syscall(SYS_futex_wake, addr, word_mask<T>(mask), nr, flags<T>(shared)));
#else
			(void)addr; (void)nr; (void)mask; (void)shared;
			errno = ENOSYS;
			return -1;
#endif
		}

		/// \return a waiter for `waitv`, which sleeps while `*addr == val`
		template<typename T>
		constexpr inline struct futex_waitv waiter(T *addr,
		                                           const T val,
		                                           const bool shared = false) noexcept {
			struct futex_waitv w{};
			w.val = word_mask<T>(uint64_t(val));
			w.uaddr = uint64_t(uintptr_t(addr));
			w.flags = flags<T>(shared);
			return w;
		}

		/// sleeps until one of the futexes is woken, if all `nr` futexes
		/// have their expected value
		/// \param waiters[in]: array of `nr <= FUTEX_WAITV_MAX` waiters
		/// \param timeout[in]: absolute timeout w.r.t. `clock`, `nullptr`: none
		/// \return the index of the woken futex, -1 with errno EAGAIN (a
		/// 	futex has not its expected value), ETIMEDOUT, ...
		inline int waitv(struct futex_waitv *waiters,
		                 const uint32_t nr,
		                 const struct timespec *timeout = nullptr,
		                 const clockid_t clock = CLOCK_MONOTONIC) noexcept {
#ifdef SYS_futex_waitv
			return int(syscall(SYS_futex_waitv, waiters, nr, 0, timeout, clock));
#else
			(void)waiters; (void)nr; (void)timeout; (void)clock;
			errno = ENOSYS;
			return -1;
#endif
		}

		/// wakes up to `nr_wake` waiters of `src`, and moves up to
		/// `nr_requeue` other waiters to `dst`, if `*src == src_val`
		/// \return number of woken and requeued waiters, -1 on error
		template<typename T>
		inline int requeue(T *src,
		                   const T src_val,
		                   T *dst,
		                   const int nr_wake,
		                   const int nr_requeue,
		                   const bool shared = false) noexcept {
#ifdef SYS_futex_requeue
			struct futex_waitv w[2] = {waiter(src, src_val, shared), waiter(dst, T(0), shared)};
			return int(syscall(SYS_futex_requeue, w, 0, nr_wake, nr_requeue));
#else
			(void)src; (void)src_val; (void)dst; (void)nr_wake; (void)nr_requeue; (void)shared;
			errno = ENOSYS;
			return -1;
#endif
		}

		/// \return true if the kernel supports futex2 wait/wake/waitv
		inline bool available() noexcept {
			uint32_t x = 0;
			if (wake(&x, 1) < 0) {
				return false;
			}

			struct futex_waitv w = waiter(&x, 1u);
			// the value differs: EAGAIN, if supported
			return (waitv(&w, 1) < 0) && (errno == EAGAIN);
		}
	} // end namespace futex2
#endif
} // end namespace cryptanalysislib
#endif//CRYPTANALYSISLIB_FUTEX_H

#endif