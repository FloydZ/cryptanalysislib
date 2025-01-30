#ifndef CRYPTANALYSISLIB_FUTEX_H
#define CRYPTANALYSISLIB_FUTEX_H

#ifndef __APPLE__
#include <errno.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <sys/time.h>

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
			const int val = *counter;

			// Don't decrement if already negative.
			if (val < 0) [[unlikely]] {
				return val;
			}

			int nval = val -1;
			const int oval = CAS(counter, &nval, val);
			if (oval == val) {
				return val-1;
			}

			// Otherwise, we have no way of knowing value.  Guess -1 (if
			// we're wrong we'll spin).
			return -1;
		}

		/* Atomic inc: return 1 if counter incremented from 0 to 1. */
		static __inline__ int __futex_up(int *counter) noexcept {
			const int val = *counter;
			int nval = val+1;
			const int oval = CAS(counter, &nval, val);
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
				if (count <= FUTEX_PASSED) {
					count = -1;
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
			count = 1;
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
					case 2:
						// passed
						return 0;
					case 0:
						woken = 1;
						// slept
						break;
				}
			}

			// If we were woken, someone else might be sleeping too: set to -1
			if (woken) {
				count = -1;
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
				count = FUTEX_PASSED;
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
            count = c;
        }
        constexpr inline int get() const noexcept {
            return count; 
        }
	};
} // end namespace cryptanalysislib
#endif//CRYPTANALYSISLIB_FUTEX_H

#endif