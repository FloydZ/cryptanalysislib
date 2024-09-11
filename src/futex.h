#ifndef CRYPTANALYSISLIB_FUTEX_H
#define CRYPTANALYSISLIB_FUTEX_H

#include <errno.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#include <sys/time.h>

#include "atomic_primitives.h"

struct futex {
	int count;
};

#define FUTEX_PASSED (-(1024 * 1024 * 1024))
// TODO merge into `atomic_primitve`
# define __arch_compare_and_exchange_val_64_acq(mem, newval, oldval)	\
  ({ __typeof (*mem) ret; \
    __asm __volatile ("lock\n" "cmpxchgq %q2, %1\n"		 \
		       : "=a" (ret), "=m" (*mem)		 \
		       : "r" ((long int) (newval)), "m" (*mem),	 \
			 "0" ((long int) (oldval)));		 \
     ret; })
# define compare_and_swap(mem, newval, oldval)			 \
  __arch_compare_and_exchange_val_64_acq(mem, newval, oldval)

// TODO tests and comments

/// Atomic dec: return new value.
/// \param counter
/// \return
static __inline__ int __futex_down(int *counter) noexcept {
	const int val = *counter;

	/* Don't decrement if already negative. */
	if (val < 0) [[unlikely]] {
		return val;
	}

	const int oval = compare_and_swap(counter, val-1, val);
	if (oval == val) {
		return val-1;
	}
	/* Otherwise, we have no way of knowing value.  Guess -1 (if
     we're wrong we'll spin). */
	return -1;
}

/* Atomic inc: return 1 if counter incremented from 0 to 1. */
static __inline__ int __futex_up(int *counter) noexcept {
	const int val = *counter;
	const int oval = compare_and_swap(counter, val+1, val);
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
                     const struct timespec *timeout) {
	return syscall(SYS_futex, uaddr, op, val, timeout, nullptr, 0);
}

/// Returns -1 on fail, 0 on wakeup, 1 on pass, 2 on didn't sleep
/// \param futx
/// \param val
/// \param rel
/// \return
static int __futex_down_slow(struct futex *futx,
                             int val,
                             struct timespec *rel) {
	if (sys_futex(&futx->count, FUTEX_WAIT, val, rel) == 0) {
		/* <= in case someone else decremented it */
		if (futx->count <= FUTEX_PASSED) {
			futx->count = -1;
			return 1;
		}
		return 0;
	}
	/* EWOULDBLOCK just means value changed before we slept: loop */
	if (errno == EWOULDBLOCK) {
		return 2;
	}
	return -1;
}

///
/// \param futx
/// \return
inline int __futex_up_slow(struct futex *futx) noexcept {
	futx->count = 1;
	return sys_futex(&futx->count, FUTEX_WAKE, 1, NULL);
}

///
/// \param futx
/// \param signal
/// \return
int futex_await(struct futex *futx, int signal) {
	return sys_futex(&futx->count, FUTEX_FD, signal, NULL);
}

///
/// \param futx
/// \param val
inline void futex_init(struct futex *futx,
                       const int val) noexcept {
	futx->count = val;
}

///
/// \param futx
/// \param rel
/// \return
static inline int futex_down_timeout(struct futex *futx,
                                     struct timespec *rel) noexcept {
	int val, woken = 0;

	/* Returns new value */
	while ((val = __futex_down(&futx->count)) != 0) {
		switch (__futex_down_slow(futx, val, rel)) {
			case -1:
				return -1; /* error */
			case 1:
				return 0; /* passed */
			case 0:
				woken = 1;
				break; /* slept */
		}
	}

	/* If we were woken, someone else might be sleeping too: set to -1 */
	if (woken) {
		futx->count = -1;
	}

	return 0;
}

/// If __futex_down decrements from 1 to 0, we have it.
/// Otherwise sleep.
/// \param futx
/// \return
static inline int futex_down(futex *futx) noexcept {
	return futex_down_timeout(futx, nullptr);
}

static inline int futex_trydown(futex *futx) noexcept {
	return (__futex_down(&futx->count) == 0 ? 0: -1);
}

/* If __futex_up increments count from 0 -> 1, none was waiting.
   Otherwise, set to 1 and tell kernel to wake them up. */
static inline int futex_up(struct futex *futx) noexcept {
	if (!__futex_up(&futx->count))
		return __futex_up_slow(futx);
	return 0;
}

static inline int futex_up_fair(struct futex *futx) noexcept {
	/* Someone waiting? */
	if (!__futex_up(&futx->count)) {
		futx->count = FUTEX_PASSED;
		/* If we wake one, they'll see it's a direct pass. */
		if (sys_futex(&futx->count, FUTEX_WAKE, 1, NULL) == 1)
			return 0;
		/* Otherwise do normal slow case */
		return __futex_up_slow(futx);
	}
	return 0;
}
#endif//CRYPTANALYSISLIB_FUTEX_H
