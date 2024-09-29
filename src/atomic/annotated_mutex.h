#ifndef CRYPTANALYSISLIB_ANNOTATED_MUTEX_H
#define CRYPTANALYSISLIB_ANNOTATED_MUTEX_H

#include <condition_variable>
#include <mutex>

#include "atomic_primitives.h"

// mutex is a wrapper around std::mutex that offers Thread Safety Analysis
// annotations.
// mutex also holds methods for performing std::condition_variable::wait() calls
// as these require a std::unique_lock<> which are unsupported by the TSA.
class CAPABILITY("annotated_mutex") annotated_mutex {
public:
	///
	inline void lock() noexcept ACQUIRE_() { _.lock(); }

	///
	inline void unlock() noexcept RELEASE_() { _.unlock(); }

	///
	inline bool try_lock() noexcept TRY_ACQUIRE(true) {
		return _.try_lock();
	}

	/// wait_locked calls cv.wait() on this already locked mutex.
	/// \tparam Predicate
	/// \param cv
	/// \param p
	template<typename Predicate>
	inline void wait_locked(std::condition_variable &cv,
	                        Predicate &&p) noexcept REQUIRES(this) {
		std::unique_lock<std::mutex> lock(_, std::adopt_lock);
		cv.wait(lock, std::forward<Predicate>(p));
		lock.release();// Keep lock held.
	}

	/// wait_until_locked calls cv.wait() on this already locked mutex.
	/// \tparam Predicate
	/// \tparam Time
	/// \param cv
	/// \param time
	/// \param p
	/// \return
	template<typename Predicate,
	         typename Time>
	inline bool wait_until_locked(std::condition_variable &cv,
	                              Time &&time,
	                              Predicate &&p) noexcept REQUIRES(this) {
		std::unique_lock<std::mutex> lock(_, std::adopt_lock);
		auto res = cv.wait_until(lock, std::forward<Time>(time),
		                         std::forward<Predicate>(p));
		lock.release();// Keep lock held.
		return res;
	}

private:
	friend class lock;
	std::mutex _;
};

// lock is a RAII lock helper that offers Thread Safety Analysis annotations.
// lock also holds methods for performing std::condition_variable::wait()
// calls as these require a std::unique_lock<> which are unsupported by the TSA.
template<class Mutex=annotated_mutex>
class SCOPED_CAPABILITY _lock {
public:
	///
	/// \param m
	inline _lock(Mutex &m) noexcept ACQUIRE_(m)  : _(m._)  {}

	///
	inline ~_lock() noexcept RELEASE_() = default;

	/// wait calls cv.wait() on this lock.
	/// \tparam Predicate
	/// \param cv
	/// \param p
	template<typename Predicate>
	inline void wait(std::condition_variable &cv,
	                 Predicate &&p) noexcept {
		cv.wait(_, std::forward<Predicate>(p));
	}

	/// wait_until calls cv.wait() on this lock.
	/// \tparam Predicate
	/// \tparam Time
	/// \param cv
	/// \param time
	/// \param p
	/// \return
	template<typename Predicate,
	         typename Time>
	inline bool wait_until(std::condition_variable &cv,
	                       Time &&time,
	                       Predicate &&p) noexcept {
		return cv.wait_until(_, std::forward<Time>(time),
		                     std::forward<Predicate>(p));
	}

	/// \return
	inline bool owns_lock() const noexcept { return _.owns_lock(); }

	/// lock_no_tsa locks the mutex outside the visibility of the thread
	/// safety analysis. Use with caution.
	inline void lock_no_tsa() noexcept { _.lock(); }

	/// unlock_no_tsa unlocks the mutex outside the visibility of the thread
	/// safety analysis. Use with caution.
	inline void unlock_no_tsa() noexcept { _.unlock(); }

private:
	std::unique_lock<Mutex> _;
};
#endif
