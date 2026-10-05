#ifndef CRYPTANALYSISLIB_ANNOTATED_MUTEX_H
#define CRYPTANALYSISLIB_ANNOTATED_MUTEX_H

#include <condition_variable>
#include <mutex>

#include "atomic_primitives.h"

/// annotated_mutex is a wrapper around std::mutex that offers Thread Safety Analysis
/// annotations.
/// mutex also holds methods for performing std::condition_variable::wait() calls
/// as these require a std::unique_lock<> which are unsupported by the TSA.
/// Annotated mutex wrapper for thread safety analysis
/// \tparam mutex[in]: underlying mutex type to wrap
template<class mutex=std::mutex>
class CAPABILITY("annotated_mutex") annotated_mutex {
public:
	/// Locks the mutex
	/// Thread Safety Analysis: Acquires the capability
	inline void lock() noexcept ACQUIRE_() { _.lock(); }

	/// Unlocks the mutex
	/// Thread Safety Analysis: Releases the capability
	inline void unlock() noexcept RELEASE_() { _.unlock(); }

	/// Attempts to lock the mutex without blocking
	/// Thread Safety Analysis: Tries to acquire the capability
	/// \return true if the lock was acquired, false otherwise
	inline bool try_lock() noexcept TRY_ACQUIRE(true) {
		return _.try_lock();
	}

	/// Wait on a condition variable with this already locked mutex
	/// \tparam Predicate[in]: predicate type for the condition
	/// \param cv[in]: condition variable to wait on
	/// \param p[in]: predicate that must become true before continuing
	template<typename Predicate>
	inline void wait_locked(std::condition_variable &cv,
	                        Predicate &&p) noexcept REQUIRES(this) {
		std::unique_lock<mutex> lock(_, std::adopt_lock);
		cv.wait(lock, std::forward<Predicate>(p));
		lock.release();// Keep lock held.
	}

	/// Wait on a condition variable with timeout with this already locked mutex
	/// \tparam Predicate[in]: predicate type for the condition
	/// \tparam Time[in]: time point type for the timeout
	/// \param cv[in]: condition variable to wait on
	/// \param time[in]: timeout time point
	/// \param p[in]: predicate that must become true before continuing
	/// \return true if predicate became true, false if timeout occurred
	template<typename Predicate,
	         typename Time>
	inline bool wait_until_locked(std::condition_variable &cv,
	                              Time &&time,
	                              Predicate &&p) noexcept REQUIRES(this) {
		std::unique_lock<mutex> lock(_, std::adopt_lock);
		auto res = cv.wait_until(lock,
								 std::forward<Time>(time),
		                         std::forward<Predicate>(p));
		lock.release();// Keep lock held.
		return res;
	}

private:
	friend class lock;
	mutex _;
};

// lock is a RAII lock helper that offers Thread Safety Analysis annotations.
// lock also holds methods for performing std::condition_variable::wait()
// calls as these require a std::unique_lock<> which are unsupported by the TSA.
template<class Mutex=annotated_mutex<std::mutex>>
class SCOPED_CAPABILITY _lock {
public:
	/// Constructor that acquires the mutex
	/// Thread Safety Analysis: Acquires the capability
	/// \param m[in]: mutex to lock
	inline _lock(Mutex &m) noexcept ACQUIRE_(m)  : _(m._)  {}

	/// Destructor that releases the mutex
	/// Thread Safety Analysis: Releases the capability
	inline ~_lock() noexcept RELEASE_() = default;

	/// Wait on a condition variable using this lock
	/// \tparam Predicate[in]: predicate type for the condition
	/// \param cv[in]: condition variable to wait on
	/// \param p[in]: predicate that must become true before continuing
	template<typename Predicate>
	inline void wait(std::condition_variable &cv,
	                 Predicate &&p) noexcept {
		cv.wait(_, std::forward<Predicate>(p));
	}

	/// Wait on a condition variable with timeout using this lock
	/// \tparam Predicate[in]: predicate type for the condition
	/// \tparam Time[in]: time point type for the timeout
	/// \param cv[in]: condition variable to wait on
	/// \param time[in]: timeout time point
	/// \param p[in]: predicate that must become true before continuing
	/// \return true if predicate became true, false if timeout occurred
	template<typename Predicate,
	         typename Time>
	inline bool wait_until(std::condition_variable &cv,
	                       Time &&time,
	                       Predicate &&p) noexcept {
		return cv.wait_until(_, std::forward<Time>(time),
		                     std::forward<Predicate>(p));
	}

	/// Checks if this lock currently owns the mutex
	/// \return true if the lock currently owns the mutex
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
