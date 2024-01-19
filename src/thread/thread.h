#ifndef CRYPTANALYSISLIB_THREAD_H
#define CRYPTANALYSISLIB_THREAD_H
#ifdef SC
#define CAS(_a,_e,_d) atomic_compare_exchange_weak(_a,_e,_d)
#define LOAD(_a)      atomic_load(_a)
#define STORE(_a,_e)  atomic_store(_a,_e)
#define FAO(_a,_e)    atomic_fetch_or(_a,_e)
#else
#define CAS(_a,_e,_d) atomic_compare_exchange_weak_explicit(_a,_e,_d,memory_order_acq_rel,memory_order_acquire)
#define LOAD(_a)      atomic_load_explicit(_a,memory_order_acquire)
#define STORE(_a,_e)  atomic_store_explicit(_a,_e,memory_order_release)
#define FAO(_a,_e)    atomic_fetch_or_explicit(_a,_e,memory_order_acq_rel)
#endif

/// CORE IDEA:
/// wrap `openmp` or `std::threads` in an easy to use interface
///

#include <cstdint>
#if defined(_OPENMP)
#include <omp.h>
#endif

class Thread {
public:
#if defined(_OPENMP)
	static uint32_t get_tid() noexcept {
		return omp_get_thread_num();
	}

	static void sync() noexcept {
		#pragma omp barrier
		return;
	}
#else
	/// this function is called if no backend is available
	/// \return 0, as there are no threads
	constexpr static uint32_t get_tid() noexcept {
		return 0;
	}

	static void sync() noexcept {
		return;
	}
#endif
};

#endif//CRYPTANALYSISLIB_THREAD_H
