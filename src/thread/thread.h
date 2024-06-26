#ifndef CRYPTANALYSISLIB_THREAD_H
#define CRYPTANALYSISLIB_THREAD_H

#include "helper.h"

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


// SRC: https://stackoverflow.com/questions/24645880/set-cpu-affinity-when-create-a-thread
// pthread CPU affinity
/// CPU_ID: integer of the CPU
/// THREAD: pthread handle
#define PTHREAD_SET_THREAD_AFFINITY(CPU_ID, THREAD) \
    cpu_set_t cpuset;								\
    CPU_ZERO(&cpuset);								\
    CPU_SET(CPU_ID, &cpuset); 						\
    int rc = pthread_setaffinity_np(THREAD.native_handle(), sizeof(cpu_set_t), &cpuset); \
	ASSERT(rc);

#endif//CRYPTANALYSISLIB_THREAD_H
