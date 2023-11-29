#ifndef CRYPTANALYSISLIB_THREAD_H
#define CRYPTANALYSISLIB_THREAD_H

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
