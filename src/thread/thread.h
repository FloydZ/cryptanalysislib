#ifndef CRYPTANALYSISLIB_THREAD_H
#define CRYPTANALYSISLIB_THREAD_H

/// CORE IDEA:
/// wrap `openmp` or `std::threads` in an easy to use interface
///

#include <cstdint>
#if defined(USE_OMP)
#include <omp.h>
#endif

class Thread {
public:
#if defined(USE_OPENMP)
	constexpr static uint32_t get_tid() noexcept {
		return omp_get_thread_num();
	}
#else
	/// this function is called if no backend is available
	/// \return 0, as there are no threads
	constexpr static uint32_t get_tid() noexcept {
		return 0;
	}
#endif
};

#endif//CRYPTANALYSISLIB_THREAD_H
