#ifndef CRYPTANALYSISLIB_ATOMIC_SEMAPHORE_H
#define CRYPTANALYSISLIB_ATOMIC_SEMAPHORE_H

#include <semaphore.h>

#include "helper.h"

// just a wrapper around the normal glib semaphore
struct semaphore {
private:
	sem_t sem;

	inline void create() noexcept {
		int err = sem_init(&sem, 0, 0);
		ASSERT(err == 0);
	}

	inline void close() noexcept {
		sem_destroy(&sem);
	}

public:
	///
	semaphore() noexcept {
		create();
	}

	///
	~semaphore() noexcept {
		close();
	}

	///
	inline void wait() noexcept {
		int err = sem_wait(&sem);
		ASSERT(err == 0);
	}

	///
	inline void signal(int cnt) {
		while (cnt-- > 0) {
			sem_post(&sem);
		}
	}
};


#endif
