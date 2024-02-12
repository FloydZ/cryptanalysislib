#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include <cstdint>
#include <memory>
#include <atomic>


/**
 * An atomic fetch-and-add.
 */
#define FAA(ptr, val) __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED)
/**
 * An atomic fetch-and-add that also ensures sequential consistency.
 */
#define FAAcs(ptr, val) __atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST)

/**
 * An atomic compare-and-swap.
 */
#define CAS(ptr, cmp, val) __atomic_compare_exchange_n(ptr, cmp, val, 0, \
    __ATOMIC_RELAXED, __ATOMIC_RELAXED)
/**
 * An atomic compare-and-swap that also ensures sequential consistency.
 */
#define CAScs(ptr, cmp, val) __atomic_compare_exchange_n(ptr, cmp, val, 0, \
    __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)
/**
 * An atomic compare-and-swap that ensures release semantic when succeed
 * or acquire semantic when failed.
 */
#define CASra(ptr, cmp, val) __atomic_compare_exchange_n(ptr, cmp, val, 0, \
    __ATOMIC_RELEASE, __ATOMIC_ACQUIRE)
/**
 * An atomic compare-and-swap that ensures acquire semantic when succeed
 * or relaxed semantic when failed.
 */
#define CASa(ptr, cmp, val) __atomic_compare_exchange_n(ptr, cmp, val, 0, \
    __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)

/**
 * An atomic swap.
 */
#define SWAP(ptr, val) __atomic_exchange_n(ptr, val, __ATOMIC_RELAXED)

/**
 * An atomic swap that ensures acquire release semantics.
 */
#define SWAPra(ptr, val) __atomic_exchange_n(ptr, val, __ATOMIC_ACQ_REL)

/**
 * A memory fence to ensure sequential consistency.
 */
#define FENCE() __atomic_thread_fence(__ATOMIC_SEQ_CST)

/**
 * An atomic store.
 */
#define STORE(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELAXED)

/**
 * A store with a preceding release fence to ensure all previous load
 * and stores completes before the current store is visiable.
 */
#define RELEASE(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELEASE)

/**
 * A load with a following acquire fence to ensure no following load and
 * stores can start before the current load completes.
 */
#define ACQUIRE(ptr) __atomic_load_n(ptr, __ATOMIC_ACQUIRE)

// TODO
inline void cmov(uint64_t a, const uint64_t b) noexcept {
	asm volatile (
			"cmova %0 %1\n\t"
			: "=r" (a)
			: "r" (b)
	);
}

struct one_byte_mutex {
	inline void lock() noexcept {
		if (state.exchange(locked, std::memory_order_acquire) == unlocked) {
			return;
		}

		while (state.exchange(sleeper, std::memory_order_acquire) != unlocked) {
			// C++ wait 
			state.wait(sleeper, std::memory_order_relaxed);
		}
	}
	
	///
	inline void unlock() noexcept {
		if (state.exchange(unlocked, std::memory_order_release) == sleeper) {
			state.notify_one();
		}
	}

private:
	std::atomic<uint8_t> state{ unlocked };

	static constexpr uint8_t unlocked = 0;
	static constexpr uint8_t locked  = 0b01;
	static constexpr uint8_t sleeper = 0b10;
};
#endif

