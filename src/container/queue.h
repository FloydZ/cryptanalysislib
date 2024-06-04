#ifndef CRYPTANALYSISLIB_QUEUE_H
#define CRYPTANALYSISLIB_QUEUE_H

#include <atomic>
#include <cstdint>
// TODO #include <stdatomic.h>
#include <limits>
#include <queue>

/// taken from: https://github.com/codecryptanalysis/mccl/blob/main/mccl/core/collection.hpp
/// multi consumer multi producer unbounded queue
/// implemented as simple wrapper around std::deque
/// \tparam T
/// \tparam Mutex
template<typename T, typename Mutex = std::mutex>
class concurrent_queue {
public:
	typedef Mutex mutex_type;
	typedef std::lock_guard<mutex_type> lock_type;
	typedef std::deque<T> queue_type;

	typedef T value_type;

	constexpr concurrent_queue() noexcept {}
	constexpr ~concurrent_queue() noexcept {}

	///
	constexpr inline std::size_t size() noexcept {
		lock_type lock(_mutex);
		return _queue.size();
	}

	///
	constexpr inline bool empty() noexcept {
		lock_type lock(_mutex);
		return _queue.empty();
	}

	///
	constexpr inline void push_back(const value_type &v) noexcept {
		_emplace_back(v);
	}

	///
	constexpr inline void push_back(value_type &&v) noexcept {
		_emplace_back(std::move(v));
	}

	///
	template<typename... Args>
	constexpr inline void emplace_back(Args &&...args) noexcept {
		_emplace_back(std::forward<Args>(args)...);
	}

	constexpr inline bool try_pop_front(value_type &v) noexcept {
		lock_type lock(_mutex);
		if (_queue.empty()) {
			return false;
		}

		v = std::move(_queue.front());
		_queue.pop_front();
		return true;
	}

private:
	template<typename... Args>
	constexpr inline void _emplace_back(Args &&...args) noexcept {
		lock_type lock(_mutex);
		_queue.emplace_back(std::forward<Args>(args)...);
	}

	mutex_type _mutex;
	queue_type _queue;
};


// TODO
//template<typename lfatomic_big_t = __uint128_t, typename lfatomic_t=uint64_t>
//static inline bool __lfbig_cmpxchg_strong(_Atomic(lfatomic_big_t) * obj,
//		lfatomic_big_t * expected, lfatomic_big_t desired,
//		memory_order succ, memory_order fail) {
//	(void)succ;
//	(void)fail;
//	lfatomic_t low = (lfatomic_t) desired;
//	lfatomic_t high = (lfatomic_t) (desired >> (sizeof(lfatomic_t) * 8));
//	bool result;
//
//#if defined(__x86_64__)
//# define __LFX86_CMPXCHG "cmpxchg16b"
//#elif defined(__i386__)
//# define __LFX86_CMPXCHG "cmpxchg8b"
//#endif
//	__asm__ __volatile__ ("lock " __LFX86_CMPXCHG " %0"
//						  : "+m" (*obj), "=@ccz" (result), "+A" (*expected)
//						  : "b" (low), "c" (high)
//	);
//#undef __LFX86_CMPXCHG
//
//	return result;
//}
//
//template<typename lfatomic_big_t = __uint128_t, typename lfatomic_t=uint64_t>
//static inline lfatomic_big_t __lfbig_fetch_and(_Atomic(lfatomic_big_t) * obj,
//		lfatomic_big_t arg, memory_order order)  {
//	lfatomic_big_t new_val, old_val = *((volatile lfatomic_big_t *) ((uintptr_t) obj));
//	do {
//		new_val = old_val & arg;
//	} while (!__lfbig_cmpxchg_strong(obj, &old_val, new_val, order, order));
//	// __LF_ASSUME(new_val == (old_val & arg));
//	return old_val;
//}
//
///// TODO: impl: https://github.com/rusnikola/lfqueue/blob/master/wfring_cas2.h
///// SRC:
///// 	- https://drops.dagstuhl.de/storage/00lipics/lipics-vol146-disc2019/LIPIcs.DISC.2019.28/LIPIcs.DISC.2019.28.pdf
/////		- https://github.com/rusnikola/lfqueue/blob/master/lfring_cas2.h
/////	NOTE: this is specially written for the x86 architecture
///// NOTE: stores pointers
//template<typename T>
//class lock_free_queue {
//	constexpr static uint32_t CACHE_SHIFTS = 7u;
//	constexpr static uint32_t CACHE_BYTES = 1u << CACHE_SHIFTS;
//	constexpr static uint32_t PTR_MIN = CACHE_SHIFTS - 5;
//
//	constexpr static size_t order = 15;
//
//	using lfsatomic_t = int64_t;
//	using lfatomic_t = uint64_t;
//	using lfatomic_big_t = __uint128_t;
//
//	constexpr static uint32_t ATOMIC_LOG2 = 3;
//	constexpr static uint32_t ATOMIC_WIDTH = sizeof(lfatomic_t) * 8;
//	constexpr static uint32_t ATOMIC_BIG_WIDTH = ATOMIC_WIDTH * 2;
//
//	constexpr static size_t __lfaba_shift = sizeof(uintptr_t) * 8;
//	constexpr static size_t __lfaptr_shift = 0;
//	constexpr static lfatomic_big_t __lfaba_mask = (~(lfatomic_big_t) 0UL) << (sizeof(uintptr_t) * 8);
//	constexpr static lfatomic_big_t __lfaba_step =    (lfatomic_big_t) 1UL << (sizeof(uintptr_t) * 8);
//	/// fields
//	__attribute__ ((aligned(CACHE_BYTES))) _Atomic(lfatomic_t) head;
//	__attribute__ ((aligned(CACHE_BYTES))) _Atomic(lfsatomic_t) threshold;
//	__attribute__ ((aligned(CACHE_BYTES))) _Atomic(lfatomic_t) tail;
//	__attribute__ ((aligned(CACHE_BYTES))) _Atomic(lfatomic_big_t) array[1u << (order + 1u)];
//
//
//	/// needed macros    TODO undef them all
//#define __lfring_cmp(x, op, y)		((lfsatomic_t) ((x) - (y)) op 0)
//#define LFRING_PTR_ALIGN			(_Alignof(struct __lfring_ptr))
//#define LFRING_PTR_SIZE(o)			(offsetof(struct __lfring_ptr, array) + (sizeof(lfatomic_big_t) << ((o) + 1)))
//#define __lfring_threshold4(n) 		((long) (2 * (n) - 1))
//#define __lfring_array_pointer(x)	((_Atomic(lfatomic_t) *) (x))
//#define __lfring_array_entry(x)		((_Atomic(lfatomic_t) *) (x) + 1)
//#define __lfring_entry(x)			((lfatomic_t) (((x) & __lfaba_mask) >> __lfaba_shift))
//#define __lfring_pointer(x)			((lfatomic_t) (((x) & ~__lfaba_mask) >>	__lfaptr_shift))
//#define __lfring_pair(e,p)			(((lfatomic_big_t) (e) << __lfaba_shift) | ((lfatomic_big_t) (p) << __lfaptr_shift))
//
//
//	/// functions
//	constexpr static inline size_t __lfring_map(lfatomic_t idx, size_t n) {
//		return (size_t) (((idx & (n - 1)) >> (order + 1 - PTR_MIN)) | ((idx << PTR_MIN) & (n - 1)));
//	}
//
//	constexpr static inline size_t lfring_pow2() {
//		return (size_t) 1U << (order + 1);
//	}
//
//public:
//	lock_free_queue() {
//		size_t n = lfring_pow2();
//
// 	   for (size_t i = 0; i != n; i++) {
//			array[i] = 0;
// 	   }
//
// 	   atomic_init(&head, n);
// 	   atomic_init(&threshold, -1);
// 	   atomic_init(&tail, n);
//	}	
//
//
//	static inline void lfring_ptr_init_lhead(lfatomic_t *lhead) {
//		*lhead = lfring_pow2();
//	}
//
//	inline bool enqueue(void * ptr, bool nonempty, bool nonfull, lfatomic_t *lhead) {
//		size_t tidx, n = lfring_pow2();
//		lfatomic_t __tail, entry, ecycle, tcycle;
//		lfatomic_big_t pair;
//	
//		if (!nonfull) {
//			__tail = atomic_load(&tail);
//			if (__tail >= *lhead + n) {
//				*lhead = atomic_load(&head);
//				if (__tail >= *lhead + n)
//					return false;
//			}
//		}
//	
//		while (true) {
//			__tail = atomic_fetch_add_explicit(&tail, 1, memory_order_acq_rel);
//			tcycle = __tail & ~(lfatomic_t) (n - 1);
//			tidx = __lfring_map(__tail,  n);
//			pair =   *((volatile lfatomic_big_t *) ((uintptr_t)(&array[tidx])));
//			
//		retry:
//			entry = __lfring_entry(pair);
//			ecycle = entry & ~(lfatomic_t) (n - 1);
//			if (__lfring_cmp(ecycle, <, tcycle) && ((entry == ecycle) ||
//			     (entry == (ecycle | 0x2) && atomic_load_explicit(&head,
//					 memory_order_acquire) <= __tail))) {
//	
//				if (!__lfbig_cmpxchg_strong(&array[tidx],
//						&pair, __lfring_pair(tcycle | 0x1, (lfatomic_t) ptr),
//						memory_order_acq_rel, memory_order_acquire)) {
//					goto retry;
//				}
//	
//				if (!nonempty && atomic_load(&threshold) != __lfring_threshold4(n)) {
//					atomic_store(&threshold, __lfring_threshold4(n));
//				}
//	
//				return true;
//			}
//	
//			if (!nonfull) {
//				if (__tail + 1 >= *lhead + n) {
//					*lhead = atomic_load(&head);
//					if (__tail + 1 >= *lhead + n)
//						return false;
//				}
//			}
//		}
//	}
//
//	inline void __lfring_ptr_catchup(lfatomic_t __tail, lfatomic_t __head) {
//		while (!atomic_compare_exchange_weak_explicit(&tail, &__tail, __head,
//				memory_order_acq_rel, memory_order_acquire)) {
//			__head = atomic_load_explicit(&head, memory_order_acquire);
//			__tail = atomic_load_explicit(&tail, memory_order_acquire);
//			if (__lfring_cmp(__tail, >=, __head)) {
//				break;
//			}
//		}
//	}
//
//	inline bool lfring_ptr_dequeue(void **ptr, bool nonempty) {
//		size_t hidx, n = lfring_pow2();
//		lfatomic_t __head, entry, entry_new, ecycle, hcycle, __tail;
//		lfatomic_big_t pair;
//	
//		if (!nonempty && atomic_load(&threshold) < 0) {
//			return false;
//		}
//	
//		while (1) {
//			__head = atomic_fetch_add_explicit(&head, 1, memory_order_acq_rel);
//			hcycle = __head & ~(lfatomic_t) (n - 1);
//			hidx = __lfring_map(__head, n);
//			entry = atomic_load_explicit(__lfring_array_entry(&array[hidx]),
//						memory_order_acquire);
//			do {
//				ecycle = entry & ~(lfatomic_t) (n - 1);
//				if (ecycle == hcycle) {
//					pair = __lfbig_fetch_and(&array[hidx],
//						__lfring_pair(~(lfatomic_t) 0x1, 0), memory_order_acq_rel);
//					*ptr = (void *) __lfring_pointer(pair);
//					return true;
//				}
//	
//				if ((entry & (~(lfatomic_t) 0x2)) != ecycle) {
//					entry_new = entry | 0x2;
//					if (entry == entry_new)
//						break;
//				} else {
//					entry_new = hcycle | (entry & 0x2);
//				}
//			} while (__lfring_cmp(ecycle, <, hcycle) &&
//					!atomic_compare_exchange_weak_explicit(
//					__lfring_array_entry(&array[hidx]),
//					&entry, entry_new,
//					memory_order_acq_rel, memory_order_acquire));
//	
//			if (!nonempty) {
//				__tail = atomic_load_explicit(&tail, memory_order_acquire);
//				if (__lfring_cmp(__tail, <=, __head + 1)) {
//					__lfring_ptr_catchup(__tail, __head + 1);
//					atomic_fetch_sub_explicit(&threshold, 1,
//						memory_order_acq_rel);
//					return false;
//				}
//	
//				if (atomic_fetch_sub_explicit(&threshold, 1,
//						memory_order_acq_rel) <= 0) {
//					return false;
//				}
//			}
//		}
//	}
//
//
//
//	/// cleanup
//#undef __lfring_cmp
//#undef LFRING_PTR_ALIGN		
//#undef LFRING_PTR_SIZE
//
//
//
//
//};

#endif//CRYPTANALYSISLIB_QUEUE_H
