#ifndef CRYPTANALYSISLIB_ATOMIC_PRIMITIVES_H
#define CRYPTANALYSISLIB_ATOMIC_PRIMITIVES_H

#include <atomic>
#include <cstdint>
#include <memory>

// needed for __ATTRIBUTE__
#include "helper.h"

/// @brief Simple concept for the Lockable and Basic Lockable types as defined by the C++
/// standard.
/// @details See https://en.cppreference.com/w/cpp/named_req/Lockable and
/// https://en.cppreference.com/w/cpp/named_req/BasicLockable for details.
template<typename Lock>
concept is_lockable = requires(Lock &&lock) {
	lock.lock();
	lock.unlock();
	{ lock.try_lock() } -> std::convertible_to<bool>;
};


/**
 * An atomic fetch-and-add.
 */
#define FAA(ptr, val) __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED)

// An atomic fetch-and-add that also ensures sequential consistency.
// Usage:
//
#define FAAcs(ptr, val) __atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST)

/**
 * This is translated into an `lock; cmpchxg` instruction
 *	https://godbolt.org/#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,selection:(endColumn:48,endLineNumber:8,positionColumn:48,positionLineNumber:8,selectionStartColumn:48,selectionStartLineNumber:8,startColumn:48,startLineNumber:8),source:'%0A%23include+%3Catomic%3E%0A%23include+%3Ccstdint%3E%0A%23include+%3Cmemory%3E%0A%0Aint+square(int+num)+%7B%0A++++return+__atomic_compare_exchange_n(%26num,+%26num,+1,+0,%0A+++++++++++++__ATOMIC_RELAXED,+__ATOMIC_RELAXED)%3B%0A%7D'),l:'5',n:'1',o:'C%2B%2B+source+%231',t:'0')),k:62.24127735068007,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:g142,filters:(b:'0',binary:'1',binaryObject:'1',commentOnly:'0',debugCalls:'1',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1',verboseDemangling:'0'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,libs:!(),options:'-O3',overrides:!(),selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),source:1),l:'5',n:'0',o:'+x86-64+gcc+14.2+(Editor+%231)',t:'0')),k:37.75872264931993,l:'4',n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4
 * An atomic compare-and-swap.
 */
#define CAS(ptr, cmp, val) __atomic_compare_exchange_n(ptr, cmp, val, 0, \
	                                                   __ATOMIC_RELAXED, __ATOMIC_RELAXED)
/// @param ptr
/// @param cmp DOES not need to be a pointer
/// @param val
#define CASnp(ptr, cmp, val) __sync_val_compare_and_swap(ptr, cmp, val)

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
 * and stores completes before the current store is visible.
 */
#define RELEASE(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELEASE)

/**
 * A load with a following acquire fence to ensure no following load and
 * stores can start before the current load completes.
 */
#define ACQUIRE(ptr) __atomic_load_n(ptr, __ATOMIC_ACQUIRE)


///
#define MEMORY_BARRIER_ACQUIRE() __asm__ __volatile__("" : : : "memory")

///
#define MEMORY_BARRIER_RELEASE() __asm__ __volatile__("" : : : "memory")


#ifdef __x86_64__
///https://godbolt.org/#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,selection:(endColumn:1,endLineNumber:18,positionColumn:1,positionLineNumber:18,selectionStartColumn:1,selectionStartLineNumber:18,startColumn:1,startLineNumber:18),source:'template%3Ctypename+T%3E%0Ainline+T+cmov(T+a,+const+T+b)+noexcept+%7B%0A%09asm+volatile+(%0A++++++++++++//%22test+%250,+%251%5Cn%5Ct%22%0A%09%09%09%22cmovne+%250,+%251%5Cn%5Ct%22%0A%09%09%09:+%22%3Dr%22+(a)%0A%09%09%09:+%22r%22+(b)%0A%09)%3B%0A%09return+a%3B%0A%7D%0A%0Ausing+T+%3D+unsigned+int%3B%0Aint+tmp(T+a,+const+T+b)+%7B%0A%09a+%3D+cmov%3CT%3E(a,+b)%3B%0A++++return+a%3B%0A%7D%0A%0A'),l:'5',n:'0',o:'C%2B%2B+source+%231',t:'0')),k:50,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:g132,filters:(b:'0',binary:'1',binaryObject:'1',commentOnly:'0',debugCalls:'1',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,libs:!(),options:'-O3',overrides:!(),selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),source:1),l:'5',n:'0',o:'+x86-64+gcc+13.2+(Editor+%231)',t:'0')),k:50,l:'4',n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4
/// \tparam T
/// \tparam test
/// \param a
/// \param b
/// \return
#define CMOVE_MACRO(version)                          \
	template<typename T, const bool test = false>     \
	inline T cmov##version(T a, const T b) noexcept { \
		if constexpr (test) {                         \
			asm volatile(                             \
			        "test %0, %1\n\t"                 \
			        : "=r"(a)                         \
			        : "r"(b));                        \
		}                                             \
		asm volatile(                                 \
		        "cmov" #version " %0, %1\n\t"         \
		        : "=r"(a)                             \
		        : "r"(b));                            \
		return a;                                     \
	}
#else

#define CMOVE_MACRO(version)                          \
	template<typename T, const bool test = false>     \
	inline T cmov##version(T a, const T b) noexcept { \
		if (a != b) {                                 \
			a = b;                                    \
		}                                             \
		return a;                                     \
	}
#endif

CMOVE_MACRO(a)
CMOVE_MACRO(ae)
CMOVE_MACRO(b)
CMOVE_MACRO(be)
CMOVE_MACRO(c)
CMOVE_MACRO(e)
CMOVE_MACRO(g)
CMOVE_MACRO(ge)
CMOVE_MACRO(l)
CMOVE_MACRO(le)
CMOVE_MACRO(na)
CMOVE_MACRO(nae)
CMOVE_MACRO(nb)
CMOVE_MACRO(nc)
CMOVE_MACRO(ng)
CMOVE_MACRO(ne)
CMOVE_MACRO(nl)
CMOVE_MACRO(nle)
CMOVE_MACRO(no)
CMOVE_MACRO(np)
CMOVE_MACRO(nz)
CMOVE_MACRO(ns)
CMOVE_MACRO(o)
CMOVE_MACRO(p)
CMOVE_MACRO(pe)
CMOVE_MACRO(po)
CMOVE_MACRO(s)
CMOVE_MACRO(z)


// NOTE: older gcc version do not support atomic::wait
#ifdef __cpp_lib_atomic_wait
struct one_byte_mutex {
	inline void lock() noexcept {
		if (state.exchange(locked, std::memory_order_acquire) == unlocked) {
			return;
		}

		while (state.exchange(sleeper, std::memory_order_acquire) != unlocked) {
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
	std::atomic<uint8_t> state{unlocked};

	static constexpr uint8_t unlocked = 0;
	static constexpr uint8_t locked = 0b01;
	static constexpr uint8_t sleeper = 0b10;
};
#endif


#define CAPABILITY(x) __ATTRIBUTE__(capability(x))
#define SCOPED_CAPABILITY __ATTRIBUTE__(scoped_lockable)
#define GUARDED_BY(x) __ATTRIBUTE__(guarded_by(x))
#define PT_GUARDED_BY(x) __ATTRIBUTE__(pt_guarded_by(x))


#define ACQUIRED_BEFORE(...) \
	__ATTRIBUTE__(acquired_before(__VA_ARGS__))

#define ACQUIRED_AFTER(...) \
	__ATTRIBUTE__(acquired_after(__VA_ARGS__))

#define REQUIRES(...) \
	__ATTRIBUTE__(requires_capability(__VA_ARGS__))

#define REQUIRES_SHARED(...) \
	__ATTRIBUTE__(requires_shared_capability(__VA_ARGS__))

#define ACQUIRE_(...) \
	__ATTRIBUTE__(acquire_capability(__VA_ARGS__))

#define ACQUIRE_SHARED(...) \
	__ATTRIBUTE__(acquire_shared_capability(__VA_ARGS__))

#define RELEASE_(...) \
	__ATTRIBUTE__(release_capability(__VA_ARGS__))

#define RELEASE_SHARED(...) \
	__ATTRIBUTE__(release_shared_capability(__VA_ARGS__))

#define TRY_ACQUIRE(...) \
	__ATTRIBUTE__(try_acquire_capability(__VA_ARGS__))

#define TRY_ACQUIRE_SHARED(...) \
	__ATTRIBUTE__(try_acquire_shared_capability(__VA_ARGS__))

#define EXCLUDES(...) __ATTRIBUTE__(locks_excluded(__VA_ARGS__))

#define ASSERT_CAPABILITY(x) __ATTRIBUTE__(assert_capability(x))

#define ASSERT_SHARED_CAPABILITY(x) \
	__ATTRIBUTE__(assert_shared_capability(x))

#define RETURN_CAPABILITY(x) __ATTRIBUTE__(lock_returned(x))

#define NO_THREAD_SAFETY_ANALYSIS \
	__ATTRIBUTE__(no_thread_safety_analysis)


namespace __atomic_impl {

	// Remove volatile and create a non-deduced context for value arguments.
	template<typename _Tp>
	using _Val = typename std::remove_volatile<_Tp>::type;

	// Like _Val<T> above, but for difference_type arguments.
	template<typename _Tp>
	using _Diff = std::__conditional_t<std::is_pointer_v<_Tp>, ptrdiff_t, _Val<_Tp>>;

	// Implementation details of atomic padding handling
	/// \tparam T
	/// \returns If T is not trivially copyable and if any two objects of type T with
	///     the same value have not the same object representation, provides the member
	///     constant value equal.
	template<typename T>
	constexpr bool __maybe_has_padding() noexcept {
		return !__has_unique_object_representations(T) &&
		       !std::is_same<T, float>::value &&
		       !std::is_same<T, double>::value;
	}

	/// \tparam
	/// \return TODO
	template<typename _Tp>
	inline _Tp *__clear_padding(_Tp &__val) noexcept {
		auto *__ptr = std::__addressof(__val);
		if constexpr (__atomic_impl::__maybe_has_padding<_Tp>()) {
			__builtin_clear_padding(__ptr);
		}
		return __ptr;
	}

	template<typename _Tp>
	inline bool
	__compare_exchange(_Tp &__val,
	                   _Val<_Tp> &__e,
	                   _Val<_Tp> &__i,
	                   bool __is_weak,
	                   std::memory_order __s, std::memory_order __f) noexcept {
		// TODO static_assert(__is_valid_cmpexch_failure_order(__f));

		using _Vp = _Val<_Tp>;

		if constexpr (__atomic_impl::__maybe_has_padding<_Vp>()) {
			// We must not modify __e on success, so cannot clear its padding.
			// Copy into a buffer and clear that, then copy back on failure.
			alignas(_Vp) unsigned char __buf[sizeof(_Vp)];
			_Vp *__exp = ::new ((void *) __buf) _Vp(__e);
			__atomic_impl::__clear_padding(*__exp);
			if (__atomic_compare_exchange(std::__addressof(__val), __exp,
			                              __atomic_impl::__clear_padding(__i),
			                              __is_weak, int(__s), int(__f))) {
				return true;
			}
			__builtin_memcpy(__builtin_addressof(__e), __exp, sizeof(_Vp));
			return false;
		} else
			return __atomic_compare_exchange(__builtin_addressof(__val),
			                                 __builtin_addressof(__e),
			                                 __builtin_addressof(__i),
			                                 __is_weak, int(__s), int(__f));
	}


	// Produce a fake, minimally aligned pointer.
	template<size_t _Size, size_t _Align>
	inline bool is_lock_free() noexcept {
		return __atomic_is_lock_free(_Size, reinterpret_cast<void *>(-_Align));
	}

	template<typename _Tp>
	_GLIBCXX_ALWAYS_INLINE void
	store(_Tp *__ptr, _Val<_Tp> __t, memory_order __m) noexcept {
		__atomic_store(__ptr, __atomic_impl::__clear_padding(__t), int(__m));
	}

	template<typename _Tp>
	_GLIBCXX_ALWAYS_INLINE _Val<_Tp>
	load(const _Tp *__ptr, memory_order __m) noexcept {
		alignas(_Tp) unsigned char __buf[sizeof(_Tp)];
		auto *__dest = reinterpret_cast<_Val<_Tp> *>(__buf);
		__atomic_load(__ptr, __dest, int(__m));
		return *__dest;
	}

	template<typename _Tp>
	_GLIBCXX_ALWAYS_INLINE _Val<_Tp>
	exchange(_Tp *__ptr, _Val<_Tp> __desired, memory_order __m) noexcept {
		alignas(_Tp) unsigned char __buf[sizeof(_Tp)];
		auto *__dest = reinterpret_cast<_Val<_Tp> *>(__buf);
		__atomic_exchange(__ptr, __atomic_impl::__clear_padding(__desired),
		                  __dest, int(__m));
		return *__dest;
	}

	template<typename _Tp>
	_GLIBCXX_ALWAYS_INLINE bool
	compare_exchange_weak(_Tp *__ptr, _Val<_Tp> &__expected,
	                      _Val<_Tp> __desired, memory_order __success,
	                      memory_order __failure) noexcept {
		return __atomic_impl::__compare_exchange(*__ptr, __expected, __desired,
		                                         true, __success, __failure);
	}

	template<typename _Tp>
	_GLIBCXX_ALWAYS_INLINE bool
	compare_exchange_strong(_Tp *__ptr, _Val<_Tp> &__expected,
	                        _Val<_Tp> __desired, memory_order __success,
	                        memory_order __failure) noexcept {
		return __atomic_impl::__compare_exchange(*__ptr, __expected, __desired,
		                                         false, __success, __failure);
	}

#if __cpp_lib_atomic_wait
	template<typename _Tp>
	_GLIBCXX_ALWAYS_INLINE void
	wait(const _Tp *__ptr, _Val<_Tp> __old,
	     memory_order __m = memory_order_seq_cst) noexcept {
		std::__atomic_wait_address_v(__ptr, __old,
		                             [__ptr, __m]() { return __atomic_impl::load(__ptr, __m); });
	}

	// TODO add const volatile overload

	template<typename _Tp>
	_GLIBCXX_ALWAYS_INLINE void
	notify_one(const _Tp *__ptr) noexcept { std::__atomic_notify_address(__ptr, false); }

	// TODO add const volatile overload

	template<typename _Tp>
	_GLIBCXX_ALWAYS_INLINE void
	notify_all(const _Tp *__ptr) noexcept { std::__atomic_notify_address(__ptr, true); }

	// TODO add const volatile overload
#endif// __cpp_lib_atomic_wait

	template<typename _Tp>
	_GLIBCXX_ALWAYS_INLINE _Tp
	fetch_add(_Tp *__ptr, _Diff<_Tp> __i, memory_order __m) noexcept { return __atomic_fetch_add(__ptr, __i, int(__m)); }

	template<typename _Tp>
	_GLIBCXX_ALWAYS_INLINE _Tp
	fetch_sub(_Tp *__ptr, _Diff<_Tp> __i, memory_order __m) noexcept { return __atomic_fetch_sub(__ptr, __i, int(__m)); }

	template<typename _Tp>
	_GLIBCXX_ALWAYS_INLINE _Tp
	fetch_and(_Tp *__ptr, _Val<_Tp> __i, memory_order __m) noexcept { return __atomic_fetch_and(__ptr, __i, int(__m)); }

	template<typename _Tp>
	_GLIBCXX_ALWAYS_INLINE _Tp
	fetch_or(_Tp *__ptr, _Val<_Tp> __i, memory_order __m) noexcept { return __atomic_fetch_or(__ptr, __i, int(__m)); }

	template<typename _Tp>
	_GLIBCXX_ALWAYS_INLINE _Tp
	fetch_xor(_Tp *__ptr, _Val<_Tp> __i, memory_order __m) noexcept { return __atomic_fetch_xor(__ptr, __i, int(__m)); }

	template<typename _Tp>
	_GLIBCXX_ALWAYS_INLINE _Tp
	__add_fetch(_Tp *__ptr, _Diff<_Tp> __i) noexcept { return __atomic_add_fetch(__ptr, __i, __ATOMIC_SEQ_CST); }

	template<typename _Tp>
	_GLIBCXX_ALWAYS_INLINE _Tp
	__sub_fetch(_Tp *__ptr, _Diff<_Tp> __i) noexcept { return __atomic_sub_fetch(__ptr, __i, __ATOMIC_SEQ_CST); }

	template<typename _Tp>
	_GLIBCXX_ALWAYS_INLINE _Tp
	__and_fetch(_Tp *__ptr, _Val<_Tp> __i) noexcept { return __atomic_and_fetch(__ptr, __i, __ATOMIC_SEQ_CST); }

	template<typename _Tp>
	_GLIBCXX_ALWAYS_INLINE _Tp
	__or_fetch(_Tp *__ptr, _Val<_Tp> __i) noexcept { return __atomic_or_fetch(__ptr, __i, __ATOMIC_SEQ_CST); }

	template<typename _Tp>
	_GLIBCXX_ALWAYS_INLINE _Tp
	__xor_fetch(_Tp *__ptr, _Val<_Tp> __i) noexcept { return __atomic_xor_fetch(__ptr, __i, __ATOMIC_SEQ_CST); }

	template<typename _Tp>
	_Tp
	__fetch_add_flt(_Tp *__ptr, _Val<_Tp> __i, memory_order __m) noexcept {
		_Val<_Tp> __oldval = load(__ptr, memory_order_relaxed);
		_Val<_Tp> __newval = __oldval + __i;
		while (!compare_exchange_weak(__ptr, __oldval, __newval, __m,
		                              memory_order_relaxed))
			__newval = __oldval + __i;
		return __oldval;
	}

	template<typename _Tp>
	_Tp
	__fetch_sub_flt(_Tp *__ptr, _Val<_Tp> __i, memory_order __m) noexcept {
		_Val<_Tp> __oldval = load(__ptr, memory_order_relaxed);
		_Val<_Tp> __newval = __oldval - __i;
		while (!compare_exchange_weak(__ptr, __oldval, __newval, __m,
		                              memory_order_relaxed))
			__newval = __oldval - __i;
		return __oldval;
	}

	template<typename _Tp>
	_Tp
	__add_fetch_flt(_Tp *__ptr, _Val<_Tp> __i) noexcept {
		_Val<_Tp> __oldval = load(__ptr, memory_order_relaxed);
		_Val<_Tp> __newval = __oldval + __i;
		while (!compare_exchange_weak(__ptr, __oldval, __newval,
		                              memory_order_seq_cst,
		                              memory_order_relaxed))
			__newval = __oldval + __i;
		return __newval;
	}

	template<typename _Tp>
	_Tp
	__sub_fetch_flt(_Tp *__ptr, _Val<_Tp> __i) noexcept {
		_Val<_Tp> __oldval = load(__ptr, memory_order_relaxed);
		_Val<_Tp> __newval = __oldval - __i;
		while (!compare_exchange_weak(__ptr, __oldval, __newval,
		                              memory_order_seq_cst,
		                              memory_order_relaxed))
			__newval = __oldval - __i;
		return __newval;
	}
}// namespace __atomic_impl

#endif
