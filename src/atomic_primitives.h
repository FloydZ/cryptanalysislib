#ifndef CRYPTANALYSISLIB_PRIMITIVES_H
#define CRYPTANALYSISLIB_PRIMITIVES_H

#include <atomic>
#include <cstdint>
#include <memory>

// #ifdef USE_STD_ATOMIC

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
 * and stores completes before the current store is visible.
 */
#define RELEASE(ptr, val) __atomic_store_n(ptr, val, __ATOMIC_RELEASE)

/**
 * A load with a following acquire fence to ensure no following load and
 * stores can start before the current load completes.
 */
#define ACQUIRE(ptr) __atomic_load_n(ptr, __ATOMIC_ACQUIRE)


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
	std::atomic<uint8_t> state{unlocked};

	static constexpr uint8_t unlocked = 0;
	static constexpr uint8_t locked = 0b01;
	static constexpr uint8_t sleeper = 0b10;
};
#endif
