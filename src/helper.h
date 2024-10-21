#ifndef CRYPTANALYSISLIB_HELPER_H
#define CRYPTANALYSISLIB_HELPER_H

// Global Includes
#include <cassert>
#include <cstdint>// needed for uint8_t and so on
#include <functional>
#include <stddef.h>
#include <string.h>
#include <type_traits>// for std::convertable_to
#include <vector>     // for __level_translation_array

#include "cpucycles.h"

#ifndef __CUDACC__
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#endif

#if defined(DEBUG) && defined(NDEBUG)
#error "debug and no debug?"
#endif


#if defined(NUMBER_THREADS) && NUMBER_THREADS > 1
#define OMP_BARRIER _Pragma("omp barrier")
#else
#define OMP_BARRIER
#endif

// optimisation flags/commands
#define STRINGIFY(a) #a

// Som little helpers for functions
#define __RESTRICT__ __restrict__
#define __FORCEINLINE__ inline __attribute__((__always_inline__))
#define __INLINE__

#include <atomic>
#include <thread>

// Some global or decoding specific macro. Maybe outsource to different file?
#if (defined(NUMBER_THREADS) && NUMBER_THREADS > 1)
#define MULTITHREADED_WRITE(x) x
#else
#define MULTITHREADED_WRITE(x)
#endif

#if defined(NUMBER_OUTER_THREADS) && NUMBER_OUTER_THREADS != 1
#define OUTER_MULTITHREADED_WRITE(x) x
#else
#define OUTER_MULTITHREADED_WRITE(x)
#endif

// Enable performance logging. Aka function return some useful performance information
#ifdef PERFORMANCE_LOGGING
#define PERFORMANE_WRITE(x) x
#else
#define PERFORMANE_WRITE(x)
#endif

#ifdef __unix__
#define MADIVE(a, b, c)//madvise(a, b, c);
#else
#define MADIVE(a, b, c)
#endif

// performance helpers
#if defined(USE_BRANCH_PREDICTION) && (!defined(DEBUG))
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)
#else
#define likely(x) x
#define unlikely(x) x
#endif

#ifdef USE_PREFETCH
/*
 * The value of addr is the address of the memory to prefetch. There are two optional arguments, rw and locality.
 * The value of rw is a compile-time constant one or zero; one means that the prefetch is preparing for a write to the
 * memory address and zero the default, means that the prefetch is preparing for a read. The value locality must be a
 * compile-time constant integer between zero and three. A value of zero means that the data has no temporal locality,
 * so it need not be left in the cache after the access. A value of three means that the data has a high degree of
 * temporal locality and should be left in all levels of cache possible. Values of one and two mean, respectively,
 * a low or moderate degree of temporal locality. The default is three.
 */
#define prefetch(m, x, y) __builtin_prefetch(m, x, y)
#define cryptanalysislib_prefetch(address) __builtin_prefetch((const void *) (address), 0, 0)
#define cryptanalysislib_prefetchw(address) __builtin_prefetch((const void *) (address), 1, 0)
constexpr std::ptrdiff_t prefetch_distance = 32 * 4;//Prefetch amount in bytes
#else
#define prefetch(m, x, y)
#define cryptanalysislib_prefetch(address)
#define cryptanalysislib_prefetchw(address)
constexpr std::ptrdiff_t prefetch_distance = 0;
#endif

#ifdef USE_LOOP_UNROLL
// Some loop unrollment optimisation
#define LOOP_UNROLL() \
	_Pragma(STRINGIFY(clang loop unroll(full)))
#else
#define LOOP_UNROLL()
#endif

#include <iostream>

#ifdef DEBUG
#include <cassert>
#ifdef USE_ARM
#include <cstdlib>
// NOTE that's the BUG. GCC on arm `assert` is not constexpr
#define ASSERT(x)										\
do {													\
	if (!(x)) {											\
		exit(EXIT_FAILURE);								\
	}													\
} while(0);

#else
#define ASSERT(x) assert(x)
#endif
#else
#define ASSERT(x)
#endif

#ifdef DEBUG
#ifndef DEBUG_MACRO
#define DEBUG_MACRO(x) x
#endif
#else
#define DEBUG_MACRO(x)
#endif


// NOTE: needed to rename `PAGE_SIZE` to `CUSTOM_PAGE_SIZE`, as apple in
// its infinite wisdom have a global variable called `PAGE_SIZE`.
#ifdef FORCE_HPAGE
// normal page, 4KiB, buts its forced to be an huge page
#define CUSTOM_PAGE_SIZE (1 << 21)
#else
// normal page, 4KiB
#define CUSTOM_PAGE_SIZE (1 << 12)
#endif

// huge page, 2MiB
#define HPAGE_SIZE (1 << 21)


#ifdef __unix__
#include <errno.h>
#include <fcntl.h>
#include <linux/kernel-page-flags.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

// See <https://www.kernel.org/doc/Documentation/vm/pagemap.txt> for
// format which these bitmasks refer to
#define PAGEMAP_PRESENT(ent) (((ent) & (1ull << 63)) != 0)
#define PAGEMAP_PFN(ent) ((ent) & ((1ull << 55) - 1))

// Checks if the page pointed at by `ptr` is huge. Assumes that `ptr` has already
// been allocated.
static void check_huge_page(void *ptr) {
	int pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
	if (pagemap_fd < 0) {
		std::cout << "could not open /proc/self/pagemap: " << strerror(errno) << "\n";
	}
	int kpageflags_fd = open("/proc/kpageflags", O_RDONLY);
	if (kpageflags_fd < 0) {
		std::cout << "could not open /proc/kpageflags: " << strerror(errno) << "\n";
	}

	// each entry is 8 bytes long
	uint64_t ent;
	if (pread(pagemap_fd, &ent, sizeof(ent), ((uintptr_t) ptr) / CUSTOM_PAGE_SIZE * 8) != sizeof(ent)) {
		std::cout << "could not read from pagemap\n";
	}

	if (!PAGEMAP_PRESENT(ent)) {
		std::cout << "page not present in /proc/self/pagemap, did you allocate it?\n";
	}
	if (!PAGEMAP_PFN(ent)) {
		std::cout << "page frame number not present, run this program as root\n";
	}

	uint64_t flags;
	if (pread(kpageflags_fd, &flags, sizeof(flags), PAGEMAP_PFN(ent) << 3) != sizeof(flags)) {
		std::cout << "could not read from kpageflags\n";
	}

	if (!(flags & (1ull << KPF_THP))) {
		std::cout << "could not allocate huge page\n";
	}

	if (close(pagemap_fd) < 0) {
		std::cout << "could not close /proc/self/pagemap: " << strerror(errno) << "\n";
	}
	if (close(kpageflags_fd) < 0) {
		std::cout << "could not close /proc/kpageflags: " << strerror(errno) << "\n";
	}
}

/// tries to alloc a huge page.
/// \param size
void *cryptanalysislib_hugepage_malloc(const size_t size) {
	const size_t nr_pages = (size + HPAGE_SIZE - 1) / HPAGE_SIZE;
	const size_t alloc_size = nr_pages * HPAGE_SIZE;
	void *ret = aligned_alloc(HPAGE_SIZE, alloc_size);
	if (ret == nullptr) {
		std::cout << "error alloc\n";
		return nullptr;
	}

	madvise(ret, size, MADV_HUGEPAGE);

	size_t buf = (size_t) ret;
	for (size_t end = buf + size; buf < end; buf += HPAGE_SIZE) {
		// allocate page
		memset((void *) buf, 0, 1);
		// check the page is indeed huge
		check_huge_page((void *) buf);
	}

	return ret;
}
#endif


/// Usage:
template<auto Start, auto End, auto Inc, class F>
constexpr inline void constexpr_for(F &&f) noexcept {
	if constexpr (Start < End) {
		f(std::integral_constant<decltype(Start), Start>());
		constexpr_for<Start + Inc, End, Inc>(f);
	}
}

/// NOTE: one day I should remove this function.
///		-> rewrite the level_translation_array as an class, implementing all of this
/// access function for the const_array level_translation_array
/// \param lower return value: lower limit to match on (inclusive)
/// \param upper return valur: upper limit to match on (exclusive)
/// \param level current level with the k-tree algorithm
/// \param level_translation_array
/// \return if level >= 0:
///				return the lower/upper bound as normal.
///			if level == -1:
///				return on all bits
static void translate_level(uint32_t *lower,
                            uint32_t *upper, const uint32_t level,
                            const std::vector<uint32_t> &level_translation_array) noexcept {
	ASSERT(lower != NULL && upper != NULL);

	// this is actually mostly only for testing.
	if (unlikely(level == uint32_t(-1))) {
		*lower = level_translation_array[0];
		*upper = level_translation_array[level_translation_array.size() - 1];
		return;
	}

	// we __MUST__ check this after the 'if' clause,
	// because otherwise this would catch the -1 test case
	ASSERT(level <= level_translation_array.size() - 1u);

	*lower = level_translation_array[level];
	*upper = level_translation_array[level + 1u];
}

/// translates an const_array 'level_filter_array' = (e.g.) [4, 0, 0] into a 'norm' = (e.g.) 2, s.t. every 'Value' with a
/// coordinate which is absolute bigger than 'Norm' needs to be filtered out.
/// assumes to count from the top to the bottom of the tree in increasing order. So the root is in lvl 0.
/// \param const lvl = current lvl
/// \param const level_filter_array input parameter
static uint32_t translate_filter(const uint8_t lvl, const uint16_t nr2,
                                 const std::vector<std::vector<uint8_t>> &level_filter_array) noexcept {
	if (level_filter_array[lvl][2] > 0) {
		// this case is the case we allow twos
		return nr2;
	} else {
		// we don't allow twos
		return uint32_t(-1);
	}
}

/// print some information about the current instance.
static void ident() {
#ifdef CUSTOM_ALIGNMENT
	std::cout << "DEFINED CUSTOM_ALIGNMENT" << std::endl;
#endif
#ifdef BINARY_CONTAINER_ALIGNMENT
	std::cout << "DEFINED BINARY_CONTAINER_ALIGNMENT" << std::endl;
#endif
#ifdef USE_PREFETCH
	std::cout << "DEFINED USE_PREFETCH" << std::endl;
#endif
#ifdef USE_BRANCH_PREDICTION
	std::cout << "DEFINED USE_BRANCH_PREDICTION" << std::endl;
#endif

	std::cout << "cryptanalysislib 0.0.1" << std::endl;
}


/// only adds two instructions:
/// https://godbolt.org/#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,selection:(endColumn:2,endLineNumber:8,positionColumn:2,positionLineNumber:8,selectionStartColumn:2,selectionStartLineNumber:8,startColumn:2,startLineNumber:8),source:'%23include+%3Cstdlib.h%3E%0A%23define+unlikely(x)+__builtin_expect(!!!!(x),+0)%0A%0A%0A%23define+ACCESS_HELPER(i,+n)%09%5C%0Aif+(unlikely(i+%3E%3D+n))+%7B%09%5C%0A%09abort()%3B%09%5C%0A%7D%0A%0A%0A%0A//+Type+your+code+here,+or+load+an+example.%0Aint+square(int+num)+%7B%0A++++ACCESS_HELPER(num,+20)%3B%0A++++return+num+*+num%3B%0A%7D%0A'),l:'5',n:'1',o:'C%2B%2B+source+%231',t:'0')),k:50,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:clang1810,filters:(b:'0',binary:'1',binaryObject:'1',commentOnly:'0',debugCalls:'1',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1',verboseDemangling:'0'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,libs:!(),options:'-O3',overrides:!(),selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),source:1),l:'5',n:'0',o:'+x86-64+clang+18.1.0+(Editor+%231)',t:'0')),k:50,l:'4',n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4
///
/// The idea behind this is to add a second security check even in realease 
/// mode, which does not add any runtime overhead.
#define ACCESS_HELPER(i, n)				\
if (__builtin_expect(!!(i >= n), 0)) { 	\
	abort();							\
}

/// Translates a given bit length into the minimal
/// data type, which can hold this many bits.
/// NOTE: only unsigned datatypes are used.
///	Usage:
/// 	using T = LogTypeTemplate<16>::type; // holding `uint16_t`
template<uint32_t n>
using LogTypeTemplate =
    typename std::conditional<(n <= 8), uint8_t,
        typename std::conditional<(n <= 16), uint16_t,
            typename std::conditional<(n <= 32), uint32_t,
                typename std::conditional<(n <= 64), uint64_t,
                    __uint128_t>::type>::type>::type>::type;

template<const uint32_t n, const uint32_t m>
using MinLogTypeTemplate =
    typename std::conditional<(n <= m), LogTypeTemplate<m>,
        typename std::conditional<(n <= 8), uint8_t,
            typename std::conditional<(n <= 16), uint16_t,
                typename std::conditional<(n <= 32), uint32_t,
                    typename std::conditional<(n <= 64), uint64_t,
                        __uint128_t>::type>::type>::type>::type>::type;

/// Translates a given number into the minimal datatype which
/// is capable of holding this datatype
template<__uint128_t n>
using TypeTemplate =
    typename std::conditional<(n <= 0xFF), uint8_t,
        typename std::conditional<(n <= 0xFFFF), uint16_t,
            typename std::conditional<(n <= 0xFFFFFFFF), uint32_t,
                typename std::conditional<(n <= 0xFFFFFFFFFFFFFFFF), uint64_t,
                    __uint128_t>::type>::type>::type>::type;


// tracy stuff
#ifdef USE_TRACY
#include <tracy/Tracy.hpp>
#else
#define ZoneScoped
#endif

#endif//SMALLSECRETLWE_HELPER_H
