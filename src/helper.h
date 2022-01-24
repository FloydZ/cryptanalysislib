#ifndef SMALLSECRETLWE_HELPER_H
#define SMALLSECRETLWE_HELPER_H

#ifndef VERSION
#define VERSION "0.0.1"
#endif

// should be passed via gcc/clang command line
// currently activated for syntax highlighting
//#define USE_LOOP_UNROLL
//#define USE_PREFETCH
//#define USE_BRANCH_PREDICTION

// Global Includes
#include <cstdint>      // needed for uint8_t and so on
#include <vector>       // for __level_translation_array
#include <array>
#include <cmath>
#include <concepts>     // for std::integral
#include <type_traits>  // for std::convertable_to

#ifndef __CUDACC__
#define __device__
#define __host__
#endif

#if __AVX2__
#include <immintrin.h>
#include <emmintrin.h>

//
#if CUSTOM_ALIGNMENT
#define MM256_LOAD(x) _mm256_load_ps(x)
#define MM256_STORE(x, y) _mm256_store_ps(x, y)
#else
#define MM256_LOAD(x) _mm256_loadu_ps(x)
#define MM256_STORE(x, y) _mm256_storeu_ps(x, y)
#endif


#define MM256_LOAD_UNALIGNED(x) _mm256_loadu_ps(x)
#define MM256_STORE_UNALIGNED(x, y) _mm256_storeu_ps(x, y)

// Load instruction which do not touch the cache
#define MM256_STREAM_LOAD256(x) _mm256_stream_load_si256(x)
#define MM256_STREAM_LOAD128(x) _mm_stream_load_si128(x)
#define MM256_STREAM_LOAD64(x)  __asm volatile("movntdq %0, (%1)\n\t"       \
												:                           \
												: "x" (x), "r" (y)          \
												: "memory");
// Write instructions which do not touch the cache
#define MM256_STREAM256(x, y) _mm256_stream_load_si256(x, y)
#define MM256_STREAM128(x, y) _mm_stream_si128(x)
#define MM256_STREAM64(x, y)  __asm volatile("movntdq %0, (%1)\n\t"         \
												:                           \
												: "x" (y), "r" (x)          \
												: "memory");
#endif // end __AVX2__

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

#include <thread>
#include <atomic>

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

// variables needed
std::atomic<bool> finished;
std::atomic<uint64_t> outerloops_all;

// Enable performance logging. Aka function return some useful performance information
#ifdef PERFORMANCE_LOGGING
#define PERFORMANE_WRITE(x) x
#else
#define PERFORMANE_WRITE(x)
#endif

#ifdef __unix__
#define MADIVE(a, b, c) //madvise(a, b, c);
#else
#define MADIVE(a, b, c)
#endif

// performance helpers
#if defined(USE_BRANCH_PREDICTION) && (!defined(DEBUG))
#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)
#else
#define likely(x)       x
#define unlikely(x)     x
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
#define cryptanalysislib_prefetch(address)  __builtin_prefetch((const void *)(address), 0, 0)
#define cryptanalysislib_prefetchw(address) __builtin_prefetch((const void *)(address), 1, 0)
constexpr std::ptrdiff_t prefetch_distance = 32*4;     //Prefetch amount in bytes
#else
#define prefetch(m, x, y)
#define cryptanalysislib_prefetch(address)
#define cryptanalysislib_prefetchw(address)
constexpr std::ptrdiff_t prefetch_distance = 0;
#endif

#ifdef USE_LOOP_UNROLL
// Some loop unrollment optimisation
#define LOOP_UNROLL()                      \
    _Pragma(STRINGIFY(clang loop unroll(full)))
#else
#define LOOP_UNROLL()
#endif

#include <iostream>

#ifdef DEBUG
#include <cassert>
#define ASSERT(x) assert(x)
#else
#define ASSERT(x)
#endif

#ifdef DEBUG
#define DEBUG_MACRO(x) x
#else
#define DEBUG_MACRO(x)
#endif

// Mem functions
static __FORCEINLINE__ void* cryptanalysislib_align_up(const void * address, size_t alignment) {
	return (void *)((((intptr_t)address) + ((intptr_t)alignment) - 1) & (-((intptr_t)alignment)));
}

static __FORCEINLINE__ void* cryptanalysislib_aligned_malloc(size_t size, size_t alignment) {
	void * address = malloc(size + sizeof(short) + alignment - 1);
	if (address != NULL) {
		void * aligned_address = cryptanalysislib_align_up((void *)((intptr_t)address + (intptr_t)(sizeof(short))), alignment);
		((short *)aligned_address)[-1] = (short)((intptr_t)aligned_address - (intptr_t)address);

		return aligned_address;
	}

	return NULL;
}

/**
 * Taken from: https://github.com/embeddedartistry/libmemory/blob/master/src/aligned_malloc.c
 * aligned_free works like free(), but we work backwards from the returned
 * pointer to find the correct offset and pointer location to return to free()
 * Note that it is VERY BAD to call free() on an aligned_malloc() pointer.
 */
void cryptanalysislib_aligned_free(void* ptr) {
	ASSERT(ptr);

	/*
	 * Walk backwards from the passed-in pointer to get the pointer offset
	 * We convert to an offset_t pointer and rely on pointer math to get the data
	 */
	uint64_t offset = uint64_t(*((uint64_t *)(uint64_t(ptr) - 1)));

	/*
	 * Once we have the offset, we can get our original pointer and call free
	 */
	void* pt = (void*)((uint8_t*)ptr - offset);
	free(pt);
}

#ifdef FORCE_HPAGE
// normal page, 4KiB, buts its forced to be an huge page
#define PAGE_SIZE (1 << 21)
#else
// normal page, 4KiB
#define PAGE_SIZE (1 << 12)
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
static void check_huge_page(void* ptr) {
	int pagemap_fd = open("/proc/self/pagemap", O_RDONLY);
	if (pagemap_fd < 0) {
		std::cout << "could not open /proc/self/pagemap: " <<  strerror(errno) << "\n";
	}
	int kpageflags_fd = open("/proc/kpageflags", O_RDONLY);
	if (kpageflags_fd < 0) {
		std::cout << "could not open /proc/kpageflags: " << strerror(errno) << "\n";
	}

	// each entry is 8 bytes long
	uint64_t ent;
	if (pread(pagemap_fd, &ent, sizeof(ent), ((uintptr_t) ptr) / PAGE_SIZE * 8) != sizeof(ent)) {
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
void*  cryptanalysislib_hugepage_malloc(const size_t size) {
	const size_t nr_pages = (size+HPAGE_SIZE-1)/HPAGE_SIZE;
	const size_t alloc_size = nr_pages*HPAGE_SIZE;
	void *ret = aligned_alloc(HPAGE_SIZE, alloc_size);
	if (ret == nullptr){
		std::cout << "error alloc\n";
		return nullptr;
	}

	madvise(ret, size, MADV_HUGEPAGE);

	size_t buf = (size_t)ret;
	for (size_t end = buf + size; buf < end; buf += HPAGE_SIZE) {
		// allocate page
		memset((void *)buf, 0, 1);
		// check the page is indeed huge
		check_huge_page((void *)buf);
	}

	return ret;
}
#endif

// simple version of the complex constexpr loop down below.
template<std::size_t N> struct num { static const constexpr auto value = N; };

template <class F, std::size_t... Is>
void constexpr_for(F func, std::index_sequence<Is...>) {
	using expander = int[];
	(void)expander{0, ((void)func(num<Is>{}), 0)...};
}

template <std::size_t N, typename F>
void constexpr_for(F func) {
	constexpr_for(func, std::make_index_sequence<N>());
}

/// jeah complex meta programming. My approach to have something like a constexpr loop.
/// \tparam for_start
/// \tparam for_end
/// \tparam ret
/// \tparam functor
/// \tparam sequence_width
/// \tparam functor_types
template<size_t for_start, size_t for_end, typename ret, typename functor, size_t sequence_width, typename... functor_types>
struct static_for_impl {
	static inline ret loop(functor_types&&... functor_args) {
		// The main sequence point is created, and then we call "next" on each point inside
		using sequence = point<for_start, for_end>;
		return next<sequence>
				(std::integral_constant<bool, sequence::is_end_point_>(),
				 std::forward<functor_types>(functor_args)...);
	}

private:

	// A point is a node of an n-ary tree
	template<size_t pt_start, size_t pt_end> struct point {
		static constexpr size_t start_        { pt_start };
		static constexpr size_t end_          { pt_end };
		static constexpr size_t count_        { end_ - start_ + 1 };
		static constexpr bool is_end_point_   { count_ <= sequence_width };

		static constexpr size_t sequence_count() {
			return points_in_sequence(sequence_width) > sequence_width ? sequence_width : points_in_sequence(sequence_width);
		}

	private:
		// Calculates the start and end indexes for a child node
		static constexpr size_t child_start(size_t index) {
			return index == 0 ? pt_start : child_end(index - 1) + 1;
		}
		static constexpr size_t child_end(size_t index) {
			return index == sequence_count() - 1
					? pt_end : pt_start + points_in_sequence(sequence_count()) * (index + 1) -
						(index < count_
							 ? 1 : 0);
		}
		static constexpr size_t points_in_sequence(size_t max) {
			return count_ / max + (
					(count_ % max) > 0
					? 1 : 0);
		}

	public:
		// Generates child nodes when needed
		template<size_t index> using child_point = point<child_start(index), child_end(index)>;
	};

	// flat_for is used to instantiate a section of our our main static_for::loop
	// A point is used to specify which numbers this instance of flat_for will use
	template<size_t flat_start, size_t flat_end, class flat_functor> struct flat_for {
		// This is the entry point for flat_for
		static inline ret flat_loop(functor_types&&... functor_args) {
			return flat_next(std::integral_constant<size_t, flat_start>(),
			          std::forward<functor_types>(functor_args)...);
		}

	private:
		// Loop termination
		static inline void flat_next
				(std::integral_constant<size_t, flat_end + 1>, functor_types&&...) {}

		// Loop function that calls the function passed to it, as well as recurses
		template<size_t index>
		static inline ret flat_next
				(std::integral_constant<size_t, index>, functor_types&&... functor_args) {
			ret r = flat_functor::template func<index>(std::forward<functor_types>(functor_args)...);
			flat_next(std::integral_constant<size_t, index + 1>(),
			          std::forward<functor_types>(functor_args)...);

			return r;
		}
	};

	// This is what gets called when we run flat_for on a point
	// It will recurse to more finer grained point until the points are no bigger than sequence_width
	template<typename sequence>
	struct flat_sequence {
		template<size_t index>
		static inline ret func(functor_types&&... functor_args) {
			using pt = typename sequence::template child_point<index>;
			return next<pt>
					(std::integral_constant<bool, pt::is_end_point_>(),
					 std::forward<functor_types>(functor_args)...);
		}
	};

	// The true_type function is called when our sequence is small enough to run out
	// and call the main functor that was provided to us
	template<typename sequence>
	static inline ret next
			(std::true_type, functor_types&&... functor_args) {
		ret r = flat_for<sequence::start_, sequence::end_, functor>::
		flat_loop(std::forward<functor_types>(functor_args)...);
		return r;
	}

	// The false_type function is called when our sequence is still too big, and we need to
	// run an internal flat_for loop on the child sequence_points
	template<typename sequence>
	static inline ret next
			(std::false_type, functor_types&&... functor_args) {
		ret r = flat_for<0, sequence::sequence_count() - 1, flat_sequence<sequence>>::
		flat_loop(std::forward<functor_types>(functor_args)...);
		return r;
	}
};

/// Metaprogramming Style of the for-loop:
///     for (size_t i = 0; i < count-1 ; i++) {
///         ret r = functor(functor_types);
///     }
/// \tparam count: number of loops
/// \tparam ret : return type of the function
/// \tparam functor: actual function to execute
/// \tparam sequence_width
/// \tparam functor_types
/// \param functor_args
/// \return
template<size_t count, typename ret, typename functor, size_t sequence_width = 70, typename... functor_types>
inline ret static_for(functor_types&&... functor_args) {
	return static_for_impl<0, count-1, ret, functor, sequence_width, functor_types...>::
	loop(std::forward<functor_types>(functor_args)...);
}

template<size_t start, size_t end, typename functor, size_t sequence_width = 70, typename... functor_types>
inline void static_for(functor_types&&... functor_args) {
	static_for_impl<start, end, void, functor, sequence_width, functor_types...>::
	loop(std::forward<functor_types>(functor_args)...);
}

// The same as the C++ templated, but for loops in C
#define CRYPTANALYSELIB_REPEAT_10(x) x CRYPTANALYSELIB_REPEAT_9(x)
#define CRYPTANALYSELIB_REPEAT_9(x) x CRYPTANALYSELIB_REPEAT_8(x)
#define CRYPTANALYSELIB_REPEAT_8(x) x CRYPTANALYSELIB_REPEAT_7(x)
#define CRYPTANALYSELIB_REPEAT_7(x) x CRYPTANALYSELIB_REPEAT_6(x)
#define CRYPTANALYSELIB_REPEAT_6(x) x CRYPTANALYSELIB_REPEAT_5(x)
#define CRYPTANALYSELIB_REPEAT_5(x) x CRYPTANALYSELIB_REPEAT_4(x)
#define CRYPTANALYSELIB_REPEAT_4(x) x CRYPTANALYSELIB_REPEAT_3(x)
#define CRYPTANALYSELIB_REPEAT_3(x) x CRYPTANALYSELIB_REPEAT_2(x)
#define CRYPTANALYSELIB_REPEAT_2(x) x CRYPTANALYSELIB_REPEAT_1(x)
#define CRYPTANALYSELIB_REPEAT_1(x) x
#define CRYPTANALYSELIB_REPEAT(x, N) CRYPTANALYSELIB_REPEAT_##N (x)

__device__ __host__
constexpr int32_t cceil(float num) {
	return (static_cast<float>(static_cast<int32_t>(num)) == num)
	       ? static_cast<int32_t>(num)
	       : static_cast<int32_t>(num) + ((num > 0) ? 1 : 0);
}

__device__ __host__
constexpr int64_t cceil(double num) {
	return (static_cast<double>(static_cast<int32_t>(num)) == num)
	       ? static_cast<int64_t>(num)
	       : static_cast<int64_t>(num) + ((num > 0) ? 1 : 0);
}

/// Binomial coefficient
/// \param nn n over k
/// \param kk n over k
/// \return nn over kk
__device__ __host__
constexpr inline uint64_t bc(uint64_t nn, uint64_t kk) noexcept {
	return
			(kk > nn  ) ? 0 :       // out of range
			(kk == 0 || kk == nn  ) ? 1 :       // edge
			(kk == 1 || kk == nn - 1) ? nn :       // first
			(kk + kk < nn  ) ?           // recursive:
			(bc(nn - 1, kk - 1) * nn) / kk :       //  path to k=1   is faster
			(bc(nn - 1, kk) * nn) / (nn - kk);      //  path to k=n-1 is faster
}

/// Sumer over all binomial coefficients nn over i, with i <= kk
/// \param nn
/// \param kk
/// \return \sum n over i
constexpr uint64_t sum_bc(uint64_t nn, uint64_t kk) {
	uint64_t sum = 0;
	for (uint64_t i = 1; i <= kk; ++i) {
		sum += bc(nn, i);
	}

	// just make sure that we do not return zero.
	return std::max(sum, uint64_t(1));
}

/// \param n input
/// \return ceil(log2(x)), only useful if you need the number of bits needed
constexpr uint64_t constexpr_bits_log2(uint64_t n) {
	return n <= 1 ? 0 : 1 + constexpr_bits_log2((n + 1) / 2);
}

// taken from https://github.com/kthohr/gcem/blob/master/include/gcem_incl/log.hpp
__device__ __host__
constexpr double log_cf_main(const double xx, const int depth) noexcept { return( depth < 25 ? double(2*depth - 1) - uint64_t(depth*depth)*xx/log_cf_main(xx,depth+1) : double(2*depth - 1) );}
__device__ __host__
constexpr double log_cf_begin(const double x) { return( double(2)*x/log_cf_main(x*x,1) ); }
__device__ __host__
constexpr uint64_t const_log(const uint64_t x) { return uint64_t(log_cf_begin((x - double(1))/(x + double(1))) ); }
//__host__ __device__
//constexpr double const_log(const double x) { return log_cf_begin((x - double(1))/(x + double(1))) ; }

// SOURCE: https://github.com/elbeno/constexpr/blob/master/src/include/cx_math.h
// test whether values are within machine epsilon, used for algorithm
// termination
template <typename T>
constexpr bool feq(T x, T y) {
	return abs(x - y) <= std::numeric_limits<T>::epsilon();
}

template <typename T>
constexpr T constexpr_internal_exp(T x, T sum, T n, int i, T t) {
	return feq(sum, sum + t/n) ?
	       sum : constexpr_internal_exp(x, sum + t/n, n * i, i+1, t * x);
}

template <typename FloatingPoint>
constexpr FloatingPoint constexpr_exp(
		FloatingPoint x,
		typename std::enable_if<std::is_floating_point<FloatingPoint>::value>::type* = nullptr){
	return true ? constexpr_internal_exp(x, FloatingPoint{1}, FloatingPoint{1}, 2, x) :
	       throw 0;
}

template <typename Integral>
constexpr double constexpr_exp(
		Integral x,
		typename std::enable_if<std::is_integral<Integral>::value>::type* = nullptr) {
	return constexpr_internal_exp<double>(x, 1.0, 1.0, 2, x);
}


template <typename T>
constexpr T constexpr_internal_log_iter(T x, T y) {
	return y + T{2} * (x - constexpr_exp(y)) / (x + constexpr_exp(y));
}
template <typename T>
constexpr T constexpr_internal_log(T x, T y) {
	return feq(y, constexpr_internal_log_iter(x, y)) ? y : constexpr_internal_log(x, constexpr_internal_log_iter(x, y));
}

constexpr long double e() {
	return 2.71828182845904523536l;
}
// For numerical stability, constrain the domain to be x > 0.25 && x < 1024
// - multiply/divide as necessary. To achieve the desired recursion depth
// constraint, we need to account for the max double. So we'll divide by
// e^5. If you want to compute a compile-time log of huge or tiny long
// doubles, YMMV.
// if x <= 1, we will multiply by e^5 repeatedly until x > 1
template <typename T>
constexpr T logGT(T x) {
	return x > T{0.25} ? constexpr_internal_log(x, T{0}) :
	       logGT<T>(x * e() * e() * e() * e() * e()) - T{5};
}
// if x >= 2e10, we will divide by e^5 repeatedly until x < 2e10
template <typename T>
constexpr T logLT(T x) {
	return x < T{1024} ? constexpr_internal_log(x, T{0}) :
	       logLT<T>(x / (e() * e() * e() * e() * e())) + T{5};
}

template <typename FloatingPoint>
constexpr FloatingPoint constexpr_log(
		FloatingPoint x,
		typename std::enable_if<std::is_floating_point<FloatingPoint>::value>::type* = nullptr) {
	return x < 0 ? 0 :
	       x >= FloatingPoint{1024} ? logLT(x) : logGT(x);
}

template <typename Integral>
constexpr double constexpr_log(
		Integral x,
		typename std::enable_if<std::is_integral<Integral>::value>::type* = nullptr) {
	return log(static_cast<double>(x));
}

//----------------------------------------------------------------------------
// other logarithms
template <typename FloatingPoint>
constexpr FloatingPoint constexpr_log10(
		FloatingPoint x,
		typename std::enable_if<std::is_floating_point<FloatingPoint>::value>::type* = nullptr) {
	return constexpr_log(x)/constexpr_log(FloatingPoint{10});
}
template <typename Integral>
constexpr double constexpr_log10(
		Integral x,
		typename std::enable_if<std::is_integral<Integral>::value>::type* = nullptr){
	return constexpr_log10(static_cast<double>(x));
}

template <typename FloatingPoint>
constexpr FloatingPoint constexpr_log2(
		FloatingPoint x,
		typename std::enable_if<std::is_floating_point<FloatingPoint>::value>::type* = nullptr){
	return constexpr_log(x)/ constexpr_log(FloatingPoint{2});
}
template <typename Integral>
constexpr double constexpr_log2(
		Integral x,
		typename std::enable_if<std::is_integral<Integral>::value>::type* = nullptr){
	return constexpr_log2(static_cast<double>(x));
}

// pow: compute x^y
// a = x^y = (exp(log(x)))^y = exp(log(x)*y)
template <typename FloatingPoint>
constexpr FloatingPoint constexpr_pow(
		FloatingPoint x, FloatingPoint y,
		typename std::enable_if<std::is_floating_point<FloatingPoint>::value>::type* = nullptr) {
	return true ? constexpr_exp(constexpr_log(x)*y) :
	       throw 0;
}

/// \param x input
/// \return H[x] := -x*Log2[x] - (1 - x)*Log2[1 - x];
double HH(const double x) {
	return -x*log2(x) - (1.-x)*log2(1.-x);
}


/// access function for the global array.
/// \param level
/// \return if level >= 0:
///				return the lower/upper bound as normal.
///			if level == -1:
///				return on all bits
static void translate_level(uint64_t *lower, uint64_t *upper, const uint32_t level,
                         const std::vector<uint64_t> &level_translation_array) {
	ASSERT(lower != NULL && upper != NULL);

	// this is actually mostly only for testing.
	if (unlikely(level == uint32_t(-1))) {
		*lower = level_translation_array[0];
		*upper = level_translation_array[level_translation_array.size()-1];
		return;
	}

	// we __MUST__ check this after the 'if' clause, because otherwise this would catch the -1 test case
	ASSERT(level <= level_translation_array.size()-1 && "wrong level");

	*lower = level_translation_array[level];
	*upper = level_translation_array[level+1];
}

/// translates an array 'level_filter_array' = (e.g.) [4, 0, 0] into a 'norm' = (e.g.) 2, s.t. every 'Value' with a
/// coordinate which is absolute bigger than 'Norm' needs to be filtered out.
/// assumes to count from the top to the bottom of the tree in increasing order. So the root is in lvl 0.
/// \param const lvl = current lvl
/// \param const level_filter_array input parameter
static uint32_t translate_filter(const uint8_t lvl, const uint16_t nr2,
										   const std::vector<std::vector<uint8_t>> &level_filter_array) {
//	const uint32_t max_lvl = level_filter_array.size();
//
//	// some sanity check
//	ASSERT(max_lvl > 0);
//	ASSERT(max_lvl >= lvl);

	if (level_filter_array[lvl][2] > 0)
		// this case is the case we allow twos
		return nr2;
	else
		// we don't allow twos
		return uint32_t(-1);
}

/// print some information about the current instance.
static void ident() {
#if defined(CONFIG_AM_MINIMAL)
	std::cout << "TEST SMALL\n";
#elif defined(CONFIG_AM_TOY)
	std::cout << "TEST TOY\n";
#elif defined(CONFIG_TEMPLATE)
	std::cout << "TEST TEMPLATE\n";
#else
	std::cout << "Non Standard Configuration\n";
#endif
#ifdef G_G_d
	std::cout << "d = " << G_d << "\n";
#endif
#ifdef G_G_n
	std::cout << "n = " << G_n << "\n";
#endif
#ifdef LOG_Q
	std::cout << "LOG(q) = " << LOG_Q << "\n";
#endif
#ifdef G_q
	std::cout << "q = " << G_q << "\n";
#endif
#ifdef G_w
	std::cout << "w = " << G_w << "\n";
#endif

#ifdef CUSTOM_ALIGNMENT
	std::cout << "DEFINED CUSTOM_ALIGNMENT" << "\n";
#endif
#ifdef BINARY_CONTAINER_ALIGNMENT
	std::cout << "DEFINED BINARY_CONTAINER_ALIGNMENT" << "\n";
#endif
#ifdef USE_PREFETCH
	std::cout << "DEFINED USE_PREFETCH" << "\n";
#endif
#ifdef USE_BRANCH_PREDICTION
	std::cout << "DEFINED USE_BRANCH_PREDICTION" << "\n";
#endif
#ifdef VALUE_BINARY
	std::cout << "DEFINED Binary" << "\n";
#elif defined(VALUE_KARY)
	std::cout << "DEFINED kAry" << "\n";
#else
	std::cout << "DEFINED no Binary no kAray ERROR" << "\n";
#endif
}


/// Templates for the win.
///	Usage:
/// using Data
template <uint32_t ll>
using LogTypeTemplate =
typename std::conditional<(ll <= 8), uint8_t,
		typename std::conditional<(ll <= 16), uint16_t,
				typename std::conditional<(ll <= 32), uint32_t,
						typename std::conditional<(ll <= 64), uint64_t,
							__uint128_t
						>::type
				>::type
		>::type
>::type;

template <uint64_t n>
using TypeTemplate =
typename std::conditional<(n <= 0xFF), uint8_t,
		typename std::conditional<(n <= 0xFFFF), uint16_t,
				typename std::conditional<(n <= 0xFFFFFFFF), uint32_t,
						uint64_t
				>::type
		>::type
>::type;

/// IMPORTANT
///		assumes x != 0
///		assumes x is an int or unsigned int
#define Count_Trailing_Zeros(x) __builtin_ctz(x);

/// simple helper function for printing binary data.
/// \tparam T	type to print. Should be an arithmetic type.
/// \param a	value to print
/// \param l1	print a space at this position
/// \param l2	print a space at this position
template<typename T>
#if __cplusplus > 201709L
	requires std::is_arithmetic_v<T>
#endif
void printbinary(T a,
				 const uint16_t l1=std::numeric_limits<uint16_t>::max(),
				 const uint16_t l2=std::numeric_limits<uint16_t>::max()) {
	const T mask = 1;
	for (uint16_t i = 0; i < sizeof(T)*8; ++i) {
		if (a & mask) {
			std::cout << "1";
		} else {
			std::cout << "0";
		}
		a >>= 1;
		if ((i == l1 - 1) || (i == l2 - 1))
			std::cout << " ";
	}
}


#if __cplusplus > 201709L
template<class Value>
concept ExtractorValueAble = requires(Value v) {
	typename Value::ContainerLimbType;
	v.data().data();

	// `ValueType` must be able to return a pointer to the underlying data.
	v.ptr();
};
#endif

/// Helper class for extracting and adding bits in specific windows.
/// \tparam ValueType
/// \tparam ArgumentLimbType
template<typename ValueType, typename ArgumentLimbType>
#if __cplusplus > 201709L
	requires ExtractorValueAble<ValueType> &&
	        std::is_integral<ArgumentLimbType>::value
#endif
class WindowExtractor {
public:
	typedef typename ValueType::ContainerLimbType ContainerLimbType;
	constexpr static uint32_t BITSIZE = sizeof(ContainerLimbType)*8;

	template<uint32_t k_lower, uint32_t k_higher, uint32_t flip=0>
	static inline ArgumentLimbType add(ValueType &v, const ValueType &w) {
		return add<k_lower, k_higher, flip>(v.data().data().data(), w.data().data().data());
	}

	///
	/// \tparam k_lower		lower coordinate to extract
	/// \tparam k_higher 	higher coordinate (nit included) to extract
	/// \tparam flop		if == 0 : nothing happens
	/// 					k_lower <= flip <= k_higher:
	///							exchanges the bits between [k_lower, ..., flip] and [flip, ..., k_upper]
	/// \param v1
	/// \param v3
	/// \return				v1+=v2 on the coordinates between [k_lower] and [k_higher]
	template<uint32_t k_lower, uint32_t k_higher, uint32_t flip=0>
	static inline ArgumentLimbType add(ContainerLimbType *v1, const ContainerLimbType *v3) {
		static_assert(k_lower < k_higher);
		// TODO somehow make sure that k_higher is valid. Maybe ensure that 'ValueType' exports an coordinate field
		constexpr uint32_t llimb = k_lower /BITSIZE;
		constexpr uint32_t hlimb = (k_higher-1)/BITSIZE;
		constexpr uint32_t l     = k_lower %BITSIZE;
		constexpr uint32_t h     = k_higher%BITSIZE;

		constexpr ContainerLimbType mask1 = ~((ContainerLimbType(1u) << l) - 1u);
		constexpr ContainerLimbType mask2 =  h == 0 ? ContainerLimbType (-1) : ((ContainerLimbType(1u) << h) - 1u);

		if constexpr(llimb == hlimb) {
			static_assert(flip == 0, "not implemented");

			v1[llimb] ^= v3[llimb] & mask1;
			return v1[llimb] >> l;
		} else {
			__uint128_t data;
			if constexpr (llimb == hlimb-1) {
				// this is the easy case, where the bits we want to extract are spread
				// over two limbs.
				v1[llimb] ^= v3[llimb] & mask1;
				v1[hlimb] ^= v3[hlimb] & mask2;

				data = v1[llimb];
				data ^= (__uint128_t(v1[hlimb]) << BITSIZE);

			} else if constexpr (llimb == hlimb-2) {
				static_assert(hlimb >= 2);
				// this is the hard case were the bits we want to extract are spread
				// over 3 limbs. This can only happen in a special configuration
				// --hm1_savefull128bit 1, or --hm1_extendtotriple

				v1[llimb]   ^= v3[llimb] & mask1;
				v1[llimb+1] ^= v3[llimb+1];
				v1[hlimb]   ^= v3[hlimb] & mask2;
				data = *(__uint128_t *) (&v1[llimb]);
				data >>= l;
				data ^= (__uint128_t(v1[hlimb]) << (128-l));

				// TODO impl flip
				return data;
			}

			if constexpr (flip == 0) {
				return data >> l;
			} else {
				static_assert(k_lower <= flip);
				static_assert(flip <= k_higher);
				constexpr uint32_t fshift1 = flip - k_lower;
				constexpr uint32_t fshift2 = k_higher - flip;
				constexpr uint32_t f = fshift1;

				// is moment:
				// k_lower          flip                        k_higher
				// [a                b|c                             d]
				// after this transformation:
				// k_higher                     flip            k+lower
				// [c                            d|a                 b]
				constexpr ArgumentLimbType fmask1 = ~((ArgumentLimbType(1) << f) - 1);// high part
				constexpr ArgumentLimbType fmask2 = (ArgumentLimbType(1) << f) - 1;   // low part

				//move     high -> low                      low -> high
				const ArgumentLimbType data2 = (data & fmask1) >> fshift1;
				const ArgumentLimbType data3 = (data & fmask2) << fshift2;
				const ArgumentLimbType data4 = data2 ^ data3;

				return data4;
			}
		}
	}

	/// extracts the bits of v between k_lower, k_higher zero aligned.
	template<uint32_t k_lower, uint32_t k_higher, uint32_t flip=0>
	static __FORCEINLINE__ ArgumentLimbType extract(const ValueType &v) {
		return extract<k_lower, k_higher, flip>(v.ptr());
	}

	/// \tparam k_lower		lower coordinate to extract
	/// \tparam k_higher 	higher coordinate (nit included) to extract
	/// \tparam flop		if == 0 : nothing happens
	/// 					k_lower <= flip <= k_higher:
	///							exchanges the bits between [k_lower, ..., flip] and [flip, ..., k_upper]
	/// \param v1
	/// \param v3
	/// \return				v on the coordinates between [k_lower] and [k_higher]
	template<uint32_t k_lower, uint32_t k_higher, uint32_t flip=0>
	static __FORCEINLINE__ ArgumentLimbType extract(const ContainerLimbType *v) {
		static_assert(k_lower < k_higher);
		constexpr uint32_t llimb = k_lower /BITSIZE;
		constexpr uint32_t hlimb = (k_higher-1)/BITSIZE;
		constexpr uint32_t l = k_lower %BITSIZE;
		constexpr uint32_t h = k_higher%BITSIZE;

		constexpr ContainerLimbType mask1 = ~((ContainerLimbType(1) << l) - 1);
		constexpr ContainerLimbType mask2 = h == uint32_t(0) ? ContainerLimbType(-1) : ((ContainerLimbType(1) << h) - 1);

		if constexpr(llimb == hlimb) {
			constexpr ContainerLimbType mask  = mask1 & mask2;
			return (v[llimb] & mask) >> l;
		} else {
			__uint128_t data;

			if constexpr (llimb == hlimb-1) {
				// simple case
				ContainerLimbType dl = v[llimb] & mask1;
				ContainerLimbType dh = v[hlimb];
				data = dl ^ (__uint128_t(dh) << BITSIZE);
			} else {
				// hard case
				data = *(__uint128_t *) (&v[llimb]);
				data >>= l;
				data ^= (__uint128_t(v[hlimb]) << (128-l));
			}


			if constexpr(flip == 0) {
				return data >> l;
			} else {
				static_assert(k_lower < flip);
				static_assert(flip < k_higher);

				constexpr uint32_t fshift1 = flip-k_lower;
				constexpr uint32_t fshift2 = k_higher-flip;
				constexpr uint32_t f       = fshift1;

				// is moment:
				// k_lower          flip                        k_higher
				// [a                b|c                             d]
				// after this transformation:
				// k_higher                     flip            k+lower
				// [c                            d|a                 b]
				constexpr ArgumentLimbType fmask1 = ~((ArgumentLimbType(1) << f) - 1);  // high part
				constexpr ArgumentLimbType fmask2 =   (ArgumentLimbType(1) << f) - 1;   // low part

				//move     high -> low                      low -> high
				ArgumentLimbType data2 = data >> fshift1;
				ArgumentLimbType data3 = (data & fmask2) << fshift2;
				ArgumentLimbType data4 = data2 ^ data3;

				return data4;
			}
		}
	}

	/// same as above only as the non constexpr version
	/// \param v
	/// \param k_lower
	/// \param k_higher
	/// \return
	static inline ArgumentLimbType extract(const ValueType &v, uint32_t k_lower, uint32_t k_higher) {
		const uint32_t llimb = k_lower /BITSIZE;
		const uint32_t hlimb = (k_higher-1)/BITSIZE;
		const uint32_t l = k_lower%BITSIZE;
		const uint32_t h = k_higher%BITSIZE;
		const ContainerLimbType mask1 = ~((ContainerLimbType(1) << l) - 1);
		const ContainerLimbType mask2 =  ((ContainerLimbType(1) << h) - 1);

		if (llimb == hlimb) {
			const ContainerLimbType mask  = mask1 & mask2;
			return (v.data().data()[llimb] & mask) >> l;
		} else {
			ContainerLimbType dl = v.data().data()[llimb] & mask1;
			ContainerLimbType dh = v.data().data()[hlimb] & mask2;

			__uint128_t data = dl ^ (__uint128_t(dh) << BITSIZE);
			return data >> l;
		}
	}
};
#endif //SMALLSECRETLWE_HELPER_H
