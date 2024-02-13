#ifndef CRYPTANALYSISLIB_HELPER_H
#define CRYPTANALYSISLIB_HELPER_H

// should be passed via gcc/clang command line
// currently activated for syntax highlighting
//#define USE_LOOP_UNROLL
//#define USE_PREFETCH
//#define USE_BRANCH_PREDICTION

// Global Includes
#include <stddef.h>
#include <string.h>
#include <cstdint>      // needed for uint8_t and so on
#include <vector>       // for __level_translation_array
#include <array>
#include <cmath>
#include <type_traits>  // for std::convertable_to
#include <cassert>
#include <inttypes.h>

#include "cpucycles.h"

#ifndef __CUDACC__
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
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
#ifndef DEBUG_MACRO
#define DEBUG_MACRO(x) x
#endif
#else
#define DEBUG_MACRO(x)
#endif


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


/// Usage:
///		// simple version of the complex constexpr loop down below.
///		template<std::size_t N> struct num { static const constexpr auto value = N; };
///
///		template <class F, std::size_t... Is>
///		void constexpr_for(F func, std::index_sequence<Is...>) {
///			using expander = int[];
///			(void)expander{0, ((void)func(num<Is>{}), 0)...};
///		}
///
///		template <std::size_t N, typename F>
///		void constexpr_for(F func) {
///			constexpr_for(func, std::make_index_sequence<N>());
///		}
template <auto Start, auto End, auto Inc, class F>
constexpr void constexpr_for(F&& f) {
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
static void translate_level(uint64_t *lower,
                            uint64_t *upper, const uint32_t level,
                         const std::vector<uint64_t> &level_translation_array) noexcept {
	ASSERT(lower != NULL && upper != NULL);

	// this is actually mostly only for testing.
	if (unlikely(level == uint32_t(-1))) {
		*lower = level_translation_array[0];
		*upper = level_translation_array[level_translation_array.size()-1];
		return;
	}

	// we __MUST__ check this after the 'if' clause,
	// because otherwise this would catch the -1 test case
	ASSERT(level <= level_translation_array.size()-1u);

	*lower = level_translation_array[level];
	*upper = level_translation_array[level+1u];
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



/// Translates a given bit length into the minimal
/// data type, which can hold this many bits.
/// NOTE: only unsigned datatypes are used.
///	Usage:
/// 	using T = LogTypeTemplate<16>::type; // holding `uint16_t`
template <uint32_t n>
using LogTypeTemplate =
typename std::conditional<(n <= 8), uint8_t,
	typename std::conditional<(n <= 16), uint16_t,
		typename std::conditional<(n <= 32), uint32_t,
			typename std::conditional<(n <= 64), uint64_t,
				__uint128_t
			>::type
		>::type
	>::type
>::type;

/// Translates a given number into the minimal datatype which
/// is capable of holding this datatype
template <uint64_t n>
using TypeTemplate =
typename std::conditional<(n <= 0xFF), uint8_t,
	typename std::conditional<(n <= 0xFFFF), uint16_t,
		typename std::conditional<(n <= 0xFFFFFFFF), uint32_t,
			typename std::conditional<(n <= 0xFFFFFFFFFFFFFFFF), uint64_t,
				__uint128_t
            >::type
		>::type
	>::type
>::type;




/// TODO when rewriting all the decoding algorithms:
/// 	rewrite also this
///		remove the flip
///		simplify code
#if __cplusplus > 201709L
/// this concept enforces the needed functions/typedefs/variables
/// needed by the `Extractor` for the `Value`
/// \tparam Value
template<class Value>
concept ExtractorValueAble = requires(Value v) {
	typename Value::ContainerLimbType;
	v.data().data();

	// `ValueType` must be able to return a pointer to the underlying data.
	v.ptr();
};

/// this concept enforces the needed functions/typedefs the Extractor class needs to implement.
/// \tparam Extractor
template<class Extractor>
concept ExtractorAble = requires(Extractor e) {
	typename Extractor::Value;

	requires requires(typename Extractor::Value &l, const uint32_t i) {
		e.extracts(l, i, i);
	};
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
	typedef ValueType Value;
	typedef typename ValueType::ContainerLimbType ContainerLimbType;
	constexpr static uint32_t BITSIZE = sizeof(ContainerLimbType)*8;

	template<uint32_t k_lower, uint32_t k_higher, uint32_t flip=0>
	static inline ArgumentLimbType add(ValueType &v, const ValueType &w) noexcept {
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
	static inline ArgumentLimbType add(ContainerLimbType *v1, const ContainerLimbType *v3) noexcept {
		static_assert(k_lower < k_higher);
		// TODO somehow make sure that k_higher is valid. Maybe ensure that 'ValueType' exports an coordinate field
		constexpr uint32_t llimb = k_lower /BITSIZE;
		constexpr uint32_t hlimb = (k_higher-1)/BITSIZE;
		constexpr uint32_t l     = k_lower %BITSIZE;
		constexpr uint32_t h     = k_higher%BITSIZE;

		constexpr ContainerLimbType mask1 = ~((ContainerLimbType(1u) << l) - 1u);
		constexpr ContainerLimbType mask2 =  h == 0 ? ContainerLimbType (-1) : ((ContainerLimbType(1u) << h) - 1u);

		if constexpr(llimb == hlimb) {
			//static_assert(flip == 0, "not implemented");
			if constexpr (flip == 0) {
				v1[llimb] ^= v3[llimb] & mask1;
				return v1[llimb] >> l;
			} else {
				// TODO not finished
				assert(0);
				constexpr uint32_t l2 = flip%BITSIZE;
				constexpr ContainerLimbType mask = (ContainerLimbType(1u) << l2) - 1u;

				v1[llimb] ^= v3[llimb] & mask1;
				ContainerLimbType ret  =  v1[llimb];
				ContainerLimbType ret2 =((ret >> (l2)) ^ (ret << l)) & mask1;
				return ret2 >> l;
			}
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
	static __FORCEINLINE__ ArgumentLimbType extract(const ValueType &v) noexcept {
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
	static __FORCEINLINE__ ArgumentLimbType extract(const ContainerLimbType *v) noexcept {
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
	static inline ArgumentLimbType extract(const ValueType &v,
	                                       const uint32_t k_lower,
	                                       const uint32_t k_higher) noexcept {
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
