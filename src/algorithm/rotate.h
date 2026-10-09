#pragma once

#include <type_traits>
#include <cstring>

#ifdef USE_AVX2 
#include <immintrin.h>
#endif

#include "copy.h"
#include "memory/memcpy.h"

namespace cryptanalysislib {
namespace internal {
    /// Swaps two contiguous blocks of elements in forward order
    ///
    /// \tparam T type of array elements
    /// \param array[in,out]: array containing the blocks to swap
    /// \param start1[in]: starting index of the first block
    /// \param start2[in]: starting index of the second block
    /// \param block_size[in]: number of elements in each block
    template<typename T>
    constexpr void forward_block_swap(T *array,
                                      const size_t start1, 
                                      const size_t start2,
                                      size_t block_size) noexcept {
    	T *pta, *ptb, swap;
    
    	pta = array + start1;
    	ptb = array + start2;
    
    	while (block_size--) {
    		swap = *pta; *pta++ = *ptb; *ptb++ = swap;
    	}
    }
    
    /// Swaps two contiguous blocks of elements in backward order
    ///
    /// \tparam T type of array elements
    /// \param array[in,out]: array containing the blocks to swap
    /// \param start1[in]: starting index of the first block
    /// \param start2[in]: starting index of the second block
    /// \param block_size[in]: number of elements in each block
    template<typename T>
    void backward_block_swap(T *array,
                             const size_t start1,
                             const size_t start2,
                             size_t block_size) noexcept {
    	T *pta, *ptb, swap;
    
    	pta = array + start1 + block_size;
    	ptb = array + start2 + block_size;
    
    	while (block_size--) {
    		swap = *--pta; *pta = *--ptb; *ptb = swap;
    	}
    }

    /// Rotates elements in an array using a temporary buffer
    ///
    /// \tparam T type of array elements
    /// \tparam MAX_AUX maximum size of the temporary buffer
    /// \param array[in,out]: array to rotate
    /// \param left[in]: number of elements in the left segment
    /// \param right[in]: number of elements in the right segment
    template<typename T, 
             const size_t MAX_AUX=8>
    constexpr void stack_rotation(T *array,
                                  const size_t left,
                                  const size_t right) noexcept {
    	T *pta, *ptb, *ptc, swap[MAX_AUX] __attribute__((aligned(32)));
    
    	pta = array;
    	ptb = array + left;
    	ptc = array + right;
    
    	if (left < right) {
    		cryptanalysislib::memcpy(swap, pta, left);
    		cryptanalysislib::memmove(pta, ptb, right);
    		cryptanalysislib::memcpy(ptc, swap, left);
    	} else {
    		cryptanalysislib::memcpy(swap, ptb, right);
    		cryptanalysislib::memmove(ptc, pta, left);
    		cryptanalysislib::memcpy(pta, swap, right);
    	}
    }
};

/// Performs a left rotation on bits of an integer value
///
/// \tparam T type of integer to rotate, defaults to uint64_t
/// \param x[in]: value to rotate
/// \param k[in]: number of bits to rotate left
/// \return left-rotated value (x <<< k)
template<typename T=uint64_t>
#if __cplusplus > 201709L
	requires std::is_arithmetic_v<T>
#endif
[[nodiscard]] constexpr static inline uint64_t rotl(const T x,
		                                            const uint32_t k) noexcept {
#ifdef USE_AVX2
    if constexpr(sizeof(T) == 8) {
        return _rotl64(x, k);
    }
    if constexpr(sizeof(T) == 4) {
        return _rotl(x, k);
    }

#endif
	using U = std::make_unsigned_t<T>;
	constexpr uint32_t bits = sizeof(T) * 8u;
	const uint32_t s = k % bits;
	if (s == 0) {
		return U(x);
	}
	// NOTE: computed and truncated in the unsigned type, so narrow types are not promoted
	return U(U(U(x) << s) | U(U(x) >> (bits - s)));
}

/// Performs a right rotation on bits of an integer value
///
/// \tparam T type of integer to rotate, defaults to uint64_t
/// \param x[in]: value to rotate
/// \param k[in]: number of bits to rotate right
/// \return right-rotated value (x >>> k)
template<typename T=uint64_t>
#if __cplusplus > 201709L
	requires std::is_arithmetic_v<T>
#endif
[[nodiscard]] constexpr static inline uint64_t rotr(const T x, 
                                                    const uint32_t k) {
#ifdef USE_AVX2
    if constexpr(sizeof(T) == 8) {
        return _rotr64(x, k);
    }
    if constexpr(sizeof(T) == 4) {
        return _rotr(x, k);
    }

#endif
	using U = std::make_unsigned_t<T>;
	constexpr uint32_t bits = sizeof(T) * 8u;
	const uint32_t s = k % bits;
	if (s == 0) {
		return U(x);
	}
	// NOTE: computed and truncated in the unsigned type, so narrow types are not promoted
	return U(U(U(x) >> s) | U(U(x) << (bits - s)));
}


/// Compile-time right rotation of bits of an integer value
///
/// \tparam num_bits number of bits to rotate
/// \tparam T type of integer to rotate
/// \param x[in]: value to rotate
/// \return result of right-rotating the bits of x by num_bits
template <std::size_t num_bits, typename T> 
#if __cplusplus > 201709L
	requires std::is_arithmetic_v<T>
#endif
consteval T rotr(const T x) noexcept {
    return (x >> num_bits) | (x << ((sizeof(T) * 8u) - num_bits));
}

/// Compile-time left rotation of bits of an integer value
///
/// \tparam num_bits number of bits to rotate
/// \tparam T type of integer to rotate
/// \param x[in]: value to rotate
/// \return result of left-rotating the bits of x by num_bits
template <std::size_t num_bits, typename T> 
#if __cplusplus > 201709L
	requires std::is_arithmetic_v<T>
#endif
consteval T rotl(const T x) noexcept {
    return (x << num_bits) | (x >> ((sizeof(T) * 8u) - num_bits));
}

// main source https://github.com/scandum/rotate/
// basically I concentrated on the implementations not using any memory
// TODO tests and benchmarks

/// Conjoined Triple Reversal rotation algorithm by Igor van den Hoven (2021)
///
/// Rotates array elements using an in-place triple reversal approach
/// without requiring additional memory
///
/// \tparam T type of array elements
/// \param array[in,out]: array to rotate
/// \param left[in]: number of elements in the left segment
/// \param right[in]: number of elements in the right segment
template<typename T>
constexpr void contrev_rotation(T *array,
                                const size_t left, 
                                const size_t right) {
	T swap;
	size_t loop;

	T *pta = array;
	T *ptb = array + left;
	T *ptc = array + left;
	T *ptd = array + left + right;

    if (left > right) {
    	loop = right / 2;
    
    	while (loop--) {
    		swap = *--ptb;
    		*ptb = *pta;
    		*pta++ = *ptc;
    		*ptc++ = *--ptd;
    		*ptd = swap;
    	}
    
    	loop = (ptb - pta) / 2;
    
    	while (loop--) {
    		swap = *--ptb;
    		*ptb = *pta;
    		*pta++ = *--ptd;
    		*ptd = swap;
    	}
    	loop = (ptd - pta) / 2;
    
    	while (loop--) {
    		swap = *pta;
    		*pta++ = *--ptd;
    		*ptd = swap;
    	}
    } else if (left < right) {
    	loop = left / 2;
    
    	while (loop--) {
    		swap = *--ptb;
    		*ptb = *pta;
    		*pta++ = *ptc;
    		*ptc++ = *--ptd;
    		*ptd = swap;
    	}
    
    	loop = (ptd - ptc) / 2;
    
    	while (loop--) {
    		swap = *ptc;
    		*ptc++ = *--ptd;
    		*ptd = *pta;
    		*pta++ = swap;
    	}
    	loop = (ptd - pta) / 2;
    
    	while (loop--) {
    		swap = *pta;
    		*pta++ = *--ptd;
    		*ptd = swap;
    	}
    } else {
    	loop = left;
    
    	while (loop--) {
    		swap = *pta;
    		*pta++ = *ptb;
    		*ptb++ = swap;
    	}
    }
}


/// Trinity rotation algorithm that combines memory-efficient buffer with fallback 
/// in-place swapping techniques
///
/// \tparam T type of array elements
/// \tparam MAX_AUX maximum size of the temporary buffer
/// \param array[in,out]: array to rotate
/// \param left[in]: number of elements in the left segment
/// \param right[in]: number of elements in the right segment
template<typename T, const size_t MAX_AUX>
constexpr void trinity_rotation(T *array, 
                                size_t left, 
                                size_t right) {
T *pta, *ptb, *ptc, *ptd, swap[MAX_AUX] __attribute__((aligned(32)));
size_t loop;

    if (left < right) {
    	if (left <= MAX_AUX) {
    		cryptanalysislib::memcpy(swap, array, left);
    		cryptanalysislib::memmove(array, array + left, right);
    		cryptanalysislib::memcpy(array + right, swap, left);
    	} else {
    		pta = array;
    		ptb = pta + left;
    
    		loop = right - left;
    
    		if (loop <= MAX_AUX && loop > 3) {
    			ptc = pta + right;
    			ptd = ptc + left;
    
    			cryptanalysislib::memcpy(swap, ptb, loop);
    
    			while (left--) {
    				*--ptc = *--ptd;
    				*ptd = *--ptb;
    			}
    			cryptanalysislib::memcpy(pta, swap, loop);
    		} else {
    			ptc = ptb;
    			ptd = ptc + right;
    
    			loop = left / 2;
    
    			while (loop--) {
    				*swap = *--ptb;
    				*ptb = *pta;
    				*pta++ = *ptc;
    				*ptc++ = *--ptd;
    				*ptd = *swap;
    			}
    
    			loop = (ptd - ptc) / 2;
    
    			while (loop--) {
    				*swap = *ptc;
    				*ptc++ = *--ptd;
    				*ptd = *pta;
    				*pta++ = *swap;
    			}
    
    			loop = (ptd - pta) / 2;
    
    			while (loop--) {
    				*swap = *pta;
    				*pta++ = *--ptd;
    				*ptd = *swap;
    			}
    		}
    	}
    } else if (right < left) {
    	if (right <= MAX_AUX) {
    		cryptanalysislib::memcpy(swap, array + left, right);
    		cryptanalysislib::memmove(array + right, array, left);
    		cryptanalysislib::memcpy(array, swap, right);
    	} else {
    		pta = array;
    		ptb = pta + left;
    
    		loop = left - right;
    
    		if (loop <= MAX_AUX && loop > 3) {
    			ptc = pta + right;
    			ptd = ptc + left;
    
    			cryptanalysislib::memcpy(swap, ptc, loop);
    
    			while (right--) {
    				*ptc++ = *pta;
    				*pta++ = *ptb++;
    			}
    			cryptanalysislib::memcpy(ptd - loop, swap, loop);
    		} else {
    			ptc = ptb;
    			ptd = ptc + right;
    
    			loop = right / 2;
    
    			while (loop--) {
    				*swap = *--ptb;
    				*ptb = *pta;
    				*pta++ = *ptc;
    				*ptc++ = *--ptd;
    				*ptd = *swap;
    			}
    
    			loop = (ptb - pta) / 2;
    
    			while (loop--) {
    				*swap = *--ptb;
    				*ptb = *pta;
    				*pta++ = *--ptd;
    				*ptd = *swap;
    			}
    
    			loop = (ptd - pta) / 2;
    
    			while (loop--) {
    				*swap = *pta;
    				*pta++ = *--ptd;
    				*ptd = *swap;
    			}
    		}
    	}
    } else {
    	pta = array;
    	ptb = pta + left;
    
    	while (left--) {
    		*swap = *pta;
    		*pta++ = *ptb;
    		*ptb++ = *swap;
    	}
    }
}

/// Gries-Mills rotation algorithm (1981) by David Gries and Harlan Mills
///
/// Efficient in-place rotation algorithm that works by repeatedly swapping
/// equal-sized blocks of elements
///
/// \tparam T[in]: type of array elements
/// \param array[in/out]: array to rotate
/// \param left[in]: number of elements in the left segment
/// \param right[in]: number of elements in the right segment
template<typename T>
constexpr void griesmills_rotation(T *array,
                                   size_t left, 
                                   size_t right) {
	size_t start = 0;

	while (left && right) {
		if (left <= right) {
			do {
				internal::forward_block_swap(array, start, start + left, left);

				start += left;
				right -= left;
			} while (left <= right);
		} else {
			do {
				internal::forward_block_swap(array, start + left - right, start + left, right);

				left -= right;
			} while (right <= left);
		}
	}
}

/// Grail rotation algorithm (2020) by the Holy Grail Sort project
///
/// Enhanced version of Gries-Mills rotation that optimizes edge cases
/// using both forward and backward block swaps, with a fallback to
/// stack-based rotation for small remaining segments
///
/// \tparam T type of array elements
/// \param array[in,out]: array to rotate
/// \param left[in]: number of elements in the left segment
/// \param right[in]: number of elements in the right segment
template<typename T>
constexpr void grail_rotation(T *array,
                              size_t left,
                              size_t right)
{
	size_t min = left <= right ? left : right;
	size_t start = 0;

	while (min > 1) {
		if (left <= right) {
			do {
				internal::forward_block_swap(array, start, start + left, left);

				start += left;
				right -= left;
			} while (left <= right);

			min = right;
		} else {
			do {
				internal::backward_block_swap(array, start + left - right, start + left, right);

				left -= right;
			} while (right <= left);

			min = left;
		}
	}

	if (min) {
		internal::stack_rotation(array + start, left, right);
	}
}

/// Piston rotation algorithm (2021) by Igor van den Hoven
///
/// Based on the successive swap approach described by Gries and Mills (1981)
/// but with optimized block swapping for improved performance
///
/// \tparam T type of array elements
/// \param array[in,out]: array to rotate
/// \param left[in]: number of elements in the left segment
/// \param right[in]: number of elements in the right segment
template<typename T>
constexpr void piston_rotation(T *array,
                               size_t left,
                               size_t right) noexcept {
	size_t start = 0;
	while (left > 0) {
		while (left <= right) {
			internal::forward_block_swap(array, start, start + right, left);
			right -= left;
		}
		if (right <= 0) {
			break;
		}
		do {
			internal::forward_block_swap(array, start, start + left, right);
			left -= right;
			start += right;
		} while (right <= left);
	}
}

/// Helix rotation algorithm (2021) by Control
///
/// Derived from the Grail algorithm but with a different approach to swapping elements
/// that optimizes for certain array patterns
///
/// \tparam T type of array elements
/// \param array[in,out]: array to rotate
/// \param left[in]: number of elements in the left segment
/// \param right[in]: number of elements in the right segment
template<typename T>
constexpr void helix_rotation(T *array,
                    size_t left,
                    size_t right) noexcept {
    T swap;
    size_t start = 0;
    size_t end = left + right;
    size_t mid = left;
    
    while (1) {
    	if (left > right) {
    		if (right <= 1) {
    			break;
    		}
    
    		while (mid > start) {
    			swap = array[--mid];
    			array[mid] = array[--end];
    			array[end] = swap;
    		}
    		mid += (left %= right);
    		right = end - mid;
    	} else {
    		if (left <= 1) {
    			break;
    		}
    
    		while (mid < end) {
    			swap = array[mid];
    			array[mid++] = array[start];
    			array[start++] = swap;
    		}
    		mid -= (right %= left);
    		left = mid - start;
    	}
    }
    
    if (left && right) {
    	internal::stack_rotation(array + start, left, right);
    }
}

/// Drill rotation algorithm (2021) by Igor van den Hoven
///
/// Combines elements from Grail, Piston, and Helix algorithms for a hybrid approach
/// that performs well across various array patterns and sizes
///
/// \tparam T type of array elements
/// \param array[in,out]: array to rotate
/// \param left[in]: number of elements in the left segment
/// \param right[in]: number of elements in the right segment
template<typename T>
constexpr void drill_rotation(T *array,
                              size_t left,
                              size_t right) noexcept {
	T swap;
	size_t start = 0;
	size_t end = left + right;
	size_t mid = left;
	size_t loop;

	while (left > 1) {
		if (left <= right) {
			loop = end - mid - (right %= left);

			do {
				swap = array[mid];
				array[mid++] = array[start];
				array[start++] = swap;
			} while (--loop);
		}

		if (right <= 1) {
			break;
		}

		loop = mid - start - (left %= right);

		do {
			swap = array[--mid];
			array[mid] = array[--end];
			array[end] = swap;
		} while (--loop);
	}

	if (left && right) {
		internal::stack_rotation(array + start, left, right);
	}
}

/// TODO doc
template<class ForwardIt>
constexpr 
ForwardIt rotate(ForwardIt first,
                 ForwardIt middle,
                 ForwardIt last) {
    if (first == middle) {
        return last;
    }
 
    if (middle == last) {
        return first;
    }
 
    ForwardIt write = first;
    ForwardIt next_read = first; // read position for when “read” hits “last”
 
    for (ForwardIt read = middle; read != last; ++write, ++read) {
        if (write == next_read)
            next_read = read; // track where “first” went
        std::iter_swap(write, read);
    }
 
    // rotate the remaining sequence into place
    cryptanalysislib::rotate(write, next_read, last);
    return write;
}

//// TODO doc
template<class ForwardIt,
         class OutputIt>
constexpr
OutputIt rotate_copy(ForwardIt first,
                     ForwardIt middle,
                     ForwardIt last,
                     OutputIt d_first) {
    d_first = std::copy(middle, last, d_first);
    return std::copy(first, middle, d_first);
}

}; // end namespace cryptanalysislib
