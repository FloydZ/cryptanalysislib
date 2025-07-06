#pragma once

#include <cstdint>
#include <type_traits>
#include <cstring>

#ifdef USE_AVX2 
#include <immintrin.h>
#endif


namespace internal {
    /// \param TODO doc everywhere
    template<typename T>
    constexpr void forward_block_swap(T *array,
                                      const size_t start1, 
                                      const size_t start2,
                                      size_t block_size) noexcept {
    	int *pta, *ptb, swap;
    
    	pta = array + start1;
    	ptb = array + start2;
    
    	while (block_size--) {
    		swap = *pta; *pta++ = *ptb; *ptb++ = swap;
    	}
    }
    
    template<typename T>
    void backward_block_swap(T *array,
                             const size_t start1,
                             const size_t start2,
                             size_t block_size) noexcept {
    	int *pta, *ptb, swap;
    
    	pta = array + start1 + block_size;
    	ptb = array + start2 + block_size;
    
    	while (block_size--) {
    		swap = *--pta; *pta = *--ptb; *ptb = swap;
    	}
    }

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
    		memcpy(swap, pta, left * sizeof(int));
    		memmove(pta, ptb, right * sizeof(int));
    		memcpy(ptc, swap, left * sizeof(int));
    	} else {
    		memcpy(swap, ptb, right * sizeof(int));
    		memmove(ptc, pta, left * sizeof(int));
    		memcpy(pta, swap, right * sizeof(int));
    	}
    }
};

/// left rotate
/// \param x value to rotate
/// \param k how much to rotate
/// \return x <<< k
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
	return (x << k) | (x >> ((sizeof(T)*8) - k));
}

/// right rotate
/// \param x value to rotate
/// \param k how much to rotate
/// \return x <<< k
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
    return (x >> k) | (x << ((-k) & ((sizeof(T)*8)-1u)));
}


/// \tparam num_bits The number of bits to rotate.
/// \tparam word_t   The type of number to rotate.
/// \param x The number to be rotated right.
/// \returns The result of right-rotating the bits of x by num_bits.
template <std::size_t num_bits, typename T> 
#if __cplusplus > 201709L
	requires std::is_arithmetic_v<T>
#endif
consteval T rotr(const T x) noexcept {
    return (x >> num_bits) | (x << ((sizeof(T) * 8u) - num_bits));
}

/// \tparam num_bits The number of bits to rotate.
/// \tparam word_t   The type of number to rotate.
/// \param x The number to be rotated left.
/// \returns The result of left-rotating the bits of x by num_bits.
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

// 2021 - Conjoined Triple Reversal rotation by Igor van den Hoven
template<typename T>
constexpr void contrev_rotation(T *array,
                                const size_t left, 
                                const size_t right) {
	T *pta, *ptb, *ptc, *ptd, swap;
	size_t loop;

	pta = array;
	ptb = array + left;
	ptc = array + left;
	ptd = array + left + right;

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



template<typename T, const size_t MAX_AUX>
constexpr void trinity_rotation(T *array, 
                                size_t left, 
                                size_t right) {
T *pta, *ptb, *ptc, *ptd, swap[MAX_AUX] __attribute__((aligned(32)));
size_t loop;

    if (left < right) {
    	if (left <= MAX_AUX) {
    		memcpy(swap, array, left * sizeof(int));
    		memmove(array, array + left, right * sizeof(int));
    		memcpy(array + right, swap, left * sizeof(int));
    	} else {
    		pta = array;
    		ptb = pta + left;
    
    		loop = right - left;
    
    		if (loop <= MAX_AUX && loop > 3) {
    			ptc = pta + right;
    			ptd = ptc + left;
    
    			memcpy(swap, ptb, loop * sizeof(int));
    
    			while (left--) {
    				*--ptc = *--ptd;
    				*ptd = *--ptb;
    			}
    			memcpy(pta, swap, loop * sizeof(int));
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
    		memcpy(swap, array + left, right * sizeof(int));
    		memmove(array + right, array, left * sizeof(int));
    		memcpy(array, swap, right * sizeof(int));
    	} else {
    		pta = array;
    		ptb = pta + left;
    
    		loop = left - right;
    
    		if (loop <= MAX_AUX && loop > 3) {
    			ptc = pta + right;
    			ptd = ptc + left;
    
    			memcpy(swap, ptc, loop * sizeof(int));
    
    			while (right--) {
    				*ptc++ = *pta;
    				*pta++ = *ptb++;
    			}
    			memcpy(ptd - loop, swap, loop * sizeof(int));
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

// 1981 - Gries-Mills rotation by David Gries and Harlan Mills
template<typename T>
constexpr void griesmills_rotation(T *array,
                                   size_t left, 
                                   size_t right) {
	size_t start = 0;

	while (left && right) {
		if (left <= right) {
			do {
				forward_block_swap(array, start, start + left, left);

				start += left;
				right -= left;
			} while (left <= right);
		} else {
			do {
				forward_block_swap(array, start + left - right, start + left, right);

				left -= right;
			} while (right <= left);
		}
	}
}

// 2020 - Grail rotation by the Holy Grail Sort project (Gries-Mills derived)
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

// 2021 - Piston rotation by Igor van den Hoven. Based on the successive swap described by Gries and Mills in 1981.
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

// 2021 - Helix rotation by Control (grail derived)
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

// 2021 - Drill rotation by Igor van den Hoven (grail derived with piston and helix loops)
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
