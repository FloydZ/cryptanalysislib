#ifndef CRYPTANALYSISLIB_ALGORITHM_ARGMAX_H
#define CRYPTANALYSISLIB_ALGORITHM_ARGMAX_H

#include <concepts>
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <limits.h>

#include "simd/simd.h"

namespace cryptanalysislib {
    struct AlgorithmArgMaxConfig {
    public:
        constexpr static size_t aligned_instructions = false;
    };
    constexpr static AlgorithmArgMaxConfig algorithmArgMaxConfig{};


    /// forward declaration
    template<typename T>
	[[nodiscard]] constexpr static inline size_t argmax(const T *a, const size_t n) noexcept;

	/// \tparam S
	/// \tparam config
	/// \param a
	/// \param n
	/// \return
	template<typename S=uint32x8_t,
            const AlgorithmArgMaxConfig &config = algorithmArgMaxConfig>
	[[nodiscard]] constexpr static inline size_t argmax_simd_u32(const uint32_t *a,
	                                               				 const size_t n) noexcept {
		uint32_t max = 0;
		size_t idx = 0;
		auto p = S::set1(max);
        
        constexpr size_t t = S::LIMBS;
		size_t i = 0;
		for (; i+t <= n; i += t) {
			auto y = S::template load<config.aligned_instructions>(a + i); 
            const uint32_t mask = S::lt(p, y);
			if (mask != 0) { [[unlikely]]
				for (uint32_t j = i; j < i + t; j++) {
					if (a[j] > max) {
						max = a[idx = j];
					}
				}

				p = S::set1(max);
			}
		}

		// tail
		for (; i < n; i++) {
			if (a[i] > max) {
				max = a[idx = i];
			}
		}

		return idx;
    }

	/// \tparam S
	/// \tparam config
	/// \param a
	/// \param n
	/// \return
	template<typename S=uint32x8_t,
             const AlgorithmArgMaxConfig &config = algorithmArgMaxConfig>
	[[nodiscard]] constexpr static inline size_t argmax_simd_u32_bl16(const uint32_t *a,
	                                                                  const size_t n) noexcept {
        constexpr size_t t = S::LIMBS;
        constexpr size_t t2 = 2*t;
		uint32_t max = 0;
		auto p = S::set1(max);
		size_t i = 0, idx = 0;
		for (; i+t2 <= n; i += t2) {
            const S y1 = S::template load<config.aligned_instructions>(a + i),
                    y2 = S::template load<config.aligned_instructions>(a + i + t);
			const S y = S::max(y1, y2);
			const uint32_t mask = S::lt(p, y);
			if (mask != 0) { [[unlikely]]
				for (uint32_t j = i; j < i + t2; j++) {
					if (a[j] > max) {
						max = a[idx = j];
					}
				}

				p = S::set1(max);
			}
		}

		// tail
		for (; i < n; i++) {
			if (a[i] > max) {
				max = a[idx = i];
			}
		}

		return idx;	}
 
    template<typename S=uint32x8_t,
             const AlgorithmArgMaxConfig &config = algorithmArgMaxConfig>
	[[nodiscard]] constexpr static inline size_t argmax_simd_u32_bl32(const uint32_t *a,
	                                                    const size_t n) noexcept {
        constexpr size_t t = S::LIMBS;
        constexpr size_t t4 = 2*t;
		uint32_t max = 0;
		auto p = S::set1(max);
		size_t i = 0, idx = 0;
		for (; i+t4 <= n; i += t4) {
            S y1 = S::template load<config.aligned_instructions>(a + i + 0*t),
              y2 = S::template load<config.aligned_instructions>(a + i + 1*t),
              y3 = S::template load<config.aligned_instructions>(a + i + 2*t),
              y4 = S::template load<config.aligned_instructions>(a + i + 3*t);

			y1 = S::max(y1, y2);
			y3 = S::max(y3, y4);
			y1 = S::max(y1, y3);
            const uint32_t mask = S::gt(p, y1);
			if (mask != 0) { [[unlikely]]
				idx = i;
				for (uint32_t j = i; j < i + t4; j++) {
					max = (a[j] > max ? a[j] : max);
				}

				p = S::set1(max);
			}
		}

		size_t idx2 = idx+t4-1;
		for (uint32_t j = idx; j < idx + t4-1; j++) {
			if (a[j] == max) {
				idx2 = j;
			}
		}

		for (; i < n; i++) {
			if (a[i] > max) {
				max = a[idx2 = i];
			}
		}

		return idx2;
	}

    /// generic fallback implementation
	/// \tparam T 
	/// \param a array you want to sort
	/// \param n number of elements in this array
	/// \return position of the max element
	template<typename T>
#if __cplusplus > 201709L
	    requires std::totally_ordered<T>
#endif
	[[nodiscard]] constexpr static inline size_t argmax(const T *a,
	                                                    const size_t n) noexcept {
		size_t k = 0;
		for (size_t i = 0; i < n; i++) {
			if (a[i] > a[k]) [[unlikely]] {
				k = i;
			}
		}

		return k;
	}

    /// helper wrapper
	template<typename T>
#if __cplusplus > 201709L
	    requires std::totally_ordered<T>
#endif
	[[nodiscard]] constexpr static inline size_t argmax(const std::vector<T> &a) noexcept {
        return argmax(a.data(), a.size());
	}

    /// helper wrapper
	template<typename T, const size_t n>
#if __cplusplus > 201709L
	    requires std::totally_ordered<T>
#endif
	[[nodiscard]] constexpr static inline size_t argmax(const std::array<T, n> &a) noexcept {
        return argmax(a.data(), n);
	}
}
#endif
