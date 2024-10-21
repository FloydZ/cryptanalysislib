#ifndef CRYPTANALYSISLIB_PREFIXSUM_H
#define CRYPTANALYSISLIB_PREFIXSUM_H

#include "algorithm/algorithm.h"
#include "simd/simd.h"
#include "math/math.h"

#ifdef USE_AVX2
#include <immintrin.h>
#endif

namespace cryptanalysislib::algorithm {
	struct AlgorithmPrefixsumConfig : public AlgorithmConfig {
		const bool aligned_instructions = false;
		const size_t min_size_per_thread = 1u << 14u;
	};
	constexpr static AlgorithmPrefixsumConfig algorithmPrefixsumConfig;

	namespace internal {

#ifdef USE_AVX2
		static inline void sse_prefixsum_u32(uint32_t *in) noexcept {
			__m128i x = _mm_loadu_si128((__m128i *) in);
			// x = 1, 2, 3, 4
			x = _mm_add_epi32(x, _mm_slli_si128(x, 4));
			// x = 1, 2, 3, 4
			//   + 0, 1, 2, 3
			//   = 1, 3, 5, 7
			x = _mm_add_epi32(x, _mm_slli_si128(x, 8));
			// x = 1, 3, 5, 7
			//   + 0, 0, 1, 3
			//   = 1, 3, 6, 10
			_mm_storeu_si128((__m128i *) in, x);
			// return x;
		}

		static inline void avx_prefix_prefixsum_u32(uint32_t *p) noexcept {
			__m256i x = _mm256_loadu_si256((__m256i *) p);
			x = _mm256_add_epi32(x, _mm256_slli_si256(x, 4));
			x = _mm256_add_epi32(x, _mm256_slli_si256(x, 8));
			_mm256_storeu_si256((__m256i *) p, x);
		}

		static inline __m128i sse_prefixsum_accumulate_u32(uint32_t *p,
								                           const __m128i s) noexcept {
			__m128i d = (__m128i) _mm_broadcast_ss((float *) &p[3]);
			__m128i x = _mm_loadu_si128((__m128i *) p);
			x = _mm_add_epi32(s, x);
			_mm_storeu_si128((__m128i *) p, x);
			return _mm_add_epi32(s, d);
		}

		// This number was set experimental by FLoyd via the benchmark in
		// `bench/algorithm/prefixsum`.
		// The benchmark was conducted on a 12th Gen Intel(R) Core(TM) i5-1240P
		constexpr size_t prefixsum_u32_block_size = 32;

		/// specialized sub routines.
		/// PrefixSum:
		/// 	a[0] = a[0]
		///	a[1] = a[0] + a[1]
		/// \param a
		/// \param s
		/// \return
		static __m128i avx2_local_prefixsum_u32(uint32_t *__restrict__ a,
							                    __m128i s) noexcept {
			for (uint32_t i = 0; i < prefixsum_u32_block_size; i += 8) {
				avx_prefix_prefixsum_u32(&a[i]);
			}

			for (uint32_t i = 0; i < prefixsum_u32_block_size; i += 4) {
				s = sse_prefixsum_accumulate_u32(&a[i], s);
			}

			return s;
		}

		/// \param a array
		/// \param n length of array
		static void prefixsum_u32_avx2(uint32_t *__restrict__ a,
								       const size_t n) noexcept {
			// simple version for small inputs
			if (n < prefixsum_u32_block_size) {
				for (uint32_t i = 1; i < n; i++) {
					a[i] += a[i - 1];
				}
				return;
			}

			__m128i s = _mm_setzero_si128();
			uint32_t i = 0;
			for (; i + prefixsum_u32_block_size <= n; i += prefixsum_u32_block_size) {
				s = avx2_local_prefixsum_u32(a + i, s);
			}

			// tail mngt.
			for (; i < n; i++) {
				a[i] += a[i - 1];
			}
		}

		/// source: https://gist.github.com/boss14420/b959a805b291f031af6f41841e6d2c80
		static void prefixsum_i32_avx2(int32_t *v,
									   const size_t n) noexcept {
			__m256i zero	= _mm256_setzero_si256(),
					sum		= _mm256_setzero_si256(),
					offset  = _mm256_set1_epi32(7),
					offset4 = _mm256_set1_epi32(3);

			auto const pe = v + n;
			auto p = v;
			for (; p+8 <= pe; p += 8) {
				__m256i tmp = _mm256_loadu_si256((__m256i*) p);
				tmp = _mm256_add_epi32(tmp, _mm256_slli_si256(tmp, 4));
				tmp = _mm256_add_epi32(tmp, _mm256_slli_si256(tmp, 8));

				// add 4th element to elements 5,6,7,8
				auto adder = _mm256_blend_epi32(zero, _mm256_permutevar8x32_epi32(tmp, offset4), 0b1111'0000);
				tmp = _mm256_add_epi32(tmp, adder);

				sum = _mm256_add_epi32(tmp, sum);
				_mm256_storeu_si256((__m256i*) p, sum);
				sum = _mm256_permutevar8x32_epi32(sum, offset);
			}

			// tail mngt
			for (; p < pe; p++) {
				*p += *(p - 1);
			}
		}

		///
		/// @param v
		/// @param n
		static void _TwoPhaseAccumulate(int32_t *v,
								        const size_t n) noexcept {
			__m128i s = _mm_loadu_si128((__m128i*) v);
			auto const pe = v + n;
			for (auto p = v+4; p+4 <= pe; p += 4) {
				s = _mm_shuffle_epi32(s, 0b11'11'11'11);
				auto x = _mm_loadu_si128((__m128i*) p);
				s = _mm_add_epi32(s, x);
				_mm_storeu_si128((__m128i*) p, s);
			}
		}

		/// \param v
		/// \param n
		static void prefixsum_i32_avx2_v2(int32_t *v,
										  const size_t n) noexcept {
			auto const pe = v + n;
			auto p = v;
			for (; p+16 <= pe; p += 16) {
				__m256i tmp = _mm256_loadu_si256((__m256i*) p);
				tmp = _mm256_add_epi32(tmp, _mm256_slli_si256(tmp, 4));
				tmp = _mm256_add_epi32(tmp, _mm256_slli_si256(tmp, 8));
				_mm256_storeu_si256((__m256i*) p, tmp);

				__m256i b = _mm256_loadu_si256((__m256i*) (p+8));
				b = _mm256_add_epi32(b, _mm256_slli_si256(b, 4));
				b = _mm256_add_epi32(b, _mm256_slli_si256(b, 8));
				_mm256_storeu_si256((__m256i*) (p+8), b);
			}

			const size_t nn = n&(0xFFFFFFFFFFFFFFF0);
			_TwoPhaseAccumulate(v, nn);

			// tail mngt
			for (; p < pe; p++) {
				*p += *(p - 1);
			}
		}
#endif

#ifdef USE_AVX512F
		static void prefixsum_u8_avx512(uint8_t *v,
									    const size_t n) noexcept {
            constexpr uint32_t limbs = 64; 
            
            __m512i mask = _mm512_set1_epi8(limbs-1);
            

            size_t i = 0;
            __m512i acc = _mm512_setzero_si512();
			for (; (i+limbs) <= n; i+=limbs) {
                const __m512i l = _mm512_loadu_si512(v + i);
                const __m512i f = __prefixsum_u8_avx512(l);
                acc = _mm512_add_epi8(acc, f);
                _mm512_storeu_si512(v + i, acc);
                acc = _mm512_permutexvar_epi8(mask, acc);
            }
			
            // tail mngt
			for (; i < n; i++) {
				v[i] += v[i - 1];
			}
        }
		static void prefixsum_u32_avx512(uint32_t *v,
									     const size_t n) noexcept {
            constexpr uint32_t limbs = 16; 
            __m512i mask = _mm512_set1_epi32(limbs-1);
            

            size_t i = 0;
            __m512i acc = _mm512_setzero_si512();
			for (; (i+limbs) <= n; i+=limbs) {
                const __m512i l = _mm512_loadu_si512(v + i);
                const __m512i f = __prefixsum_u32_avx512(l);
                acc = _mm512_add_epi32(acc, f);
                _mm512_storeu_si512(v + i, acc);
                acc =_mm512_permutevar_epi32(mask, acc);
            }
			
            // tail mngt
			for (; i < n; i++) {
				v[i] += v[i - 1];
			}
        }
#endif

		/// \tparam T
		/// \tparam config
		/// \param v
		/// \param n
		template<typename T,
				 const AlgorithmPrefixsumConfig &config=algorithmPrefixsumConfig>
		static void prefixsum_uXX_simd(T *v,
									   const size_t n) noexcept {
#ifdef USE_AVX512F
			constexpr uint32_t limbs = 64/sizeof(T);
#else
			constexpr uint32_t limbs = 32/sizeof(T);
#endif
            constexpr uint32_t t = floor_log2(limbs) - 1;
			using S = TxN_t<T, limbs>;

			size_t i = 0;
			for (; (i+limbs) <= n; i+=limbs) {
				auto d = S::template load<config.aligned_instructions>(v + i);
                S d2 = d;
                for (uint32_t j = 0; j < t; j++) {
                    d2 = S::sll(d2, j*t); 
                    d = d + d2;
                }

				// TODO unfinished
                S::template store<config.aligned_instructions>(v + i, d);
			}

			// tailmngt
			for (; i < n; i++) {
				v[i] += v[i - 1];
			}
		}
	} // end namespace internal
} // end namespace cryptanalysislib::algorithm


namespace cryptanalysislib::algorithm {

	/// inplace prefix sum algorithm
	/// \tparam T
	/// \param data
	/// \param len
	/// \return
	template<typename T,
			 const AlgorithmPrefixsumConfig &config=algorithmPrefixsumConfig>
#if __cplusplus > 201709L
		requires std::is_arithmetic_v<T>
#endif
	constexpr void prefixsum(T *data,
	                      	 const size_t len) {
#ifdef USE_AVX2
		if constexpr (std::is_same_v<T, uint32_t>) {
			internal::prefixsum_u32_avx2(data, len);
			return;
		}
#endif

		for (size_t i = 1; i < len; ++i) {
			data[i] += data[i-1];
		}
	}

	/// \tparam ForwardIt
	/// \param first
	/// \param last
	/// \return
	template<typename ForwardIt,
			 const AlgorithmPrefixsumConfig &config=algorithmPrefixsumConfig>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt>
#endif
	void prefixsum(ForwardIt first,
	               ForwardIt last) noexcept {
		using T = ForwardIt::value_type;
		static_assert(std::is_arithmetic_v<T>);
		const auto count = std::distance(first, last);
		prefixsum<T, config>(&(*first), count);
	}

	/// \tparam InputIt
	/// \tparam OutputIt
	/// \tparam BinaryOp
	/// \param first
	/// \param last
	/// \param d_first
	/// \param op
	/// \return
	template<class InputIt,
			 class OutputIt,
			 class BinaryOp,
			 const AlgorithmPrefixsumConfig &config=algorithmPrefixsumConfig>
#if __cplusplus > 201709L
	    requires std::forward_iterator<InputIt> &&
	    		 std::forward_iterator<OutputIt> &&
    		     std::regular_invocable<BinaryOp,
										const typename InputIt::value_type&,
										const typename InputIt::value_type&>
#endif
	constexpr OutputIt prefixsum(InputIt first,
								  InputIt last,
								  OutputIt d_first,
								  BinaryOp op) noexcept {
		if (first == last) {
			return d_first;
		}

		typename std::iterator_traits<InputIt>::value_type acc = *first;
		*d_first = acc;

		while (++first != last) {
			acc = op(std::move(acc), *first);
			*d_first = acc;
			d_first += 1;
		}

		return ++d_first;
	}

	/// \tparam InputIt
	/// \tparam OutputIt
	/// \tparam BinaryOp
	/// \param first
	/// \param last
	/// \param d_first
	/// \param init
	/// \param op
	/// \return
	template<class InputIt,
			 class OutputIt,
			 class BinaryOp,
			 const AlgorithmPrefixsumConfig &config=algorithmPrefixsumConfig>
#if __cplusplus > 201709L
	    requires std::forward_iterator<InputIt> &&
	    		 std::forward_iterator<OutputIt> &&
    		     std::regular_invocable<BinaryOp,
										const typename InputIt::value_type&,
										const typename InputIt::value_type&>
#endif
	constexpr OutputIt prefixsum(InputIt first,
								 InputIt last,
								 OutputIt d_first,
								 const typename InputIt::value_type init,
								 BinaryOp op) noexcept {
		if (first == last) {
			return d_first;
		}

		typename std::iterator_traits<InputIt>::value_type acc = *first;
		acc = op(std::move(acc), init);
		*d_first = acc;

		while (++first != last) {
			acc = op(std::move(acc), *first);
			*++d_first = acc;
		}

		return ++d_first;
	}

	/// NOTE: ignores any multithreading
	/// \tparam ExecPolicy
	/// \tparam InputIt
	/// \tparam OutputIt
	/// \tparam BinaryOp
	/// \tparam config
	/// \param policy
	/// \param first1
	/// \param last1
	/// \param d_first
	/// \param init
	/// \param op
	/// \return
	template<class ExecPolicy,
			 class InputIt,
			 class OutputIt,
			 class BinaryOp,
			  const AlgorithmPrefixsumConfig &config=algorithmPrefixsumConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<InputIt> &&
    		 std::random_access_iterator<OutputIt> &&
    		 std::regular_invocable<BinaryOp,
									const typename InputIt::value_type&,
									const typename InputIt::value_type&>
#endif
	 OutputIt prefixsum(ExecPolicy&& policy,
						InputIt first1,
			    		InputIt last1,
						OutputIt d_first,
						const typename InputIt::value_type init,
						BinaryOp op) noexcept {
		(void) policy;
		return prefixsum
			<InputIt, OutputIt, BinaryOp, config>
			(first1, last1, d_first, init, op);
	}

};
#endif//CRYPTANALYSISLIB_PREFIXSUM_H
