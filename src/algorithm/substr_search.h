#pragma once 
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <cstring>

#ifdef USE_AVX2
#include <immintrin.h>
#endif

/// A simple structure representing a span of bytes
struct span_t {
    uint8_t *data = nullptr;  ///< Pointer to the data
    size_t len = 0;           ///< Length of the data in bytes

    /// Returns a new span starting at the specified offset from this span
    ///
    /// \param offset[in]: Number of bytes to skip from the beginning
    /// \return A new span starting at data+offset with adjusted length
    constexpr inline span_t after_n(const size_t offset) const noexcept {
        return (offset < len) ? span_t {data + offset, len - offset} : span_t {};
    }
};

namespace cryptanalysislib::algorithm {
    namespace internal {
        static constexpr size_t not_found_k = std::numeric_limits<size_t>::max();

        /// Compares two arrays for equality
        ///
        /// \tparam T type of array elements to compare
        /// \param a[in]: Pointer to the first array
        /// \param b[in]: Pointer to the second array
        /// \param len[in]: Number of elements to compare
        /// \return true if arrays are equal, false otherwise
        template <typename T>
        constexpr inline bool are_equal(T const *a,
                                        T const *b,
                                        const size_t len) noexcept {
            T const *const a_end = a + len;
            for (; a != a_end && *a == *b; a++, b++)
                ;
            return a_end == a;
        }

        /// A naive substring matching algorithm with O(|haystack|*|needle|) comparisons
        /// Matching performance fluctuates between 200 MB/s and 2 GB/s
        ///
        /// \param haystack[in,out]: The string to search in
        /// \param needle[in]: The substring to search for
        /// \return Position of the first match or not_found_k if not found
        constexpr size_t naive_substr(span_t &haystack,
                                      const span_t &needle) {
            if (haystack.len < needle.len) {
                return not_found_k;
            }

            for (size_t off = 0; off <= haystack.len - needle.len; off++) {
                if (are_equal(haystack.data + off, needle.data, needle.len)) {
                    return off;
                }
            }

            return not_found_k;
        }
    
        /// Modified version inspired by Rabin-Karp algorithm
        /// Matching performance fluctuates between 1 GB/s and 3.5 GB/s
        /// 
        /// Similar to Rabin-Karp Algorithm, instead of comparing variable length
        /// strings - we can compare some fixed size fingerprints, which can make
        /// the number of nested loops smaller. But preprocessing text to generate
        /// hashes is very expensive.
        /// Instead - we compare the first 4 bytes of the `needle` to every 4 byte
        /// substring in the `haystack`. If those match - compare the rest.
        ///
        /// \param haystack[in,out]: The string to search in
        /// \param needle[in]: The substring to search for
        /// \return Position of the first match or not_found_k if not found
        constexpr size_t prefix_substr(span_t &haystack,
                                       const span_t &needle) noexcept {
            if (needle.len < 5) {
                return naive_substr(haystack, needle);
            }

            // Precomputed constants.
            uint8_t const *h_ptr = haystack.data;
            uint8_t const *const h_end = haystack.data + haystack.len - needle.len;
            size_t const n_suffix_len = needle.len - 4;
            uint32_t const n_prefix = *reinterpret_cast<uint32_t const *>(needle.data);
            uint8_t const *n_suffix_ptr = needle.data + 4;

            for (; h_ptr <= h_end; h_ptr++) {
                if (n_prefix == *reinterpret_cast<uint32_t const *>(h_ptr)) {
                    if (are_equal(h_ptr + 4, n_suffix_ptr, n_suffix_len)) {
                        return h_ptr - haystack.data;
                    }
                }
            }

            return not_found_k;
        }

#ifdef USE_AVX2
        /// A SIMD vectorized version for AVX2 instruction set
        /// Matching performance is ~ 9 GB/s
        /// 
        /// This version processes 32 `haystack` substrings per iteration,
        /// so the number of instructions is only:
        ///  + 4 loads
        ///  + 4 comparisons
        ///  + 3 bitwise ORs
        ///  + 1 masking
        /// for every 32 consecutive substrings.
        ///
        /// \param haystack[in,out]: The string to search in
        /// \param needle[in]: The substring to search for
        /// \return Position of the first match or not_found_k if not found
        constexpr size_t avx2_prefix_substr(span_t &haystack, 
                                            const span_t &needle) noexcept {

            if (needle.len < 5) {
                return naive_substr(haystack, needle);
            }

            uint8_t const *const h_end = haystack.data + haystack.len - needle.len;
            __m256i const n_prefix = _mm256_set1_epi32(*(uint32_t const *)(needle.data));

            uint8_t const *h_ptr = haystack.data;
            for (; (h_ptr + 32) <= h_end; h_ptr += 32) {
                __m256i h0 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr)), n_prefix);
                __m256i h1 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr + 1)), n_prefix);
                __m256i h2 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr + 2)), n_prefix);
                __m256i h3 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr + 3)), n_prefix);
                __m256i h_any = _mm256_or_si256(_mm256_or_si256(h0, h1), _mm256_or_si256(h2, h3));
                int mask = _mm256_movemask_epi8(h_any);

                if (mask) {
                    for (size_t i = 0; i < 32; i++) {
                        if (are_equal(h_ptr + i, needle.data, needle.len))
                            return i + (h_ptr - haystack.data);
                    }
                }
            }

            // Don't forget the last (up to 35) characters.
            span_t t = haystack.after_n(h_ptr - haystack.data);
            size_t last_match = prefix_substr(t, needle);
            return (last_match != not_found_k) ? last_match + (h_ptr - haystack.data) : not_found_k;
        }

        /// Speculative SIMD version for AVX2 instruction set
        /// Matching performance is ~ 12 GB/s
        ///
        /// Up to 40% of performance in modern CPUs comes from speculative
        /// out-of-order execution. The `prefixed_avx2_t` version has
        /// 4 explicit local memory barriers: 3 ORs and 1 IF branch.
        /// This has only 1 IF branch in the main loop.
        ///
        /// \param haystack[in,out]: The string to search in
        /// \param needle[in]: The substring to search for
        /// \return Position of the first match or not_found_k if not found
        constexpr size_t avx2_speculative_substr(span_t &haystack, 
                                                 const span_t &needle) noexcept {
            if (needle.len < 5) {
                return naive_substr(haystack, needle);
            }

            // Precomputed constants.
            uint8_t const *const h_end = haystack.data + haystack.len - needle.len;
            __m256i const n_prefix = _mm256_set1_epi32(*(uint32_t const *)(needle.data));

            // Top level for-loop changes dramatically.
            // In sequentail computing model for 32 offsets we would do:
            //  + 32 comparions.
            //  + 32 branches.
            // In vectorized computations models:
            //  + 4 vectorized comparisons.
            //  + 4 movemasks.
            //  + 3 bitwise ANDs.
            //  + 1 heavy (but very unlikely) branch.
            uint8_t const *h_ptr = haystack.data;
            for (; (h_ptr + 32) <= h_end; h_ptr += 32) {

                __m256i h0_prefixes = _mm256_loadu_si256((__m256i const *)(h_ptr));
                int masks0 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(h0_prefixes, n_prefix));
                __m256i h1_prefixes = _mm256_loadu_si256((__m256i const *)(h_ptr + 1));
                int masks1 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(h1_prefixes, n_prefix));
                __m256i h2_prefixes = _mm256_loadu_si256((__m256i const *)(h_ptr + 2));
                int masks2 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(h2_prefixes, n_prefix));
                __m256i h3_prefixes = _mm256_loadu_si256((__m256i const *)(h_ptr + 3));
                int masks3 = _mm256_movemask_epi8(_mm256_cmpeq_epi32(h3_prefixes, n_prefix));

                if (masks0 | masks1 | masks2 | masks3) {
                    for (size_t i = 0; i < 32; i++) {
                        if (are_equal(h_ptr + i, needle.data, needle.len))
                            return i + (h_ptr - haystack.data);
                    }
                }
            }

            // Don't forget the last (up to 35) characters.
            span_t t = haystack.after_n(h_ptr - haystack.data);
            size_t last_match = prefix_substr(t, needle);
            return (last_match != not_found_k) ? last_match + (h_ptr - haystack.data) : not_found_k;
        }

        /// A hybrid of `avx_prefixed` and `avx2_speculative_substr`
        /// Matching performance is superior to both individual approaches
        ///
        /// Demonstrates the current inability of scheduler to optimize
        /// the execution flow better than manual optimization. Processes
        /// 64 bytes at once for improved throughput.
        ///
        /// \param haystack[in,out]: The string to search in
        /// \param needle[in]: The substring to search for
        /// \return Position of the first match or not_found_k if not found
        constexpr size_t avx2_hybrid_substr(span_t &haystack, 
                                            const span_t &needle) noexcept {
           if (needle.len < 5)
                return naive_substr(haystack, needle);

            uint8_t const *const h_end = haystack.data + haystack.len - needle.len;
            __m256i const n_prefix = _mm256_set1_epi32(*(uint32_t const *)(needle.data));

            uint8_t const *h_ptr = haystack.data;
            for (; (h_ptr + 64) <= h_end; h_ptr += 64) {

                __m256i h0 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr)), n_prefix);
                __m256i h1 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr + 1)), n_prefix);
                __m256i h2 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr + 2)), n_prefix);
                __m256i h3 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr + 3)), n_prefix);
                int mask03 = _mm256_movemask_epi8(_mm256_or_si256(_mm256_or_si256(h0, h1), _mm256_or_si256(h2, h3)));

                __m256i h4 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr + 32)), n_prefix);
                __m256i h5 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr + 33)), n_prefix);
                __m256i h6 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr + 34)), n_prefix);
                __m256i h7 = _mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i const *)(h_ptr + 35)), n_prefix);
                int mask47 = _mm256_movemask_epi8(_mm256_or_si256(_mm256_or_si256(h4, h5), _mm256_or_si256(h6, h7)));

                if (mask03 | mask47) {
                    for (size_t i = 0; i < 64; i++) {
                        if (are_equal(h_ptr + i, needle.data, needle.len))
                            return i + (h_ptr - haystack.data);
                    }
                }
            }

            // Don't forget the last (up to 67) characters.
            span_t t = haystack.after_n(h_ptr - haystack.data);
            size_t last_match = prefix_substr(t, needle);
            return (last_match != not_found_k) ? last_match + (h_ptr - haystack.data) : not_found_k;
        }
        

        /// An AVX2 optimized substring search that works for any size needles
        /// Performance varies based on needle size, generally 6-10 GB/s
        ///
        /// Uses a two-stage approach - first checking first and last characters
        /// simultaneously with SIMD, then verifying matches with full comparison.
        /// This approach reduces the number of full comparisons needed.
        ///
        /// \param haystack[in,out]: The string to search in
        /// \param needle[in]: The substring to search for
        /// \return Position of the first match or not_found_k if not found
        constexpr size_t avx2_strstr_anysize(span_t &haystack, 
                                                const span_t &needle) noexcept {
            const size_t n = haystack.len; 
            const size_t k = needle.len; 
            const uint8_t *s = haystack.data;

            const __m256i first = _mm256_set1_epi8(needle.data[0]);
            const __m256i last  = _mm256_set1_epi8(needle.data[k - 1]);
        
            for (size_t i = 0; i < n; i += 32) {
        
                const __m256i block_first = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i));
                const __m256i block_last  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + k - 1));
        
                const __m256i eq_first = _mm256_cmpeq_epi8(first, block_first);
                const __m256i eq_last  = _mm256_cmpeq_epi8(last, block_last);
        
                uint32_t mask = _mm256_movemask_epi8(_mm256_and_si256(eq_first, eq_last));
        
                while (mask != 0) {
        
                    const auto bitpos = __builtin_ctz(mask);
        
                    if (memcmp(s + i + bitpos + 1, needle.data + 1, k - 2) == 0) {
                        return i + bitpos;
                    }
        
                    mask = mask & (mask - 1u);
                }
            }
        
            return not_found_k;
        }
#endif

#ifdef USE_AVX512F 
        /// Speculative SIMD version for AVX512 instruction set
        /// Matching performance is ~ 15-18 GB/s on supported hardware
        ///
        /// Similar to the AVX2 speculative version but leverages AVX512's
        /// wider vectors and mask operations for improved throughput.
        /// Processes 64 bytes per iteration with optimized branching.
        ///
        /// \param haystack[in,out]: The string to search in
        /// \param needle[in]: The substring to search for
        /// \return Position of the first match or not_found_k if not found
        constexpr size_t avx512_speculative_substr(span_t &haystack, 
                                            const span_t &needle) noexcept {

            if (needle.len < 5)
                return naive_substr(haystack, needle);

            // Precomputed constants.
            uint8_t const *const h_end = haystack.data + haystack.len - needle.len;
            __m512i const n_prefix = _mm512_set1_epi32(*(uint32_t const *)(needle.data));

            uint8_t const *h_ptr = haystack.data;
            for (; (h_ptr + 64) <= h_end; h_ptr += 64) {

                __m512i h0_prefixes = _mm512_loadu_si512((__m512i const *)(h_ptr));
                int masks0 = _mm512_cmpeq_epi32_mask(h0_prefixes, n_prefix);
                __m512i h1_prefixes = _mm512_loadu_si512((__m512i const *)(h_ptr + 1));
                int masks1 = _mm512_cmpeq_epi32_mask(h1_prefixes, n_prefix);
                __m512i h2_prefixes = _mm512_loadu_si512((__m512i const *)(h_ptr + 2));
                int masks2 = _mm512_cmpeq_epi32_mask(h2_prefixes, n_prefix);
                __m512i h3_prefixes = _mm512_loadu_si512((__m512i const *)(h_ptr + 3));
                int masks3 = _mm512_cmpeq_epi32_mask(h3_prefixes, n_prefix);

                if (masks0 | masks1 | masks2 | masks3) {
                    for (size_t i = 0; i < 64; i++) {
                        if (are_equal(h_ptr + i, needle.data, needle.len))
                            return i + (h_ptr - haystack.data);
                    }
                }
            }

            // Don't forget the last (up to 64+3=67) characters.
            size_t last_match = prefix_substr(haystack.after_n(h_ptr - haystack.data), needle);
            return (last_match != not_found_k) ? last_match + (h_ptr - haystack.data) : not_found_k;
        }
#endif

#ifdef USE_NEON

        /// Speculative SIMD version for ARM NEON instruction set
        /// Matching performance is ~ 5-8 GB/s on ARM processors
        ///
        /// ARM NEON equivalent of the speculative approach used in AVX2/AVX512.
        /// Optimized for ARM architecture with 128-bit vector operations.
        /// Processes 16 bytes per iteration with similar branch optimization.
        ///
        /// \param haystack[in,out]: The string to search in
        /// \param needle[in]: The substring to search for
        /// \return Position of the first match or not_found_k if not found
        constexpr size_t neon_speculative_substr(span_t &haystack, 
                                            const span_t &needle) noexcept {
            if (needle.len < 5)
                return naive_substr(haystack, needle);

            // Precomputed constants.
            uint8_t const *const h_end = haystack.data + haystack.len - needle.len;
            uint32x4_t const n_prefix = vld1q_dup_u32((uint32_t const *)(needle.data));

            uint8_t const *h_ptr = haystack.data;
            for (; (h_ptr + 16) <= h_end; h_ptr += 16) {

                uint32x4_t masks0 = vceqq_u32(vld1q_u32((uint32_t const *)(h_ptr)), n_prefix);
                uint32x4_t masks1 = vceqq_u32(vld1q_u32((uint32_t const *)(h_ptr + 1)), n_prefix);
                uint32x4_t masks2 = vceqq_u32(vld1q_u32((uint32_t const *)(h_ptr + 2)), n_prefix);
                uint32x4_t masks3 = vceqq_u32(vld1q_u32((uint32_t const *)(h_ptr + 3)), n_prefix);

                // Extracting matches from masks:
                // vmaxvq_u32 (only a64)
                // vgetq_lane_u32 (all)
                // vorrq_u32 (all)
                uint32x4_t masks = vorrq_u32(vorrq_u32(masks0, masks1), vorrq_u32(masks2, masks3));
                uint64x2_t masks64x2 = vreinterpretq_u64_u32(masks);
                bool has_match = vgetq_lane_u64(masks64x2, 0) | vgetq_lane_u64(masks64x2, 1);

                if (has_match) {
                    for (size_t i = 0; i < 16; i++) {
                        if (are_equal(h_ptr + i, needle.data, needle.len))
                            return i + (h_ptr - haystack.data);
                    }
                }
            }

            // Don't forget the last (up to 16+3=19) characters.
            size_t last_match = prefix_substr(haystack.after_n(h_ptr - haystack.data), needle);
            return (last_match != not_found_k) ? last_match + (h_ptr - haystack.data) : not_found_k;
        }
#endif
    }; // end namespace intern

}; // end namespace cryptanalyslib::algorithm


