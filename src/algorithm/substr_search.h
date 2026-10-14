#pragma once 
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <cstring>

#if defined(USE_AVX2) || defined(USE_AVX512F)
#include <immintrin.h>
#endif

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

#ifdef USE_ARM
#include <arm_neon.h>
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
            // NOTE: `h_end` must not be computed if the needle is longer (UB)
            if (haystack.len < needle.len) {
                return not_found_k;
            }
            uint8_t const *const h_end = haystack.data + haystack.len - needle.len;
            size_t const n_suffix_len = needle.len - 4;
            // NOTE: unaligned 4-byte loads via memcpy (a reinterpret_cast load is UB)
            uint32_t n_prefix;
            __builtin_memcpy(&n_prefix, needle.data, 4);
            uint8_t const *n_suffix_ptr = needle.data + 4;

            for (; h_ptr <= h_end; h_ptr++) {
                uint32_t h_prefix;
                __builtin_memcpy(&h_prefix, h_ptr, 4);
                if (n_prefix == h_prefix) {
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

            // NOTE: `h_end` must not be computed if the needle is longer (UB)
            if (haystack.len < needle.len) {
                return not_found_k;
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
            // NOTE: `h_end` must not be computed if the needle is longer (UB)
            if (haystack.len < needle.len) {
                return not_found_k;
            }
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

            // NOTE: `h_end` must not be computed if the needle is longer (UB)
            if (haystack.len < needle.len) {
                return not_found_k;
            }
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
        /// source: http://0x80.pl/notesen/2016-11-28-simd-strfind.html
        ///
        /// Uses a two-stage approach - first checking first and last characters
        /// simultaneously with SIMD, then verifying matches with full comparison.
        /// This approach reduces the number of full comparisons needed.
        /// NOTE: the block loop only runs while both loads are inside the
        /// 	haystack, the remaining positions are checked by `naive_substr`.
        ///
        /// \param haystack[in,out]: The string to search in
        /// \param needle[in]: The substring to search for
        /// \return Position of the first match or not_found_k if not found
        constexpr size_t avx2_strstr_anysize(span_t &haystack,
                                                const span_t &needle) noexcept {
            const size_t n = haystack.len;
            const size_t k = needle.len;
            const uint8_t *s = haystack.data;
            if (k == 0) { return 0; }
            if (n < k) { return not_found_k; }

            const __m256i first = _mm256_set1_epi8(needle.data[0]);
            const __m256i last  = _mm256_set1_epi8(needle.data[k - 1]);

            size_t i = 0;
            for (; i + k - 1 + 32 <= n; i += 32) {
                const __m256i block_first = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i));
                const __m256i block_last  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + i + k - 1));

                const __m256i eq_first = _mm256_cmpeq_epi8(first, block_first);
                const __m256i eq_last  = _mm256_cmpeq_epi8(last, block_last);

                uint32_t mask = _mm256_movemask_epi8(_mm256_and_si256(eq_first, eq_last));
                while (mask != 0) {
                    const uint32_t bitpos = __builtin_ctz(mask);
                    if ((k <= 2) || are_equal(s + i + bitpos + 1, needle.data + 1, k - 2)) {
                        return i + bitpos;
                    }
                    mask = mask & (mask - 1u);
                }
            }

            span_t t = haystack.after_n(i);
            const size_t last_match = naive_substr(t, needle);
            return (last_match != not_found_k) ? last_match + i : not_found_k;
        }
#endif

#if defined(__SSE2__)
        /// SSE2 version of `avx2_strstr_anysize` (16 positions per block)
        /// source: http://0x80.pl/notesen/2016-11-28-simd-strfind.html
        /// \param haystack[in,out]: The string to search in
        /// \param needle[in]: The substring to search for
        /// \return Position of the first match or not_found_k if not found
        inline size_t sse2_strstr_anysize(span_t &haystack,
                                          const span_t &needle) noexcept {
            const size_t n = haystack.len;
            const size_t k = needle.len;
            const uint8_t *s = haystack.data;
            if (k == 0) { return 0; }
            if (n < k) { return not_found_k; }

            const __m128i first = _mm_set1_epi8(char(needle.data[0]));
            const __m128i last  = _mm_set1_epi8(char(needle.data[k - 1]));

            size_t i = 0;
            for (; i + k - 1 + 16 <= n; i += 16) {
                const __m128i block_first = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + i));
                const __m128i block_last  = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + i + k - 1));
                uint32_t mask = uint32_t(_mm_movemask_epi8(_mm_and_si128(_mm_cmpeq_epi8(first, block_first),
                                                                         _mm_cmpeq_epi8(last, block_last))));
                while (mask != 0) {
                    const uint32_t bitpos = __builtin_ctz(mask);
                    if ((k <= 2) || are_equal(s + i + bitpos + 1, needle.data + 1, k - 2)) {
                        return i + bitpos;
                    }
                    mask = mask & (mask - 1u);
                }
            }

            span_t t = haystack.after_n(i);
            const size_t last_match = naive_substr(t, needle);
            return (last_match != not_found_k) ? last_match + i : not_found_k;
        }
#endif

#if defined(USE_AVX512F) && defined(USE_AVX512BW)
        /// AVX-512BW version of `avx2_strstr_anysize` (64 positions per block)
        /// source: http://0x80.pl/notesen/2016-11-28-simd-strfind.html
        /// \param haystack[in,out]: The string to search in
        /// \param needle[in]: The substring to search for
        /// \return Position of the first match or not_found_k if not found
        inline size_t avx512bw_strstr_anysize(span_t &haystack,
                                              const span_t &needle) noexcept {
            const size_t n = haystack.len;
            const size_t k = needle.len;
            const uint8_t *s = haystack.data;
            if (k == 0) { return 0; }
            if (n < k) { return not_found_k; }

            const __m512i first = _mm512_set1_epi8(char(needle.data[0]));
            const __m512i last  = _mm512_set1_epi8(char(needle.data[k - 1]));

            size_t i = 0;
            for (; i + k - 1 + 64 <= n; i += 64) {
                const __m512i block_first = _mm512_loadu_si512(s + i);
                const __m512i block_last  = _mm512_loadu_si512(s + i + k - 1);
                uint64_t mask = _mm512_cmpeq_epi8_mask(first, block_first) &
                                _mm512_cmpeq_epi8_mask(last, block_last);
                while (mask != 0) {
                    const uint32_t bitpos = __builtin_ctzll(mask);
                    if ((k <= 2) || are_equal(s + i + bitpos + 1, needle.data + 1, k - 2)) {
                        return i + bitpos;
                    }
                    mask = mask & (mask - 1u);
                }
            }

            span_t t = haystack.after_n(i);
            const size_t last_match = naive_substr(t, needle);
            return (last_match != not_found_k) ? last_match + i : not_found_k;
        }
#endif

        /// SWAR (64 bit words) version of `avx2_strstr_anysize`, portable
        /// source: http://0x80.pl/notesen/2016-11-28-simd-strfind.html
        /// \param haystack[in,out]: The string to search in
        /// \param needle[in]: The substring to search for
        /// \return Position of the first match or not_found_k if not found
        constexpr size_t swar64_strstr_anysize(span_t &haystack,
                                               const span_t &needle) noexcept {
            const size_t n = haystack.len;
            const size_t k = needle.len;
            const uint8_t *s = haystack.data;
            if (k == 0) { return 0; }
            if (n < k) { return not_found_k; }

            constexpr uint64_t lo7 = 0x7f7f7f7f7f7f7f7full;
            const uint64_t first = 0x0101010101010101ull * needle.data[0];
            const uint64_t last  = 0x0101010101010101ull * needle.data[k - 1];

            size_t i = 0;
            for (; i + k - 1 + 8 <= n; i += 8) {
                uint64_t block_first = 0, block_last = 0;
                // little endian words: byte j is the position i + j
                for (uint32_t j = 0; j < 8; ++j) {
                    block_first |= uint64_t(s[i + j]) << (8u * j);
                    block_last  |= uint64_t(s[i + k - 1 + j]) << (8u * j);
                }

                // 0x80 in every byte, which is zero in `eq`
                const uint64_t eq = (block_first ^ first) | (block_last ^ last);
                uint64_t zeros = ~((((eq & lo7) + lo7) | eq) | lo7);
                while (zeros != 0) {
                    const uint32_t pos = uint32_t(__builtin_ctzll(zeros)) / 8u;
                    if ((k <= 2) || are_equal(s + i + pos + 1, needle.data + 1, k - 2)) {
                        return i + pos;
                    }
                    zeros &= zeros - 1u;
                }
            }

            span_t t = haystack.after_n(i);
            const size_t last_match = naive_substr(t, needle);
            return (last_match != not_found_k) ? last_match + i : not_found_k;
        }

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
            // NOTE: `h_end` must not be computed if the needle is longer (UB)
            if (haystack.len < needle.len) {
                return not_found_k;
            }
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
            // NOTE: `prefix_substr` takes a non-const reference, hence the named span
            span_t t = haystack.after_n(h_ptr - haystack.data);
            size_t last_match = prefix_substr(t, needle);
            return (last_match != not_found_k) ? last_match + (h_ptr - haystack.data) : not_found_k;
        }
#endif

#ifdef USE_ARM

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
            // NOTE: `h_end` must not be computed if the needle is longer (UB)
            if (haystack.len < needle.len) {
                return not_found_k;
            }
            uint8_t const *const h_end = haystack.data + haystack.len - needle.len;
            // NOTE: unaligned loads only via memcpy/byte loads (see `prefix_substr`)
            uint32_t n_prefix32;
            __builtin_memcpy(&n_prefix32, needle.data, 4);
            uint32x4_t const n_prefix = vdupq_n_u32(n_prefix32);

            uint8_t const *h_ptr = haystack.data;
            for (; (h_ptr + 16) <= h_end; h_ptr += 16) {

                uint32x4_t masks0 = vceqq_u32(vreinterpretq_u32_u8(vld1q_u8(h_ptr)), n_prefix);
                uint32x4_t masks1 = vceqq_u32(vreinterpretq_u32_u8(vld1q_u8(h_ptr + 1)), n_prefix);
                uint32x4_t masks2 = vceqq_u32(vreinterpretq_u32_u8(vld1q_u8(h_ptr + 2)), n_prefix);
                uint32x4_t masks3 = vceqq_u32(vreinterpretq_u32_u8(vld1q_u8(h_ptr + 3)), n_prefix);

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
            // NOTE: `prefix_substr` takes a non-const reference, hence the named span
            span_t t = haystack.after_n(h_ptr - haystack.data);
            size_t last_match = prefix_substr(t, needle);
            return (last_match != not_found_k) ? last_match + (h_ptr - haystack.data) : not_found_k;
        }
#endif
    }; // end namespace intern

    /// \param haystack[in]: the string to search in
    /// \param haystack_len[in]: its length
    /// \param needle[in]: the string to search for
    /// \param needle_len[in]: its length
    /// \return position of the first occurrence of `needle` in `haystack`,
    /// 	`internal::not_found_k` if there is none
    [[nodiscard]] inline size_t substr_search(const uint8_t *haystack,
                                              const size_t haystack_len,
                                              const uint8_t *needle,
                                              const size_t needle_len) noexcept {
        span_t h{const_cast<uint8_t *>(haystack), haystack_len};
        const span_t n{const_cast<uint8_t *>(needle), needle_len};
#if defined(USE_AVX512F) && defined(USE_AVX512BW)
        return internal::avx512bw_strstr_anysize(h, n);
#elif defined(USE_AVX2)
        return internal::avx2_strstr_anysize(h, n);
#elif defined(__SSE2__)
        return internal::sse2_strstr_anysize(h, n);
#else
        return internal::swar64_strstr_anysize(h, n);
#endif
    }
}; // end namespace cryptanalyslib::algorithm


