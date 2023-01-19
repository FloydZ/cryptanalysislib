sort.c: In function ‘djbsort_int32’:
sort.c:939:38: warning: AVX vector return without AVX enabled changes the ABI [-Wpsabi]
  939 |     for (i = q>>3;i < q>>2;++i) y[i] = _mm256_set1_epi32(0x7fffffff);
      |                                 ~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sort.c: In function ‘merge16_finish’:
sort.c:54:13: note: the ABI for passing parameters with 32-byte alignment has changed in GCC 4.6
   54 | static void merge16_finish(int32 *x,int32x8 x0,int32x8 x1,int flagdown)
      |             ^~~~~~~~~~~~~~
In file included from /usr/lib/gcc/x86_64-pc-linux-gnu/12.2.1/include/immintrin.h:43,
                 from sort.c:4:
/usr/lib/gcc/x86_64-pc-linux-gnu/12.2.1/include/avxintrin.h: In function ‘minmax_vector’:
/usr/lib/gcc/x86_64-pc-linux-gnu/12.2.1/include/avxintrin.h:933:1: error: inlining failed in call to ‘always_inline’ ‘_mm256_storeu_si256’: target specific option mismatch
  933 | _mm256_storeu_si256 (__m256i_u *__P, __m256i __A)
      | ^~~~~~~~~~~~~~~~~~~
sort.c:9:28: note: called from here
    9 | #define int32x8_store(z,i) _mm256_storeu_si256((__m256i *) (z),(i))
      |                            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sort.c:37:5: note: in expansion of macro ‘int32x8_store’
   37 |     int32x8_store(y + n - 8,y0);
      |     ^~~~~~~~~~~~~
/usr/lib/gcc/x86_64-pc-linux-gnu/12.2.1/include/avxintrin.h:933:1: error: inlining failed in call to ‘always_inline’ ‘_mm256_storeu_si256’: target specific option mismatch
  933 | _mm256_storeu_si256 (__m256i_u *__P, __m256i __A)
      | ^~~~~~~~~~~~~~~~~~~
sort.c:9:28: note: called from here
    9 | #define int32x8_store(z,i) _mm256_storeu_si256((__m256i *) (z),(i))
      |                            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sort.c:36:5: note: in expansion of macro ‘int32x8_store’
   36 |     int32x8_store(x + n - 8,x0);
      |     ^~~~~~~~~~~~~
In file included from /usr/lib/gcc/x86_64-pc-linux-gnu/12.2.1/include/immintrin.h:47:
/usr/lib/gcc/x86_64-pc-linux-gnu/12.2.1/include/avx2intrin.h:363:1: error: inlining failed in call to ‘always_inline’ ‘_mm256_max_epi32’: target specific option mismatch
  363 | _mm256_max_epi32 (__m256i __A, __m256i __B)
      | ^~~~~~~~~~~~~~~~
sort.c:11:21: note: called from here
   11 | #define int32x8_max _mm256_max_epi32
      |                     ^
sort.c:16:7: note: in expansion of macro ‘int32x8_max’
   16 |   b = int32x8_max(a,b); \
      |       ^~~~~~~~~~~
sort.c:35:5: note: in expansion of macro ‘int32x8_MINMAX’
   35 |     int32x8_MINMAX(x0,y0);
      |     ^~~~~~~~~~~~~~
/usr/lib/gcc/x86_64-pc-linux-gnu/12.2.1/include/avx2intrin.h:405:1: error: inlining failed in call to ‘always_inline’ ‘_mm256_min_epi32’: target specific option mismatch
  405 | _mm256_min_epi32 (__m256i __A, __m256i __B)
      | ^~~~~~~~~~~~~~~~
sort.c:10:21: note: called from here
   10 | #define int32x8_min _mm256_min_epi32
      |                     ^
sort.c:15:15: note: in expansion of macro ‘int32x8_min’
   15 |   int32x8 c = int32x8_min(a,b); \
      |               ^~~~~~~~~~~
sort.c:35:5: note: in expansion of macro ‘int32x8_MINMAX’
   35 |     int32x8_MINMAX(x0,y0);
      |     ^~~~~~~~~~~~~~
/usr/lib/gcc/x86_64-pc-linux-gnu/12.2.1/include/avxintrin.h:927:1: error: inlining failed in call to ‘always_inline’ ‘_mm256_loadu_si256’: target specific option mismatch
  927 | _mm256_loadu_si256 (__m256i_u const *__P)
      | ^~~~~~~~~~~~~~~~~~
sort.c:8:25: note: called from here
    8 | #define int32x8_load(z) _mm256_loadu_si256((__m256i *) (z))
      |                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sort.c:34:18: note: in expansion of macro ‘int32x8_load’
   34 |     int32x8 y0 = int32x8_load(y + n - 8);
      |                  ^~~~~~~~~~~~
/usr/lib/gcc/x86_64-pc-linux-gnu/12.2.1/include/avxintrin.h:927:1: error: inlining failed in call to ‘always_inline’ ‘_mm256_loadu_si256’: target specific option mismatch
  927 | _mm256_loadu_si256 (__m256i_u const *__P)
      | ^~~~~~~~~~~~~~~~~~
sort.c:8:25: note: called from here
    8 | #define int32x8_load(z) _mm256_loadu_si256((__m256i *) (z))
      |                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sort.c:33:18: note: in expansion of macro ‘int32x8_load’
   33 |     int32x8 x0 = int32x8_load(x + n - 8);
      |                  ^~~~~~~~~~~~
/usr/lib/gcc/x86_64-pc-linux-gnu/12.2.1/include/avxintrin.h:933:1: error: inlining failed in call to ‘always_inline’ ‘_mm256_storeu_si256’: target specific option mismatch
  933 | _mm256_storeu_si256 (__m256i_u *__P, __m256i __A)
      | ^~~~~~~~~~~~~~~~~~~
sort.c:9:28: note: called from here
    9 | #define int32x8_store(z,i) _mm256_storeu_si256((__m256i *) (z),(i))
      |                            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sort.c:45:5: note: in expansion of macro ‘int32x8_store’
   45 |     int32x8_store(y,y0);
      |     ^~~~~~~~~~~~~
/usr/lib/gcc/x86_64-pc-linux-gnu/12.2.1/include/avxintrin.h:933:1: error: inlining failed in call to ‘always_inline’ ‘_mm256_storeu_si256’: target specific option mismatch
  933 | _mm256_storeu_si256 (__m256i_u *__P, __m256i __A)
      | ^~~~~~~~~~~~~~~~~~~
sort.c:9:28: note: called from here
    9 | #define int32x8_store(z,i) _mm256_storeu_si256((__m256i *) (z),(i))
      |                            ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sort.c:44:5: note: in expansion of macro ‘int32x8_store’
   44 |     int32x8_store(x,x0);
      |     ^~~~~~~~~~~~~
/usr/lib/gcc/x86_64-pc-linux-gnu/12.2.1/include/avx2intrin.h:363:1: error: inlining failed in call to ‘always_inline’ ‘_mm256_max_epi32’: target specific option mismatch
  363 | _mm256_max_epi32 (__m256i __A, __m256i __B)
      | ^~~~~~~~~~~~~~~~
sort.c:11:21: note: called from here
   11 | #define int32x8_max _mm256_max_epi32
      |                     ^
sort.c:16:7: note: in expansion of macro ‘int32x8_max’
   16 |   b = int32x8_max(a,b); \
      |       ^~~~~~~~~~~
sort.c:43:5: note: in expansion of macro ‘int32x8_MINMAX’
   43 |     int32x8_MINMAX(x0,y0);
      |     ^~~~~~~~~~~~~~
/usr/lib/gcc/x86_64-pc-linux-gnu/12.2.1/include/avx2intrin.h:405:1: error: inlining failed in call to ‘always_inline’ ‘_mm256_min_epi32’: target specific option mismatch
  405 | _mm256_min_epi32 (__m256i __A, __m256i __B)
      | ^~~~~~~~~~~~~~~~
sort.c:10:21: note: called from here
   10 | #define int32x8_min _mm256_min_epi32
      |                     ^
sort.c:15:15: note: in expansion of macro ‘int32x8_min’
   15 |   int32x8 c = int32x8_min(a,b); \
      |               ^~~~~~~~~~~
sort.c:43:5: note: in expansion of macro ‘int32x8_MINMAX’
   43 |     int32x8_MINMAX(x0,y0);
      |     ^~~~~~~~~~~~~~
/usr/lib/gcc/x86_64-pc-linux-gnu/12.2.1/include/avxintrin.h:927:1: error: inlining failed in call to ‘always_inline’ ‘_mm256_loadu_si256’: target specific option mismatch
  927 | _mm256_loadu_si256 (__m256i_u const *__P)
      | ^~~~~~~~~~~~~~~~~~
sort.c:8:25: note: called from here
    8 | #define int32x8_load(z) _mm256_loadu_si256((__m256i *) (z))
      |                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sort.c:42:18: note: in expansion of macro ‘int32x8_load’
   42 |     int32x8 y0 = int32x8_load(y);
      |                  ^~~~~~~~~~~~
/usr/lib/gcc/x86_64-pc-linux-gnu/12.2.1/include/avxintrin.h:927:1: error: inlining failed in call to ‘always_inline’ ‘_mm256_loadu_si256’: target specific option mismatch
  927 | _mm256_loadu_si256 (__m256i_u const *__P)
      | ^~~~~~~~~~~~~~~~~~
sort.c:8:25: note: called from here
    8 | #define int32x8_load(z) _mm256_loadu_si256((__m256i *) (z))
      |                         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
sort.c:41:18: note: in expansion of macro ‘int32x8_load’
   41 |     int32x8 x0 = int32x8_load(x);
      |                  ^~~~~~~~~~~~
