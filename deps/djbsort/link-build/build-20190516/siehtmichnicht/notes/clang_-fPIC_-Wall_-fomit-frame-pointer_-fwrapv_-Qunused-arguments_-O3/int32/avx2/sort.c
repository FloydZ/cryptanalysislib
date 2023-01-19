sort.c:939:40: error: always_inline function '_mm256_set1_epi32' requires target feature 'avx', but would be inlined into function 'djbsort_int32' that is compiled without support for 'avx'
    for (i = q>>3;i < q>>2;++i) y[i] = _mm256_set1_epi32(0x7fffffff);
                                       ^
sort.c:939:40: error: AVX vector return of type '__m256i' (vector of 4 'long long' values) without 'avx' enabled changes the ABI
sort.c:955:22: error: always_inline function '_mm256_loadu_si256' requires target feature 'avx', but would be inlined into function 'djbsort_int32' that is compiled without support for 'avx'
        int32x8 x0 = int32x8_load(&x[i]);
                     ^
sort.c:8:25: note: expanded from macro 'int32x8_load'
#define int32x8_load(z) _mm256_loadu_si256((__m256i *) (z))
                        ^
sort.c:955:22: error: AVX vector return of type '__m256i' (vector of 4 'long long' values) without 'avx' enabled changes the ABI
sort.c:8:25: note: expanded from macro 'int32x8_load'
#define int32x8_load(z) _mm256_loadu_si256((__m256i *) (z))
                        ^
sort.c:956:22: error: always_inline function '_mm256_loadu_si256' requires target feature 'avx', but would be inlined into function 'djbsort_int32' that is compiled without support for 'avx'
        int32x8 x1 = int32x8_load(&x[i+q]);
                     ^
sort.c:8:25: note: expanded from macro 'int32x8_load'
#define int32x8_load(z) _mm256_loadu_si256((__m256i *) (z))
                        ^
sort.c:956:22: error: AVX vector return of type '__m256i' (vector of 4 'long long' values) without 'avx' enabled changes the ABI
sort.c:8:25: note: expanded from macro 'int32x8_load'
#define int32x8_load(z) _mm256_loadu_si256((__m256i *) (z))
                        ^
sort.c:957:22: error: always_inline function '_mm256_loadu_si256' requires target feature 'avx', but would be inlined into function 'djbsort_int32' that is compiled without support for 'avx'
        int32x8 x2 = int32x8_load(&x[i+2*q]);
                     ^
sort.c:8:25: note: expanded from macro 'int32x8_load'
#define int32x8_load(z) _mm256_loadu_si256((__m256i *) (z))
                        ^
sort.c:957:22: error: AVX vector return of type '__m256i' (vector of 4 'long long' values) without 'avx' enabled changes the ABI
sort.c:8:25: note: expanded from macro 'int32x8_load'
#define int32x8_load(z) _mm256_loadu_si256((__m256i *) (z))
                        ^
sort.c:958:22: error: always_inline function '_mm256_loadu_si256' requires target feature 'avx', but would be inlined into function 'djbsort_int32' that is compiled without support for 'avx'
        int32x8 x3 = int32x8_load(&x[i+3*q]);
                     ^
sort.c:8:25: note: expanded from macro 'int32x8_load'
#define int32x8_load(z) _mm256_loadu_si256((__m256i *) (z))
                        ^
sort.c:958:22: error: AVX vector return of type '__m256i' (vector of 4 'long long' values) without 'avx' enabled changes the ABI
sort.c:8:25: note: expanded from macro 'int32x8_load'
#define int32x8_load(z) _mm256_loadu_si256((__m256i *) (z))
                        ^
sort.c:959:9: error: always_inline function '_mm256_min_epi32' requires target feature 'avx2', but would be inlined into function 'djbsort_int32' that is compiled without support for 'avx2'
        int32x8_MINMAX(x0,x2);
        ^
sort.c:15:15: note: expanded from macro 'int32x8_MINMAX'
  int32x8 c = int32x8_min(a,b); \
              ^
sort.c:10:21: note: expanded from macro 'int32x8_min'
#define int32x8_min _mm256_min_epi32
                    ^
sort.c:959:9: error: AVX vector argument of type '__m256i' (vector of 4 'long long' values) without 'avx' enabled changes the ABI
sort.c:15:15: note: expanded from macro 'int32x8_MINMAX'
  int32x8 c = int32x8_min(a,b); \
              ^
sort.c:10:21: note: expanded from macro 'int32x8_min'
#define int32x8_min _mm256_min_epi32
                    ^
sort.c:959:9: error: always_inline function '_mm256_max_epi32' requires target feature 'avx2', but would be inlined into function 'djbsort_int32' that is compiled without support for 'avx2'
sort.c:16:7: note: expanded from macro 'int32x8_MINMAX'
  b = int32x8_max(a,b); \
      ^
sort.c:11:21: note: expanded from macro 'int32x8_max'
#define int32x8_max _mm256_max_epi32
                    ^
sort.c:959:9: error: AVX vector argument of type '__m256i' (vector of 4 'long long' values) without 'avx' enabled changes the ABI
sort.c:16:7: note: expanded from macro 'int32x8_MINMAX'
  b = int32x8_max(a,b); \
      ^
sort.c:11:21: note: expanded from macro 'int32x8_max'
#define int32x8_max _mm256_max_epi32
                    ^
sort.c:960:9: error: always_inline function '_mm256_min_epi32' requires target feature 'avx2', but would be inlined into function 'djbsort_int32' that is compiled without support for 'avx2'
        int32x8_MINMAX(x1,x3);
        ^
sort.c:15:15: note: expanded from macro 'int32x8_MINMAX'
  int32x8 c = int32x8_min(a,b); \
              ^
sort.c:10:21: note: expanded from macro 'int32x8_min'
#define int32x8_min _mm256_min_epi32
                    ^
sort.c:960:9: error: AVX vector argument of type '__m256i' (vector of 4 'long long' values) without 'avx' enabled changes the ABI
sort.c:15:15: note: expanded from macro 'int32x8_MINMAX'
  int32x8 c = int32x8_min(a,b); \
              ^
sort.c:10:21: note: expanded from macro 'int32x8_min'
#define int32x8_min _mm256_min_epi32
                    ^
sort.c:960:9: error: always_inline function '_mm256_max_epi32' requires target feature 'avx2', but would be inlined into function 'djbsort_int32' that is compiled without support for 'avx2'
sort.c:16:7: note: expanded from macro 'int32x8_MINMAX'
  b = int32x8_max(a,b); \
      ^
sort.c:11:21: note: expanded from macro 'int32x8_max'
#define int32x8_max _mm256_max_epi32
                    ^
sort.c:960:9: error: AVX vector argument of type '__m256i' (vector of 4 'long long' values) without 'avx' enabled changes the ABI
sort.c:16:7: note: expanded from macro 'int32x8_MINMAX'
  b = int32x8_max(a,b); \
      ^
sort.c:11:21: note: expanded from macro 'int32x8_max'
#define int32x8_max _mm256_max_epi32
                    ^
sort.c:961:9: error: always_inline function '_mm256_min_epi32' requires target feature 'avx2', but would be inlined into function 'djbsort_int32' that is compiled without support for 'avx2'
        int32x8_MINMAX(x0,x1);
        ^
sort.c:15:15: note: expanded from macro 'int32x8_MINMAX'
  int32x8 c = int32x8_min(a,b); \
              ^
sort.c:10:21: note: expanded from macro 'int32x8_min'
#define int32x8_min _mm256_min_epi32
                    ^
fatal error: too many errors emitted, stopping now [-ferror-limit=]
20 errors generated.
