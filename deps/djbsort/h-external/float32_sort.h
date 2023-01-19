#ifndef float32_sort_H
#define float32_sort_H

typedef float float32_t;

#define float32_sort djbsort_float32
#define float32_sort_implementation djbsort_float32_implementation
#define float32_sort_version djbsort_float32_version
#define float32_sort_compiler djbsort_float32_compiler

#ifdef __cplusplus
extern "C" {
#endif

extern void float32_sort(float32_t *,long long) __attribute__((visibility("default")));

extern const char float32_sort_implementation[] __attribute__((visibility("default")));
extern const char float32_sort_version[] __attribute__((visibility("default")));
extern const char float32_sort_compiler[] __attribute__((visibility("default")));

#ifdef __cplusplus
}
#endif

#endif
