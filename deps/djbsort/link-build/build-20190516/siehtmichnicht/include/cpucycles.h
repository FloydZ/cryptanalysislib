#ifndef cpucycles_H
#define cpucycles_H

#define cpucycles djbsort_cpucycles
#define cpucycles_persecond djbsort_cpucycles_persecond
#define cpucycles_implementation djbsort_cpucycles_implementation

#ifdef __cplusplus
extern "C" {
#endif

extern long long cpucycles(void) __attribute__((visibility("default")));
extern long long cpucycles_persecond(void) __attribute__((visibility("default")));
extern const char cpucycles_implementation[] __attribute__((visibility("default")));

#ifdef __cplusplus
}
#endif

#endif
