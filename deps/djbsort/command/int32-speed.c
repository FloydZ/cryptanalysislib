#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "int32_sort.h"
#include "djbsort_cpucycles.h"
#include "limits.h"

#define TIMINGS4 32
#define TIMINGS (4*(TIMINGS4) - 1)

static unsigned long long t[TIMINGS + 1];

int cmp(const void *av,const void *bv)
{
  const unsigned long long *a = av;
  const unsigned long long *b = bv;
  if (*a < *b) return -1;
  if (*a > *b) return 1;
  return 0;
}

static void printquartiles(const char *name,long long size)
{
  long long i, q1, q2, q3;

  for (i = 0;i < TIMINGS;++i) t[i] = t[i + 1] - t[i];
  qsort(t,TIMINGS,sizeof(t[0]),cmp);
  q1 = t[TIMINGS4 - 1];
  q2 = t[2 * TIMINGS4 - 1];
  q3 = t[3 * TIMINGS4 - 1];
  printf("%s %lld",name,size);
  printf(" %lld %lld %lld",q1,q2,q3);
  if (size > 0) {
    double per = 0.25/size;
    printf(" %f %f %f",q1*per,q2*per,q3*per);
  }
  printf("\n");
}

int main()
{
  long long i;
  void *m;
  int32_t *x;
  long long size;

  djbsort_cpucycles();
  limits();

  printf("int32 implementation %s\n",int32_sort_implementation);
  printf("int32 version %s\n",int32_sort_version);
  printf("int32 compiler %s\n",int32_sort_compiler);

  for (i = 0;i <= TIMINGS;++i) t[i] = djbsort_cpucycles();
  for (i = 0;i <= TIMINGS;++i) t[i] = djbsort_cpucycles();
  printquartiles("overhead",0);

  if (posix_memalign(&m,128,(1048576 + 64) * sizeof(int32_t)))
    exit(111);
  x = m;
  for (i = 0;i < 1048576 + 64;++i) x[i] = random();
  x += 32;

  for (size = 1;size <= 1048576;size += size) {
    for (i = 0;i <= TIMINGS;++i) t[i] = djbsort_cpucycles();
    for (i = 0;i <= TIMINGS;++i) {
      t[i] = djbsort_cpucycles();
      int32_sort(x,size);
    }
    printquartiles("int32",size);
  }

  for (size = 3;size <= 1048576;size += size) {
    for (i = 0;i <= TIMINGS;++i) t[i] = djbsort_cpucycles();
    for (i = 0;i <= TIMINGS;++i) {
      t[i] = djbsort_cpucycles();
      int32_sort(x,size);
    }
    printquartiles("int32",size);
  }

  return 0;
}
