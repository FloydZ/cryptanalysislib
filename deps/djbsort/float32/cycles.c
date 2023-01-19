#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/resource.h>
#include "float32_sort.h"
#include "cpucycles.h"

void limits()
{
#ifdef RLIM_INFINITY
  struct rlimit r;
  r.rlim_cur = 0;
  r.rlim_max = 0;
#ifdef RLIMIT_NOFILE
  setrlimit(RLIMIT_NOFILE,&r);
#endif
#ifdef RLIMIT_NPROC
  setrlimit(RLIMIT_NPROC,&r);
#endif
#ifdef RLIMIT_CORE
  setrlimit(RLIMIT_CORE,&r);
#endif
#endif
}

#define TIMINGS 127
static long long cycles[TIMINGS + 1];

int main()
{
  long long i;
  long long j;
  long long abovej;
  long long belowj;
  long long timings;
  void *m;
  float32_t *x;

  alarm(3600);
  cpucycles();
  limits();

  if (posix_memalign(&m,128,(1024 + 64) * sizeof(float32_t)))
    exit(111);
  x = m;
  for (i = 0;i < 1024 + 64;++i) x[i] = random() - random();
  x += 32;

  timings = 3;
  for (;;) {

    for (i = 0;i <= timings;++i) {
      cycles[i] = cpucycles();
    }
    for (i = 0;i < timings;++i) {
      cycles[i] = cpucycles();
      float32_sort(x,1024);
    }
    cycles[timings] = cpucycles();

    for (i = 0;i < timings;++i) cycles[i] = cycles[i + 1] - cycles[i];

    for (j = 0;j < timings;++j) {
      belowj = 0;
      for (i = 0;i < timings;++i) if (cycles[i] < cycles[j]) ++belowj;
      abovej = 0;
      for (i = 0;i < timings;++i) if (cycles[i] > cycles[j]) ++abovej;
      if (belowj * 2 < timings && abovej * 2 < timings) break;
    }

    if (timings == 3) {
      if (cycles[j] < 100000) { timings = TIMINGS; continue; }
      if (cycles[j] < 1000000) { timings = 15; continue; }
    }

    printf("%lld\n",cycles[j]);

    return 0;
  }
}
