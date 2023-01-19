#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "uint32_sort.h"

#define uint32 uint32_t
#define sort uint32_sort

static int cmp(const void *x,const void *y)
{
  const uint32 a = *(uint32 *) x;
  const uint32 b = *(uint32 *) y;
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

#define NMAX 4096

static uint32 x[NMAX];
static uint32 y[NMAX];

int main()
{
  long long n, j;

  for (n = 0;n <= NMAX;++n) {
    for (j = 0;j < n;++j) x[j] = random() + random() + random() + random();
    for (j = 0;j < n;++j) y[j] = x[j];
    sort(x,n);
    for (j = 1;j < n;++j) assert(x[j - 1] <= x[j]);
    qsort(y,n,sizeof(uint32),cmp);
    for (j = 0;j < n;++j) assert(x[j] == y[j]);

    for (j = 0;j < n;++j) x[j] = j;
    for (j = 0;j < n;++j) y[j] = x[j];
    sort(x,n);
    for (j = 1;j < n;++j) assert(x[j - 1] <= x[j]);
    qsort(y,n,sizeof(uint32),cmp);
    for (j = 0;j < n;++j) assert(x[j] == y[j]);

    for (j = 0;j < n;++j) x[j] = -j;
    for (j = 0;j < n;++j) y[j] = x[j];
    sort(x,n);
    for (j = 1;j < n;++j) assert(x[j - 1] <= x[j]);
    qsort(y,n,sizeof(uint32),cmp);
    for (j = 0;j < n;++j) assert(x[j] == y[j]);
  }
  return 0;
}
