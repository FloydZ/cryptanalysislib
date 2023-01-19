#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "int32_sort.h"

#define int32 int32_t
#define sort int32_sort

static int cmp(const void *x,const void *y)
{
  const int32 a = *(int32 *) x;
  const int32 b = *(int32 *) y;
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

#define NMAX 65536

static int32 x[NMAX];
static int32 y[NMAX];

static void try(long long n,long long loops)
{
  long long j;

  assert(n <= NMAX);

  while (loops > 0) {
    for (j = 0;j < n;++j) x[j] = random() + random() - random();
    for (j = 0;j < n;++j) y[j] = x[j];
    sort(x,n);
    for (j = 1;j < n;++j) assert(x[j - 1] <= x[j]);
    qsort(y,n,sizeof(int32),cmp);
    for (j = 0;j < n;++j) assert(x[j] == y[j]);
    --loops;
  }

  for (j = 0;j < n;++j) x[j] = j;
  for (j = 0;j < n;++j) y[j] = x[j];
  sort(x,n);
  for (j = 1;j < n;++j) assert(x[j - 1] <= x[j]);
  qsort(y,n,sizeof(int32),cmp);
  for (j = 0;j < n;++j) assert(x[j] == y[j]);

  for (j = 0;j < n;++j) x[j] = -j;
  for (j = 0;j < n;++j) y[j] = x[j];
  sort(x,n);
  for (j = 1;j < n;++j) assert(x[j - 1] <= x[j]);
  qsort(y,n,sizeof(int32),cmp);
  for (j = 0;j < n;++j) assert(x[j] == y[j]);
}

int main()
{
  long long n;

  for (n = 0;n <= 4096;++n) try(n,1);
  for (n = 0;n <= 128;++n) try(n,100);
  for (n = 1;n <= NMAX;n += n) {
    try(n,1);
    try(n - 1,1);
    if (n < NMAX) try(n + 1,1);
    try(random() % n,1);
  }

  return 0;
}
