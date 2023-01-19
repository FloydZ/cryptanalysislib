#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "float32_sort.h"

#define float32 float32_t
#define sort float32_sort

static int cmp(const void *x,const void *y)
{
  const float32 a = *(float32 *) x;
  const float32 b = *(float32 *) y;
  if (a < b) return -1;
  if (a > b) return 1;
  return 0;
}

#define NMAX 4096

static float32 x[NMAX];
static float32 y[NMAX];

int main()
{
  long long n, j;

  for (n = 0;n <= NMAX;++n) {
    for (j = 0;j < n;++j) x[j] = (random() + random() - random()) / (1.0 + random());
    for (j = 0;j < n;++j) y[j] = x[j];
    sort(x,n);
    for (j = 1;j < n;++j) assert(x[j - 1] <= x[j]);
    qsort(y,n,sizeof(float32),cmp);
    for (j = 0;j < n;++j) assert(x[j] == y[j]);

    for (j = 0;j < n;++j) x[j] = j;
    for (j = 0;j < n;++j) y[j] = x[j];
    sort(x,n);
    for (j = 1;j < n;++j) assert(x[j - 1] <= x[j]);
    qsort(y,n,sizeof(float32),cmp);
    for (j = 0;j < n;++j) assert(x[j] == y[j]);

    for (j = 0;j < n;++j) x[j] = -j;
    for (j = 0;j < n;++j) y[j] = x[j];
    sort(x,n);
    for (j = 1;j < n;++j) assert(x[j - 1] <= x[j]);
    qsort(y,n,sizeof(float32),cmp);
    for (j = 0;j < n;++j) assert(x[j] == y[j]);
  }
  return 0;
}
