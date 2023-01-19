#include "int32_sort.h"
#include "float32_sort.h"

/* can save time by vectorizing xor loops */
/* can save time by integrating xor loops with int32_sort */

void float32_sort(float32_t *x,long long n)
{
  int32_t *y = (int32_t *) x;
  long long j;

  for (j = 0;j < n;++j) {
    int32_t yj = y[j];
    yj ^= ((uint32_t) (yj >> 31)) >> 1;
    y[j] = yj;
  }
  int32_sort(y,n);
  for (j = 0;j < n;++j) {
    int32_t yj = y[j];
    yj ^= ((uint32_t) (yj >> 31)) >> 1;
    y[j] = yj;
  }
}
