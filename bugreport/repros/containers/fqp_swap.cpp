#include "container/fq_packed_vector.h"
#include <cstdio>
int main() {
  using V = FqPackedVector<10, 7, uint64_t>; V v; for (uint32_t i=0;i<10;i++) v.set((3*i+5)%7, i);
  printf("before: "); for (uint32_t i=0;i<10;i++) printf("%u", (unsigned)v.get(i)); printf("\n");
  v.swap(0, 3);
  printf("after swap(0,3): "); for (uint32_t i=0;i<10;i++) printf("%u", (unsigned)v.get(i)); printf("  (expected 0 and 3 exchanged)\n");
}
