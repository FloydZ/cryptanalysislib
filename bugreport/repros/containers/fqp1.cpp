#include "container/fq_packed_vector.h"
#include <cstdio>
int main() { setvbuf(stdout,NULL,_IONBF,0); printf("start\n");
  // 1. swap
  { using V = FqPackedVector<10, 7, uint64_t>; V v; for (uint32_t i=0;i<10;i++) v.set(i%7, i);
    v.swap(1, 2); printf("swap(1,2): v[1]=%u (exp 2) v[2]=%u (exp 1)\n", (unsigned)v.get(1), (unsigned)v.get(2)); }
  // 2. right_shift
  { using V = FqPackedVector<10, 7, uint64_t>; V v; for (uint32_t i=0;i<10;i++) v.set(i%7, i);
    v.right_shift(2); printf("right_shift(2): "); for (uint32_t i=0;i<10;i++) printf("%u", (unsigned)v.get(i)); printf("  exp 2345601200->23456012 00\n"); }
  // 3. add overflow in DataType for q in (128,256]
  { using V = FqPackedVector<4, 251, uint64_t>; V a, b, c; a.set(250, 0); b.set(250, 0);
    V::add(c, a, b); printf("q=251 add 250+250 = %u (exp %u)\n", (unsigned)c.get(0), (500u%251)); }
  // 4. mul overflow q=65521
  { using V = FqPackedVector<4, 65521, uint64_t>; V a, b, c; a.set(65520, 0); b.set(65520, 0);
    V::mul(c, a, b); printf("q=65521 mul = %u (exp %u)\n", (unsigned)c.get(0), (unsigned)((65520ull*65520ull)%65521)); }
  return 0;
}
