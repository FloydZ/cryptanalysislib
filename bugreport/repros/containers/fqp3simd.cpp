#include "container/fq_packed_vector.h"
#include <cstdio>
int main() {
  using V = FqPackedVector<128, 3, uint64_t>;
  using S = V::S;
  printf("S=%s sizeof=%zu\n", __PRETTY_FUNCTION__, sizeof(S));
  V a, b, c, d;
  for (uint32_t i=0;i<128;i++){ a.set((i*7+i/5)%3, i); b.set((i*5+1+i/3)%3, i);}
  // scalar limb add
  for (int l=0;l<4;l++) d.__data[l] = V::add_T<uint64_t>(a.__data[l], b.__data[l]);
  S t = V::add256_T(S::unaligned_load(a.__data.data()), S::unaligned_load(b.__data.data()));
  S::unaligned_store(c.__data.data(), t);
  int bad=0, badd=0, first=-1;
  for (uint32_t i=0;i<128;i++){ unsigned e=(a.get(i)+b.get(i))%3; if(c.get(i)!=e){bad++; if(first<0) first=i;} if(d.get(i)!=e) badd++; }
  printf("add256_T wrong=%d first=%d ; add_T wrong=%d\n", bad, first, badd);
  // shift semantics
  S x = S::set1(0x8000000000000001ull); S y = x >> 1u; S z = x << 1u;
  printf("x>>1 limb0=%016llx  x<<1 limb0=%016llx\n", (unsigned long long)y.v64[0], (unsigned long long)z.v64[0]);
  V full; V::add(c, a, b); bad=0; first=-1;
  for (uint32_t i=0;i<128;i++){ unsigned e=(a.get(i)+b.get(i))%3; if(c.get(i)!=e){bad++; if(first<0) first=i;}}
  printf("V::add n=128 (internal_limbs=4 path) wrong=%d first=%d\n", bad, first);
}
