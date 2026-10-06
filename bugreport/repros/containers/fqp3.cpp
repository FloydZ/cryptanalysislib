#include "container/fq_packed_vector.h"
#include <cstdio>
#include <vector>
template<class V> std::vector<unsigned> ref(const V&v){ std::vector<unsigned> r; for(uint32_t i=0;i<V::length;i++) r.push_back(v.get(i)); return r;}
template<class V> void fill(V&v, uint32_t seed){ for(uint32_t i=0;i<V::length;i++) v.set((i*7+seed+i/5)%3, i);}
int main() {
  { // runtime neg(lower,upper)
    using V = FqPackedVector<100, 3, uint64_t>; V v; fill(v,1); auto r = ref(v);
    v.neg(0, 100); int bad=0, first=-1;
    for (uint32_t i=0;i<100;i++){ if (v.get(i) != (3-r[i])%3) { bad++; if(first<0) first=i; } }
    printf("q3 neg(0,100): %d wrong coords, first=%d\n", bad, first);
    V w; fill(w,1); auto r2=ref(w); w.neg(5, 40); bad=0; first=-1;
    for (uint32_t i=0;i<100;i++){ unsigned e = (i>=5&&i<40)?(3-r2[i])%3:r2[i]; if (w.get(i)!=e){bad++; if(first<0) first=i;} }
    printf("q3 neg(5,40): %d wrong coords, first=%d\n", bad, first);
  }
  { // template neg<k_lower,k_upper>
    using V = FqPackedVector<100, 3, uint64_t>; V v; fill(v,2); auto r = ref(v);
    v.neg<0, 80>(); int bad=0, first=-1;
    for (uint32_t i=0;i<100;i++){ unsigned e = (i<80)?(3-r[i])%3:r[i]; if (v.get(i)!=e){bad++; if(first<0) first=i;} }
    printf("q3 neg<0,80>: %d wrong coords, first=%d\n", bad, first);
  }
  { // add with n >= 1024
    using V = FqPackedVector<1100, 3, uint64_t>; V a,b,c; fill(a,0); fill(b,1);
    V::add(c,a,b); int bad=0, first=-1;
    for (uint32_t i=0;i<1100;i++){ unsigned e=(a.get(i)+b.get(i))%3; if (c.get(i)!=e){bad++; if(first<0) first=i;} }
    printf("q3 add n=1100: %d wrong coords, first=%d\n", bad, first);
  }
  { // add_only_weight_partly within one limb
    using V = FqPackedVector<100, 3, uint64_t>; V a,b,c; fill(a,0); fill(b,1);
    uint32_t w = V::add_only_weight_partly<2,10>(c,a,b); uint32_t e=0;
    for (uint32_t i=2;i<10;i++) e += ((a.get(i)+b.get(i))%3)!=0;
    printf("q3 add_only_weight_partly<2,10>: got %u exp %u\n", w, e);
  }
  { // filter2count_range_T<kl,ku>
    using V = FqPackedVector<32, 3, uint64_t>; V v; for(uint32_t i=0;i<32;i++) v.set(2,i);
    uint32_t c = V::filter2count_range_T<4, 10, uint64_t>(v.ptr()[0]);
    printf("q3 filter2count_range_T<4,10> all-twos: got %u exp 6\n", c);
    uint32_t d = v.filter2count_T<32, uint64_t>();
    printf("q3 filter2count_T<32>() all-twos: got %u exp 32\n", d);
    uint32_t d2 = v.filter2count_T<20, uint64_t>();
    printf("q3 filter2count_T<20>() all-twos: got %u exp 20\n", d2);
  }
  { // mod_T on raw values 3
    using V = FqPackedVector<32, 3, uint64_t>;
    uint64_t a=0; for (int i=0;i<32;i++) a |= uint64_t(i%4) << (2*i);
    uint64_t m = V::mod_T<uint64_t>(a); int bad=0;
    for (int i=0;i<32;i++) if (((m>>(2*i))&3) != (uint64_t)((i%4)%3)) bad++;
    printf("q3 mod_T: %d wrong coords\n", bad);
  }
}
