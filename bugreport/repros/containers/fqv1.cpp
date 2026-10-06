#include "container/fq_vector.h"
#include "container/fq_packed_vector.h"
#include "container/kAry_type.h"
#include <cstdio>
template<class V> int hashcheck(){ V v; int bad=0; 
  for (uint32_t i=0;i<V::length;i++) v.set((i*13+7+i/3)%V::q, i);
  for (uint32_t l=0;l<V::length;l++) for (uint32_t h=l+1; h<=V::length; h++){
    if ((h-l)*V::qbits > 63) break;
    uint64_t e=0; for (uint32_t i=l;i<h;i++) e |= uint64_t(v.get(i)) << ((i-l)*V::qbits);
    if (v.hash(l,h)!=e) bad++; }
  return bad; }
int main() {
  setvbuf(stdout,NULL,_IONBF,0);
  { using V = FqNonPackedVector<10, 7, uint8_t>; V v; v.minus_one(); printf("FqNonPackedVector<10,7,u8>::minus_one -> v[0]=%u (exp 6)\n", (unsigned)v.get(0)); }
  { using V = FqNonPackedVector<8, 7, uint8_t>; V in, out; for (uint32_t i=0;i<8;i++) in.set(i%7, i);
    V::rol(out, in, 3); printf("rol(01234560,3): "); for (uint32_t i=0;i<8;i++) printf("%u",(unsigned)out.get(i));
    V::ror(out, in, 3); printf("   ror: "); for (uint32_t i=0;i<8;i++) printf("%u",(unsigned)out.get(i)); printf("  (rol drops wrapped elems)\n"); }
  { using V = FqNonPackedVector<40, 4, uint8_t>; V v; for (uint32_t i=0;i<40;i++) v.set(3, i);
    printf("FqNonPackedVector<40,4>::hash(0,32) (64 bits) = %llx (exp ffffffffffffffff)\n", (unsigned long long)v.hash(0,32)); }
  printf("FqPacked hash mismatches: q4u8=%d q5u64=%d q7u8=%d q16u64=%d q5u8=%d q11u16=%d\n",
    hashcheck<FqPackedVector<77,4,uint8_t>>(), hashcheck<FqPackedVector<77,5,uint64_t>>(), hashcheck<FqPackedVector<77,7,uint8_t>>(),
    hashcheck<FqPackedVector<77,16,uint64_t>>(), hashcheck<FqPackedVector<77,5,uint8_t>>(), hashcheck<FqPackedVector<77,11,uint16_t>>());
  { using K = kAry_Type_T<7>; K x(5), a(3), b(4); x.addmul(a, b); printf("kAry<7> 5 + 3*4 addmul = %u (exp %u)\n", (unsigned)x.value(), (5+12)%7); }
  { using K = kAry_Type_T<4294967291ull>; uint32_t a = 4294967290u; printf("kAry<2^32-5>::add_T(q-1,q-1)=%u (exp %u), sub_T(1,q-1)=%u (exp 2)\n",
      (unsigned)K::add_T(a,a), (unsigned)((2ull*a)%4294967291ull), (unsigned)K::sub_T(1,a)); }
  { using V = FqPackedVector<16, 7, uint64_t>; using S = V::S; S a = S::set1(5ull); S m = V::mod256_T(a);
    printf("FqPackedVectorMeta<..,7>::mod256_T(5) lane0 low elem = %llu (exp 5)\n", (unsigned long long)(m.v64[0] & 7)); }
}
