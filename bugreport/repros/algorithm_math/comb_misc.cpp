typedef unsigned long ulong;
#include <cstdint>
#include <set>
#include <iostream>
#include "fork.h"
#include <algorithm>
#define CRYPTANALYSISLIB_ALGORITHM_MAX_H
#define CRYPTANALYSISLIB_ALGORITHM_MIN_H
namespace cryptanalysislib { using std::max; using std::min; }
#include "math/math.h"
#include "combination/revolving_door.h"
#include "combination/colex.h"
#include "combination/shifts.h"
#include "combination/lexicographic.h"

void revdoor(uint32_t n, uint32_t k){
  combination_revdoor c(n,k);
  std::set<uint64_t> seen; size_t bad=0, posbad=0, cnt=0;
  auto mask=[&]{ uint64_t m=0; for(uint32_t i=0;i<k;i++) m|=1ull<<c.data()[i]; return m; };
  uint64_t prev=mask(); seen.insert(prev); cnt=1;
  uint32_t k1,k2;
  while(c.next(&k1,&k2)){ uint64_t m=mask(); cnt++; if(__builtin_popcountll(m)!=k || (n<64 && (m>>n))) bad++;
    uint64_t expect = prev ^ (1ull<<k1) ^ (1ull<<k2);
    if(expect!=m || !((prev>>k1)&1) || ((prev>>k2)&1)) { if(posbad<3) printf("  revdoor(%u,%u) step %zu: k1=%u k2=%u prev=%llx now=%llx\n",n,k,cnt,k1,k2,(unsigned long long)prev,(unsigned long long)m); posbad++;}
    seen.insert(m); prev=m; if(cnt>bc(n,k)+5) break; }
  printf("revdoor(%u,%u): count=%zu distinct=%zu expected=%llu badweight=%zu posmismatch=%zu\n",n,k,cnt,seen.size(),(unsigned long long)bc(n,k),bad,posbad);
}
template<typename T, uint32_t n, uint32_t k> void colex(){
  enumeration_colex<T,n,k> e; std::set<T> s; size_t bad=0;
  T last = enumeration_colex<T,n,k>::last_comb();
  for(size_t i=0;i<bc(n,k);i++){ T w=e.next(); if(__builtin_popcountll((uint64_t)w)!=k || (n<64 && ((uint64_t)w>>n))) bad++; s.insert(w);}
  printf("colex<%zu-bit,%u,%u>: first=%llx distinct=%zu expected=%llu badweight=%zu last_comb=%llx\n",sizeof(T)*8,n,k,(unsigned long long)enumeration_colex<T,n,k>::first_comb(),s.size(),(unsigned long long)bc(n,k),bad,(unsigned long long)last);
}
template<uint32_t n, uint32_t k> void shifts(){
  bit_comb_shifts<uint64_t,n,k> b; std::set<uint64_t> s; s.insert(b.x_); size_t cnt=1;
  uint64_t x; while((x=b.next())!=0 && cnt<1000){ s.insert(x); cnt++; }
  printf("shifts<%u,%u>: count=%zu distinct=%zu expected=%llu\n",n,k,cnt,s.size(),(unsigned long long)bc(n,k));
}
static uint64_t ref_negidx2lexrev(uint64_t k){ uint64_t z=0; if(!k) return 0; uint64_t h=1ull<<(63-__builtin_clzll(k)); while(k){ while(0==(h&k)) h>>=1; z^=h; ++k; k&=h-1;} return z; }
int main(){
  for (auto [n,k] : std::initializer_list<std::pair<uint32_t,uint32_t>>{{5,1},{5,2},{5,3},{6,3},{10,4},{7,7},{1,1},{20,10},{33,2},{64,3}}) CASE("rd", revdoor(n,k));
  CASE("rd k=0", revdoor(5,0));
  CASE("colex64", (colex<uint64_t,10,3>()));
  CASE("colex32", (colex<uint32_t,10,3>()));
  CASE("colex16", (colex<uint16_t,10,3>()));
  CASE("colex64 n=64", (colex<uint64_t,64,2>()));
  CASE("colex k=n", (colex<uint64_t,5,5>()));
  CASE("sh 5,1", (shifts<5,1>()));
  CASE("sh 5,2", (shifts<5,2>()));
  CASE("sh 5,3", (shifts<5,3>()));
  CASE("sh 6,3", (shifts<6,3>()));
  CASE("enum_t p4", { enumerate_t<10,4> e; size_t c=0; e.enumerate([&](const uint16_t*){c++;}); printf("enumerate_t<10,4>::enumerate calls=%zu expected %llu; list_size=%zu\n",c,(unsigned long long)bc(10,4),e.list_size()); });
  CASE("enum_t p3", { enumerate_t<10,3> e; size_t c=0; e.enumerate([&](const uint16_t*){c++;}); printf("enumerate_t<10,3>::enumerate calls=%zu expected %llu; list_size=%zu\n",c,(unsigned long long)bc(10,3),e.list_size()); });
  CASE("enum_t p2 n=3", { enumerate_t<3,2> e; size_t c=0; e.enumerate([&](const uint16_t*){c++;}); printf("enumerate_t<3,2> calls=%zu expected 3\n",c); });
  for (uint64_t k : {1,2,3,4,5,8,16,17}) CASE("negidx", printf("negidx2lexrev(%llu)=%llx expected %llx\n",(unsigned long long)k,(unsigned long long)BinaryLexicographic<uint64_t>::negidx2lexrev(k),(unsigned long long)ref_negidx2lexrev(k)));
  for (uint64_t k : {1,2,3,4,5,8,16,17}) CASE("roundtrip", printf("lexrev2negidx(ref(%llu))=%zu\n",(unsigned long long)k,BinaryLexicographic<uint64_t>::lexrev2negidx(ref_negidx2lexrev(k))));
}
