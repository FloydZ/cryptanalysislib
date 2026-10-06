#include <cstdint>
#include <set>
#include <vector>
#include "fork.h"
#include "combination/chase.h"
template<uint32_t n, uint32_t p>
void dump_chase_t(){
  chase_t<n,p> c;
  uint64_t v = (1ull<<p)-1; size_t i=0;
  std::set<uint64_t> seen; seen.insert(v);
  c.enumerate([&](uint16_t a, uint16_t b){ uint64_t old=v; if (a>=n||b>=n||a==b) printf("  call %zu: bad indices (%u,%u)\n",i,a,b);
    v ^= 1ull<<a; v ^= 1ull<<b; if(__builtin_popcountll(v)!=p || !seen.insert(v).second) { printf("  call %zu: (%u,%u) %llx -> %llx %s\n",i,a,b,(unsigned long long)old,(unsigned long long)v, __builtin_popcountll(v)!=p?"WRONG WEIGHT":"DUPLICATE"); } i++; });
  printf("chase_t<%u,%u> calls=%zu expected %llu\n",n,p,i,(unsigned long long)bc(n,p)-1);
}
int main(){
  CASE("explicit", { Combinations_Binary_Chase<uint64_t,10,2> c; std::vector<std::pair<uint16_t,uint16_t>> r; c.changelist<false>(r, bc(10,2)); printf("explicit size ok=%zu\n", r.size()); });
  CASE("default", { Combinations_Binary_Chase<uint64_t,10,2> c; std::vector<std::pair<uint16_t,uint16_t>> r; printf("before call cap=%zu\n", r.capacity()); c.changelist<false>(r); printf("default size=%zu\n", r.size()); });
  CASE("6,3", dump_chase_t<6,3>());
  CASE("7,3", dump_chase_t<7,3>());
  CASE("4,3", dump_chase_t<4,3>());
}
