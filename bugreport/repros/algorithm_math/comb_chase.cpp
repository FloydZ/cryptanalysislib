#include <cstdint>
#include <set>
#include <vector>
#include "fork.h"
#include "combination/chase.h"

// Combinations_Binary_Chase::left_step with writing
template<uint32_t n, uint32_t w>
void test_left_step(){
  Combinations_Binary_Chase<uint64_t,n,w> c;
  constexpr uint32_t L = (n+63)/64;
  uint64_t A[L] = {0}, B[L]={0};
  std::set<std::vector<uint64_t>> seen; size_t cnt=0, wbad=0, posbad=0;
  uint16_t p1,p2;
  bool first=true;
  while (c.left_step(A,&p1,&p2)) {
    size_t wt=0; for(auto x:A) wt+=__builtin_popcountll(x);
    if (wt!=w) wbad++;
    if(!first){ // check diff positions
      uint16_t d1=0xffff,d2=0xffff; Combinations_Binary_Chase<uint64_t,n,w>::__diff(A,B,L,&d1,&d2);
      if(!((d1==p1&&d2==p2)||(d1==p2&&d2==p1))) { if(posbad<3) printf("  n=%u w=%u step %zu: reported (%u,%u) actual (%u,%u)\n",n,w,cnt,p1,p2,d1,d2); posbad++; }
    }
    first=false;
    seen.insert(std::vector<uint64_t>(A,A+L)); cnt++;
    for(uint32_t i=0;i<L;i++)B[i]=A[i];
    if (cnt > 10*bc(n,w)+10) break;
  }
  printf("Binary_Chase<%u,%u>: steps=%zu distinct=%zu expected=%llu wrongweight=%zu posmismatch=%zu\n",n,w,cnt,seen.size(),(unsigned long long)bc(n,w),wbad,posbad);
}

template<uint32_t n, uint32_t p>
void test_chase_t(){
  chase_t<n,p> c;
  uint64_t lo = (p==0)?0:((1ull<<p)-1);
  unsigned __int128 v = lo;
  std::set<unsigned __int128> seen; seen.insert(v); size_t calls=0, wbad=0;
  c.enumerate([&](uint16_t a, uint16_t b){ v ^= ((unsigned __int128)1)<<a; v ^= ((unsigned __int128)1)<<b; calls++; 
     if (__builtin_popcountll((uint64_t)v)+__builtin_popcountll((uint64_t)(v>>64))!=p) wbad++; seen.insert(v);});
  printf("chase_t<%u,%u>: calls=%zu distinct=%zu expected=%llu list_size=%zu wrongweight=%zu\n",n,p,calls,seen.size(),(unsigned long long)bc(n,p),c.list_size(),wbad);
}

template<uint32_t n, uint32_t t>
void test_bce(){
  std::vector<std::pair<uint16_t,uint16_t>> ret;
  BinaryChaseEnumerator<n,t>::changelist(ret);
  unsigned __int128 v = (((unsigned __int128)1)<<t)-1;
  std::set<unsigned __int128> seen; seen.insert(v); size_t wbad=0;
  for(auto &pr: ret){ v ^= ((unsigned __int128)1)<<pr.first; v ^= ((unsigned __int128)1)<<pr.second; if(__builtin_popcountll((uint64_t)v)+__builtin_popcountll((uint64_t)(v>>64))!=t) wbad++; seen.insert(v);}
  printf("BinaryChaseEnumerator<%u,%u>: changes=%zu distinct=%zu expected=%llu wrongweight=%zu\n",n,t,ret.size(),seen.size(),(unsigned long long)bc(n,t),wbad);
}

int main(){
  CASE("ls 5,2", test_left_step<5,2>());
  CASE("ls 6,1", test_left_step<6,1>());
  CASE("ls 6,3", test_left_step<6,3>());
  CASE("ls 10,4", test_left_step<10,4>());
  CASE("ls 6,5", test_left_step<6,5>());
  CASE("ls 65,2", test_left_step<65,2>());
  CASE("ls 70,3", test_left_step<70,3>());
  CASE("ls 6,0", test_left_step<6,0>());
  CASE("changelist default", { Combinations_Binary_Chase<uint64_t,10,2> c; std::vector<std::pair<uint16_t,uint16_t>> r; c.changelist<false>(r); printf("changelist size=%zu\n", r.size()); });
  CASE("ct 5,1", test_chase_t<5,1>());
  CASE("ct 5,2", test_chase_t<5,2>());
  CASE("ct 6,2", test_chase_t<6,2>());
  CASE("ct 10,2", test_chase_t<10,2>());
  CASE("ct 11,2", test_chase_t<11,2>());
  CASE("ct 6,3", test_chase_t<6,3>());
  CASE("ct 7,3", test_chase_t<7,3>());
  CASE("ct 10,3", test_chase_t<10,3>());
  CASE("ct 65,2", test_chase_t<65,2>());
  CASE("ct 4,3", test_chase_t<4,3>());
  CASE("bce 4,1", test_bce<4,1>());
  CASE("bce 5,2", test_bce<5,2>());
  CASE("bce 6,2", test_bce<6,2>());
  CASE("bce 7,3", test_bce<7,3>());
  CASE("bce 10,4", test_bce<10,4>());
  CASE("bce 65,2", test_bce<65,2>());
  CASE("bce 6,5", test_bce<6,5>());
}
