#include <cstdio>
#include "simd/simd.h"
#include "algorithm/bits/popcount.h"
int main(){
  using S = TxN_t<uint8_t,32>;
  uint8_t a[32]; for(int i=0;i<32;i++)a[i]=1;
  auto d=S::load(a); auto t=S::set1(1);
  auto s = d==t;
  printf("type=%s val=%llx pop=%d eq=%x\n", __PRETTY_FUNCTION__, (unsigned long long)s, (int)cryptanalysislib::popcount::popcount(s), S::eq(d,t));
  using S16 = TxN_t<uint16_t,16>; uint16_t b[16]; for(int i=0;i<16;i++)b[i]=1;
  auto s2 = S16::load(b)==S16::set1(1); printf("u16 val=%llx\n",(unsigned long long)s2);
  using S64 = TxN_t<uint64_t,4>; uint64_t c[4]={1,1,1,1};
  auto s3 = S64::load(c)==S64::set1(1); printf("u64 val=%llx\n",(unsigned long long)s3);
}
