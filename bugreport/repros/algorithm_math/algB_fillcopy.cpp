#include <cstdint>
#include <cstdio>
#include <vector>
#include <span>
#include <cstring>
#include <algorithm>
#include "forkB.h"
#include "guard.h"
#include "algorithm/fill.h"
#include "algorithm/copy.h"
static const size_t sizes[] = {0,1,2,3,7,8,9,15,16,17,31,32,33,47,48,63,64,65,95,96,97,127,128,129,1000};
template<typename T> void test_fill(const char* tn){
  int bad=0;
  for (size_t n : sizes) for (int where=0; where<2; where++) for (size_t off=0; off<4; off++) {
    T* p = where==0 ? guard_end<T>(n+off) + off : guard_begin<T>(n+off+8) + off;
    for (size_t i=0;i<n;i++) p[i]=(T)0x5a;
    T v = (T)-3;
    std::vector<T> vv(p,p+n); // unused
    // use a vector-like iterator? fill requires Iterator::value_type -> use std::span? use vector iter on raw memory isn't possible; use std::span
    std::span<T> sp(p,n);
    CASE("fill", cryptanalysislib::fill(sp.begin(), sp.end(), v);
      for(size_t i=0;i<n;i++) if(p[i]!=v){ printf("fill<%s> n=%zu off=%zu where=%d: p[%zu]=%lld expected %lld\n",tn,n,off,where,i,(long long)p[i],(long long)v); _exit(3);} );
  }
}
template<typename T> void test_copy(const char* tn){
  for (size_t n : sizes) for (int where=0; where<2; where++) for (size_t soff=0; soff<4; soff++) for (size_t doff=0; doff<4; doff++) {
    T* s = where==0 ? guard_end<T>(n+soff)+soff : guard_begin<T>(n+soff+8)+soff;
    T* d = where==0 ? guard_end<T>(n+doff)+doff : guard_begin<T>(n+doff+8)+doff;
    for (size_t i=0;i<n;i++){ s[i]=(T)(i*7+1); d[i]=0; }
    std::span<T> ss(s,n), ds(d,n);
    CASE("copy", auto r = cryptanalysislib::copy(ss.begin(), ss.end(), ds.begin());
      if (r != ds.end()) { printf("copy<%s> n=%zu bad return\n",tn,n); _exit(4);}
      for(size_t i=0;i<n;i++) if(d[i]!=s[i]){ printf("copy<%s> n=%zu soff=%zu doff=%zu where=%d: d[%zu]=%lld expected %lld\n",tn,n,soff,doff,where,i,(long long)d[i],(long long)s[i]); _exit(3);} );
  }
}
int main(){
  test_fill<uint8_t>("u8"); test_fill<int8_t>("i8"); test_fill<uint16_t>("u16"); test_fill<int32_t>("i32"); test_fill<uint32_t>("u32"); test_fill<uint64_t>("u64"); test_fill<int64_t>("i64");
  puts("fill done");
  test_copy<uint8_t>("u8"); test_copy<uint16_t>("u16"); test_copy<uint32_t>("u32"); test_copy<uint64_t>("u64");
  puts("copy done");
}
