#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include "forkB.h"
#include "algorithm/rotate.h"
using namespace cryptanalysislib;
template<typename T, typename F> void run(const char* name, F f){
  int bad=0;
  for (size_t n=0;n<=70;n++) for(size_t l=0;l<=n;l++){
    std::vector<T> a(n+4), b; for(size_t i=0;i<n+4;i++) a[i]=(T)(i+1); b=a;
    f(a.data(), l, n-l);
    std::rotate(b.begin(), b.begin()+l, b.begin()+n);
    if (a!=b){ if(bad++<2) printf("%s<%zu-byte> n=%zu left=%zu right=%zu mismatch\n",name,sizeof(T),n,l,n-l);}
  }
  printf("%s<%zu-byte> bad=%d\n",name,sizeof(T),bad);
}
#define R(fn) CASE(#fn "_i32", run<int32_t>(#fn,[](int32_t*p,size_t l,size_t r){fn(p,l,r);})); CASE(#fn "_u64", run<uint64_t>(#fn,[](uint64_t*p,size_t l,size_t r){fn(p,l,r);})); CASE(#fn "_u8", run<uint8_t>(#fn,[](uint8_t*p,size_t l,size_t r){fn(p,l,r);}));
int main(){
#ifdef T_CONTREV
  R(contrev_rotation)
#endif
#ifdef T_TRINITY
  R((trinity_rotation<std::remove_pointer_t<decltype(p)>,8>))
#endif
#ifdef T_HELIX
  R(helix_rotation)
#endif
#ifdef T_DRILL
  R(drill_rotation)
#endif
#ifdef T_GRAIL
  R(grail_rotation)
#endif
#ifdef T_PISTON
  R(piston_rotation)
#endif
#ifdef T_GRIES
  R(griesmills_rotation)
#endif
}
