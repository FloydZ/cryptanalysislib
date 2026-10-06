#include <cstdint>
#include <cstdio>
#include <vector>
#include <algorithm>
#include "forkB.h"
#include "algorithm/search.h"
using V = std::vector<int>;
int main(){
  V a{1,2,3,1,2,4}, s{1,2,4};
  auto r=cryptanalysislib::search(a.begin(),a.end(),s.begin(),s.end()); printf("search idx %ld exp 3\n",(long)(r-a.begin()));
  V b{1,2,2,3,2,2,2}; auto r2=cryptanalysislib::search_n(b.begin(),b.end(),3,2); printf("search_n idx %ld exp 4\n",(long)(r2-b.begin()));
#ifdef T_PRED
  auto r3=cryptanalysislib::search(a.begin(),a.end(),s.begin(),s.end(),[](int x,int y){return x==y;});
#endif
#ifdef T_PREDN
  auto r4=cryptanalysislib::search_n(b.begin(),b.end(),3,2,[](int x,int y){return x==y;});
#endif
#ifdef T_SIMD
  std::vector<uint32_t> d(64,0); for(int i=20;i<24;i++) d[i]=7;
  size_t k = cryptanalysislib::internal::search_n_uXX_simd<uint32_t>(d.data(), d.size(), 7u);
  printf("search_n_uXX_simd(len 64, run of 4x 7 at 20) = %zu\n", k);
#endif
}
