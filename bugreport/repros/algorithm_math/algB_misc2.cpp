#include <cstdint>
#include <cstdio>
#include <vector>
#include <list>
#include <set>
#include <algorithm>
#include <numeric>
#include "forkB.h"
#ifdef T_SETU
#include "algorithm/set.h"
#endif
#ifdef T_PART
#include "algorithm/partition.h"
#endif
#ifdef T_I2W
#include "algorithm/int2weight.h"
#endif
#ifdef T_RI
#include "algorithm/random_index.h"
#endif
#if defined(T_HIST)||defined(T_HISTPAR)
#include "algorithm/histogram.h"
#endif
#ifdef T_TR
#include "algorithm/transform.h"
#endif
#ifdef T_SHIFT
#include "algorithm/shift.h"
#endif
int main(){
#ifdef T_SETU
  std::vector<uint32_t> a{1,2,3}, b{2}, o(3);
  cryptanalysislib::set_difference(a.begin(),a.end(),b.begin(),b.end(),o.begin());
#endif
#ifdef T_PART
  std::vector<int> a{1,2,3,4}; cryptanalysislib::partition(a.begin(),a.end(),[](int x){return x&1;});
#endif
#ifdef T_I2W
  for (uint32_t n=1;n<=12;n++) for(uint32_t w=1;w<=n;w++){ std::set<std::vector<uint16_t>> seen; uint64_t N=bc(n,w); int bad=0;
    for(uint64_t i=0;i<N;i++){ std::vector<uint16_t> wts(w,0xffff); int2weights<uint16_t,uint64_t>(wts.data(),i,n,w,w); std::vector<uint16_t> s=wts; std::sort(s.begin(),s.end());
      bool ok = std::adjacent_find(s.begin(),s.end())==s.end() && s.back()<n; if(!ok) bad++; seen.insert(s);
      }
    if(bad || seen.size()!=N) printf("int2weights n=%u w=%u: bad=%d distinct=%zu expected %llu\n",n,w,bad,seen.size(),(unsigned long long)N); }
  puts("int2weights done");
#endif
#ifdef T_I2WV
  std::vector<uint16_t> v(2); int2weights(v, 3ull, 5u, 2u);
#endif
#ifdef T_RI
  CASE("ri_small", uint32_t d[6]; generate_random_indices<uint32_t>(d, 6, 5); printf("returned:"); for(auto x:d)printf(" %u",x); puts(""));
  CASE("ri_eq", uint32_t d[5]; generate_random_indices<uint32_t>(d, 5, 5); printf("returned(5,5):"); for(auto x:d)printf(" %u",x); puts(""));
  CASE("ri_min", uint32_t d[4]; generate_random_indices<uint32_t>(d, 4, 10, 8); printf("returned(len4,[8,10)):"); for(auto x:d)printf(" %u",x); puts(""));
  CASE("ri_range", int bad=0; for(int it=0;it<200;it++){ uint32_t d[8]; generate_random_indices<uint32_t>(d, 8, 40, 20); for(auto x:d) if(x<20||x>=40) bad++; std::sort(d,d+8); if(std::adjacent_find(d,d+8)!=d+8) bad++;} printf("ri_range bad=%d\n",bad));
#endif
#ifdef T_HIST
  for (size_t n : {0,1,3,4,5,7,8,9,63,64,65,1000}) { std::vector<uint8_t> in(n); for(size_t i=0;i<n;i++) in[i]=(uint8_t)(i*37+5);
    uint32_t ref[256]={0}; for(auto x:in) ref[x]++;
    uint32_t c1[256]={0}, c4[256]={0}, c8[256]={0}, cg[256]={0};
    histogram_u8_1x(c1,in.data(),n); histogram_u8_4x(c4,in.data(),n); histogram_u8_8x(c8,in.data(),n); cryptanalysislib::algorithm::histogram(cg,in.data(),n);
    if (memcmp(c1,ref,1024)||memcmp(c4,ref,1024)||memcmp(c8,ref,1024)||memcmp(cg,ref,1024)) printf("hist n=%zu mismatch\n",n); }
  // u16 input generic
  { std::vector<uint16_t> in{1,2,300,300}; std::vector<uint32_t> c(65536,0); cryptanalysislib::algorithm::histogram<uint16_t,uint32_t>(c.data(),in.data(),in.size()); printf("hist u16 c[300]=%u exp 2\n",c[300]); }
  // accumulate semantics: 1x accumulates, 4x overwrites
  { std::vector<uint8_t> in{1,1}; uint32_t c1[256]={0}, c4[256]={0}; histogram_u8_1x(c1,in.data(),2); histogram_u8_1x(c1,in.data(),2); histogram_u8_4x(c4,in.data(),2); histogram_u8_4x(c4,in.data(),2); printf("hist twice: 1x c[1]=%u 4x c[1]=%u\n",c1[1],c4[1]); }
#ifdef T_HISTPAR
  CASE("hist_par", size_t n=1<<20; std::vector<uint8_t> in(n+7); for(size_t i=0;i<in.size();i++) in[i]=(uint8_t)(i*37+5); uint32_t ref[256]={0}; for(auto x:in) ref[x]++;
       std::vector<uint32_t> c(256,0); cryptanalysislib::algorithm::histogram(cryptanalysislib::par_if(true), c.data(), in.data(), in.size());
       int bad=0; for(int i=0;i<256;i++) if(c[i]!=ref[i]) { if(bad++<3) printf("hist par c[%d]=%u exp %u\n",i,c[i],ref[i]); } printf("hist par bad=%d\n",bad));
#endif
#endif
#ifdef T_TR
  CASE("tr_par", std::vector<int> a(100000,1), b(100000,1); auto r=cryptanalysislib::transform_reduce(cryptanalysislib::par_if(true),a.begin(),a.end(),b.begin(),5); auto e=std::transform_reduce(a.begin(),a.end(),b.begin(),5); printf("transform_reduce par = %d expected %d\n",r,e));
  CASE("tr_seq", std::vector<int> a(100,1), b(100,2); auto r=cryptanalysislib::transform_reduce(a.begin(),a.end(),b.begin(),5); printf("transform_reduce seq = %d expected 205\n",r));
#endif
#ifdef T_TRU
  std::vector<int> a(10,1); auto r=cryptanalysislib::transform_reduce(a.begin(),a.end(),0,std::plus<int>(),[](int x){return 2*x;});
#endif
#ifdef T_SHIFT
  { std::list<int> l{1,2,3,4,5}; auto r=cryptanalysislib::shift(l.begin(),l.end(),-2); printf("shift(list,-2):"); for(int x:l)printf(" %d",x); printf(" ; ret->%d (std::shift_right gives _ _ 1 2 3, ret->1)\n", *r); }
  { std::vector<int> v{1,2,3,4,5}; auto r=cryptanalysislib::shift(v.begin(),v.end(),-2); printf("shift(vec,-2) ret idx %ld (expected 2)\n",(long)(r-v.begin())); }
#endif
}
