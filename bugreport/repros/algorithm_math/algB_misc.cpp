#include <cstdint>
#include <cstdio>
#include <vector>
#include <span>
#include <algorithm>
#include <numeric>
#include "forkB.h"
#include "algorithm/reverse.h"
#include "algorithm/shift.h"
#include "algorithm/merge.h"
#include "algorithm/remove.h"
#include "algorithm/replace.h"
#include "algorithm/set.h"
#include "algorithm/all_of.h"
#include "algorithm/rotate.h"
using V = std::vector<int>;
static void pr(const char*n,const V&v){printf("%s=[",n);for(auto x:v)printf("%d,",x);printf("]\n");}
int main(){
  // shift_right return value
  CASE("shift_right", V a{1,2,3,4,5}; V b=a; auto r1=cryptanalysislib::shift_right(a.begin(),a.end(),2); auto r2=std::shift_right(b.begin(),b.end(),2);
       printf("shift_right ret idx %ld expected %ld; tail ok=%d\n",(long)(r1-a.begin()),(long)(r2-b.begin()), std::equal(a.begin()+2,a.end(),b.begin()+2)));
  CASE("shift_neg", V a{1,2,3,4,5}; auto r=cryptanalysislib::shift(a.begin(),a.end(),-2); pr("shift(-2) [1..5]",a); printf(" ret idx %ld (expected std::shift_right equivalent: [?,?,1,2,3], ret 2)\n",(long)(r-a.begin())));
  CASE("shift_pos", V a{1,2,3,4,5}; auto r=cryptanalysislib::shift(a.begin(),a.end(),2); pr("shift(+2) [1..5]",a); printf(" ret idx %ld\n",(long)(r-a.begin())));
  CASE("shift_left", V a{1,2,3,4,5}; V b=a; auto r1=cryptanalysislib::shift_left(a.begin(),a.end(),2); auto r2=std::shift_left(b.begin(),b.end(),2); printf("shift_left ret %ld exp %ld\n",(long)(r1-a.begin()),(long)(r2-b.begin())));
  // reverse
  CASE("reverse", for(int n=0;n<70;n++){ V a(n); std::iota(a.begin(),a.end(),0); V b=a; ::reverse(a.begin(),a.end()); std::reverse(b.begin(),b.end()); if(a!=b){printf("reverse n=%d bad\n",n);} });
  // merge
  CASE("merge", V a{1,3,5,7}, b{2,3,4,8,9}; V o(9), e(9); cryptanalysislib::merge(a.begin(),a.end(),b.begin(),b.end(),o.begin()); std::merge(a.begin(),a.end(),b.begin(),b.end(),e.begin()); if(o!=e) pr("merge bad",o); else puts("merge ok"));
  // remove/replace
  CASE("remove", V a{1,2,1,3,1,4}; V b=a; auto r1=cryptanalysislib::remove(a.begin(),a.end(),1); auto r2=std::remove(b.begin(),b.end(),1); if((r1-a.begin())!=(r2-b.begin())||!std::equal(a.begin(),r1,b.begin())) puts("remove bad"); else puts("remove ok"));
  CASE("remove_if", V a{1,2,1,3,1,4}; V b=a; auto p=[](int x){return x==1;}; auto r1=cryptanalysislib::remove_if(a.begin(),a.end(),p); auto r2=std::remove_if(b.begin(),b.end(),p); if((r1-a.begin())!=(r2-b.begin())||!std::equal(a.begin(),r1,b.begin())) puts("remove_if bad"); else puts("remove_if ok"));
  // set ops vs std, unsigned SIMD path
  CASE("set_difference", int bad=0; srand(1); for(int it=0;it<3000;it++){ int n1=rand()%80, n2=rand()%80; std::vector<int32_t> a(n1), b(n2); for(auto&x:a)x=rand()%60; for(auto&x:b)x=rand()%60; std::sort(a.begin(),a.end()); std::sort(b.begin(),b.end());
       std::vector<uint32_t> o(n1+1), e(n1+1); auto r1=cryptanalysislib::set_difference(a.begin(),a.end(),b.begin(),b.end(),o.begin()); auto r2=std::set_difference(a.begin(),a.end(),b.begin(),b.end(),e.begin());
       if((r1-o.begin())!=(r2-e.begin())||!std::equal(o.begin(),r1,e.begin())){ if(bad++<3){printf("set_difference n1=%d n2=%d got %ld exp %ld\n",n1,n2,(long)(r1-o.begin()),(long)(r2-e.begin())); printf(" a=");for(auto x:a)printf("%u,",x);printf("\n b=");for(auto x:b)printf("%u,",x);printf("\n");}} } printf("set_difference bad=%d\n",bad));
  CASE("includes", int bad=0; srand(2); for(int it=0;it<3000;it++){ int n1=rand()%70, n2=rand()%10; std::vector<int32_t> a(n1), b(n2); for(auto&x:a)x=rand()%40; for(auto&x:b)x=rand()%40; if(it%2 && n1){ for(auto&x:b) x=a[rand()%n1]; } std::sort(a.begin(),a.end()); std::sort(b.begin(),b.end());
       bool r1=cryptanalysislib::includes(a.begin(),a.end(),b.begin(),b.end()); bool r2=std::includes(a.begin(),a.end(),b.begin(),b.end());
       if(r1!=r2){ if(bad++<3){printf("includes n1=%d n2=%d got %d exp %d\n a=",n1,n2,r1,r2);for(auto x:a)printf("%u,",x);printf("\n b=");for(auto x:b)printf("%u,",x);printf("\n");}} } printf("includes bad=%d\n",bad));
  CASE("set_sym", V a{1,2,3,5}, b{2,4,5,6}; V o(8), e(8); auto r1=cryptanalysislib::set_symmetric_difference(a.begin(),a.end(),b.begin(),b.end(),o.begin()); auto r2=std::set_symmetric_difference(a.begin(),a.end(),b.begin(),b.end(),e.begin()); printf("symdiff ok=%d\n",(r1-o.begin())==(r2-e.begin())&&std::equal(o.begin(),r1,e.begin())));
  CASE("set_union", V a{1,2,2,3,5}, b{2,4,5,5,6}; V o(10), e(10); auto r1=cryptanalysislib::set_union(a.begin(),a.end(),b.begin(),b.end(),o.begin()); auto r2=std::set_union(a.begin(),a.end(),b.begin(),b.end(),e.begin()); printf("union ok=%d\n",(r1-o.begin())==(r2-e.begin())&&std::equal(o.begin(),r1,e.begin())));
  // search
  // all_of / any_of / none_of
  CASE("allof", for(int n=0;n<70;n++){ V a(n,1); if(n) a[n-1]=0; auto p=[](const int&x){return x==1;}; bool r1=cryptanalysislib::all_of(a.begin(),a.end(),p), r2=std::all_of(a.begin(),a.end(),p); bool q1=cryptanalysislib::any_of(a.begin(),a.end(),[](const int&x){return x==0;}), q2=std::any_of(a.begin(),a.end(),[](const int&x){return x==0;}); bool z1=cryptanalysislib::none_of(a.begin(),a.end(),[](const int&x){return x==0;}), z2=std::none_of(a.begin(),a.end(),[](const int&x){return x==0;}); if(r1!=r2||q1!=q2||z1!=z2) printf("allof n=%d mismatch\n",n);} puts("allof done"));
  // rotate (std-like)
  CASE("rotate", for(int n=0;n<20;n++) for(int m=0;m<=n;m++){ V a(n); std::iota(a.begin(),a.end(),0); V b=a; auto r1=cryptanalysislib::rotate(a.data(),a.data()+m,a.data()+n); auto r2=std::rotate(b.data(),b.data()+m,b.data()+n); if(a!=b||(r1-a.data())!=(r2-b.data())) printf("rotate n=%d m=%d bad ret %ld exp %ld\n",n,m,(long)(r1-a.data()),(long)(r2-b.data()));} puts("rotate done"));
  // rotl/rotr
  CASE("rotl8", printf("rotl<uint8_t>(0x81,1)=0x%llx expected 0x3\n",(unsigned long long)cryptanalysislib::rotl<uint8_t>(0x81,1)));
  CASE("rotl0", printf("rotl<uint64_t>(5,0)=%llu expected 5\n",(unsigned long long)cryptanalysislib::rotl<uint64_t>(5,0)));
  CASE("rotr8", printf("rotr<uint8_t>(0x81,1)=0x%llx expected 0xc0\n",(unsigned long long)cryptanalysislib::rotr<uint8_t>(0x81,1)));
}
