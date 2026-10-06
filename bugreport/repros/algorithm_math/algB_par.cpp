#include <cstdint>
#include <cstdio>
#include <vector>
#include <numeric>
#include <algorithm>
#include "forkB.h"
#include "algorithm/fill.h"
#include "algorithm/copy.h"
#include "algorithm/transform.h"
int main(){
  CASE("fill_par", std::vector<uint32_t> v((1<<21)+3, 0); cryptanalysislib::fill(cryptanalysislib::par_if(true), v.begin(), v.end(), 7u); size_t bad=std::count_if(v.begin(),v.end(),[](uint32_t x){return x!=7;}); printf("fill par bad=%zu\n",bad));
  CASE("copy_par", std::vector<uint32_t> s((1<<20)+3), d((1<<20)+3,0); std::iota(s.begin(),s.end(),1); cryptanalysislib::copy(cryptanalysislib::par_if(true), s.begin(), s.end(), d.begin()); printf("copy par ok=%d\n", s==d));
  CASE("tr_par_small", std::vector<int> a(2000,1), b(2000,1); auto r=cryptanalysislib::transform_reduce(cryptanalysislib::par_if(true),a.begin(),a.end(),b.begin(),5); printf("transform_reduce par(2000) = %d expected 2005\n",r));
}
