#include <vector>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include "algorithm/argmin.h"
#include "algorithm/argmax.h"
#include "algorithm/max.h"
#include "algorithm/equal.h"
#include "algorithm/mismatch.h"
#include "algorithm/exclusive_scan.h"
#include "algorithm/accumulate.h"
using namespace cryptanalysislib;
int main(){
  setvbuf(stdout,0,_IONBF,0);
  const size_t N = 1u<<20;
  std::vector<uint32_t> v(N); for (size_t i=0;i<N;i++) v[i] = 1000 + (i*7919)%100000;
  v[777777] = 1; v[555555] = 9999999;
#ifdef XTEST_1
  printf("par argmin = %zu expected 777777\n", argmin(par_if(true), v.begin(), v.end()));
#endif
#ifdef XTEST_2
  printf("par argmax = %zu expected 555555\n", argmax(par_if(true), v.begin(), v.end()));
#endif
#ifdef XTEST_3
  printf("par max = %u expected 9999999\n", cryptanalysislib::max(par_if(true), v.begin(), v.end()));
#endif
#ifdef XTEST_4
  auto w = v; w[N-1] ^= 1;
  printf("par equal(v,v)=%d expected 1; par equal(v,w)=%d expected 0\n", (int)equal(par_if(true), v.begin(), v.end(), v.begin()), (int)equal(par_if(true), v.begin(), v.end(), w.begin()));
#endif
#ifdef XTEST_5
  auto w = v; w[123] ^= 1;
  auto p = mismatch(par_if(true), v.begin(), v.end(), w.begin());
  printf("par mismatch = %ld expected 123\n", (long)(p.first - v.begin()));
#endif
#ifdef XTEST_6
  std::vector<uint32_t> o(N), e(N); std::exclusive_scan(v.begin(), v.end(), e.begin(), 5u);
  exclusive_scan(par_if(true), v.begin(), v.end(), o.begin(), 5u);
  printf("par exclusive_scan ok=%d\n", (int)(o==e));
#endif
#ifdef XTEST_7
  printf("par accumulate = %u\n", accumulate(par_if(true), v.begin(), v.end(), 0u));
#endif
}
