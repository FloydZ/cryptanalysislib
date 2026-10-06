#include <vector>
#include <cstdio>
#include <cstdint>
#include "algorithm/reduce.h"
#include "algorithm/count.h"
#include "algorithm/find.h"
int main(){
  setvbuf(stdout,0,_IONBF,0);
  printf("threads=%u\n", cryptanalysislib::par_if(true).pool()->get_num_threads());
  std::vector<uint32_t> v(1u<<20, 1);
  auto g = cryptanalysislib::reduce(cryptanalysislib::par_if(true), v.begin(), v.end(), (uint32_t)100);
  printf("par reduce(2^20 ones, init=100) = %u expected %u\n", g, (1u<<20)+100);
  auto c = cryptanalysislib::count(cryptanalysislib::par_if(true), v.begin(), v.end(), 1u);
  printf("par count = %ld expected %u\n", (long)c, 1u<<20);
  std::vector<uint32_t> w(1u<<22, 0); w[3000000]=7; w[3500000]=7;
  auto it = cryptanalysislib::find(cryptanalysislib::par_if(true), w.begin(), w.end(), 7u);
  printf("par find idx = %ld expected 3000000\n", (long)(it-w.begin()));
  return 0;
}
