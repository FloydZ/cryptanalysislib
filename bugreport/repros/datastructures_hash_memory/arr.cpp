#include <cstdio>
#include <cstdlib>
#include <array>
#include <algorithm>
#include "container/array.h"
using cryptanalysislib::const_array;
int main(){
  constexpr const_array<int,2> a{2,1}, b{1,5};
  std::array<int,2> ra{2,1}, rb{1,5};
  (void)a;(void)b;(void)ra;(void)rb;
  constexpr const_array<int,5> c{5,3,4,1,2};
  auto s = c.mergesort([](int x,int y){return x<y;});
  printf("mergesort:"); for (size_t i=0;i<5;i++) printf(" %d", s[i]); printf("\n");
#ifdef ITER
  for (auto x: c) printf("%d",x);
#endif
}
