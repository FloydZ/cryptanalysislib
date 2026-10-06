#include <cstdio>
#include <cstdint>
#include "container/stack.h"
int main(){ Stack<uint32_t> s(8); size_t r1=s.push(1); size_t r2=s.push(2);
  printf("[Stack cap 8] push returns %zu,%zu (doc: size after push -> 1,2); capacity()=%zu (expected 8)\n", r1, r2, s.capacity());
  for(int i=0;i<6;i++) (void)s.push(i); printf("[Stack] 9th push on full stack returns %zu (expected 0)\n", s.push(9)); }
