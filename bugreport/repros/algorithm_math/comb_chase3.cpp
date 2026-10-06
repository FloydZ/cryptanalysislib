#include <cstdint>
#include <cstdio>
#include "combination/chase.h"
int main(){
  chase_t<4,3> c; uint64_t v=7;
  c.enumerate([&](uint16_t a, uint16_t b){ v^=1ull<<a; v^=1ull<<b; printf("(%u,%u) -> %llx\n",a,b,(unsigned long long)v);});
  chase_t<7,3> d; v=7; int i=0;
  d.enumerate([&](uint16_t a, uint16_t b){ v^=1ull<<a; v^=1ull<<b; if(i++<45) printf("(%u,%u) -> %02llx w=%d\n",a,b,(unsigned long long)v,__builtin_popcountll(v));});
}
