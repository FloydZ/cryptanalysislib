#include <cstdio>
#include "simd/simd.h"
int main(){
  uint32_t a[8]={0xffffffff,0,0,0,0xffffffff,0,0,0};
  printf("u32x8 movemask lanes0,4 -> %x (expect 11)\n", uint32x8_t::move(uint32x8_t::load(a)));
  uint64_t b[4]={0,0,~0ull,0};
  printf("u64x4 movemask lane2 -> %x (expect 4)\n", uint64x4_t::move(uint64x4_t::load(b)));
  uint64_t c[4]={0,0,7,0}, z[4]={1,1,7,1};
  printf("u64x4 eq lane2 -> %x (expect 4)\n", uint64x4_t::eq(uint64x4_t::load(c),uint64x4_t::load(z)));
}
