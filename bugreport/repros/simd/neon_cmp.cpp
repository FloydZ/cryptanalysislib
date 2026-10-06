#include "simd/simd.h"
#include <cstdio>
#define CHECK(name, got, exp) do { auto _g=(uint64_t)(got); auto _e=(uint64_t)(exp); printf("%-40s got=0x%llx exp=0x%llx %s\n", name, (unsigned long long)_g, (unsigned long long)_e, _g==_e?"OK":"BUG"); } while(0)
int main() {
  // uint32x8_t::move
  { uint32x8_t a = uint32x8_t::set1(0); for (int i=0;i<8;i++) a.d[i]=0x80000000u;
    CHECK("uint32x8_t::move(all msb)", uint32x8_t::move(a), 0xff); }
  // uint64x4_t::move
  { uint64x4_t a = uint64x4_t::set1(-1ull);
    CHECK("uint64x4_t::move(all msb)", uint64x4_t::move(a), 0xf); }
  // uint64x4_t::eq
  { uint64x4_t a = uint64x4_t::set1(5), b = uint64x4_t::set1(5);
    CHECK("uint64x4_t::eq(equal)", uint64x4_t::eq(a,b), 0xf);
    CHECK("operator== uint64x4_t", (a==b), 0xf); }
  // uint64x4_t::srli
  { uint64x4_t a = uint64x4_t::set1(8);
    CHECK("uint64x4_t::srli(8,1)", uint64x4_t::srli(a,1).d[0], 4);
    CHECK("operator>> uint64x4_t (8>>2)", (a>>2).d[0], 2); }
  // uint32x8_t popcnt
  { uint32x8_t a = uint32x8_t::set1(0xffffffffu);
    CHECK("uint32x8_t::popcnt(0xffffffff)", uint32x8_t::popcnt(a).d[0], 32);
    uint32x8_t b = uint32x8_t::set1(0x00010001u);
    CHECK("uint32x8_t::popcnt(0x00010001)", uint32x8_t::popcnt(b).d[0], 2); }
  // uint64x4_t popcnt
  { uint64x4_t a = uint64x4_t::set1(-1ull);
    CHECK("uint64x4_t::popcnt(~0)", uint64x4_t::popcnt(a).d[0], 64);
    uint64x4_t b = uint64x4_t::set1(1ull<<40);
    CHECK("uint64x4_t::popcnt(1<<40)", uint64x4_t::popcnt(b).d[0], 1); }
  // uint16x16_t le_
  { uint16x16_t a = uint16x16_t::set1(7), b = uint16x16_t::set1(7);
    CHECK("uint16x16_t::le_(7,7)[0]", uint16x16_t::le_(a,b).d[0], 0xffff); }
  // int8x32_t signed gt
  { int8x32_t a, b; for(int i=0;i<32;i++){a.d[i]=-1;b.d[i]=1;}
    CHECK("int8x32_t::gt(-1,1)", int8x32_t::gt(a,b), 0);
    CHECK("int8x32_t::lt(-1,1)", int8x32_t::lt(a,b), 0xffffffff); }
  { int32x8_t a, b; for(int i=0;i<8;i++){a.d[i]=-1;b.d[i]=1;}
    CHECK("int32x8_t::gt(-1,1)", int32x8_t::gt(a,b), 0); }
  // cmp mask packing (uint16x16_t: 16 lanes; uint32x8_t: 8 lanes)
  { uint16x16_t a = uint16x16_t::set1(1), b = uint16x16_t::set1(0);
    CHECK("uint16x16_t::cmp(1,0) lanes!=", uint16x16_t::cmp(a,b), 0xffff); }
  { uint32x8_t a = uint32x8_t::set1(1), b = uint32x8_t::set1(0);
    CHECK("uint32x8_t::cmp(1,0)", uint32x8_t::cmp(a,b), 0xff); }
  { uint64x4_t a = uint64x4_t::set1(1), b = uint64x4_t::set1(0);
    CHECK("uint64x4_t::cmp(1,0)", uint64x4_t::cmp(a,b), 0xf); }
  // set ordering: uint8x32_t::set puts first arg in lane 31 (Intel convention, and avx2 backend)
  { uint8x32_t a = uint8x32_t::set(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
    CHECK("uint8x32_t::set first arg -> d[31]", a.d[31], 1);
    uint16x16_t b = uint16x16_t::set(1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
    CHECK("uint16x16_t::set first arg -> d[15]", b.d[15], 1);
    uint32x8_t c = uint32x8_t::set(1,0,0,0,0,0,0,0);
    CHECK("uint32x8_t::set first arg -> d[7]", c.d[7], 1);
    uint64x4_t d = uint64x4_t::set(1,0,0,0);
    CHECK("uint64x4_t::set first arg -> d[3]", d.d[3], 1); }
  return 0;
}
