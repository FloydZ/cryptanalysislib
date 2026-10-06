#include "simd/simd.h"
#include <cstdio>
#define CHECK(name, got, exp) do { auto _g=(uint64_t)(got); auto _e=(uint64_t)(exp); printf("%-44s got=0x%llx exp=0x%llx %s\n", name, (unsigned long long)_g, (unsigned long long)_e, _g==_e?"OK":"BUG"); } while(0)
using namespace cryptanalysislib;
int main() {
  // _uint8x16_t::srli shifts left
  { _uint8x16_t a = _uint8x16_t::set1(0x10);
    CHECK("_uint8x16_t::srli(0x10,4)", _uint8x16_t::srli(a,4).d[0], 0x01);
    CHECK("_uint8x16_t operator>> (0x10>>1)", (a>>1).d[0], 0x08); }
  // _uint8x16_t::reverse reverses bits, not bytes
  { _uint8x16_t a; for (int i=0;i<16;i++) a.d[i]=i;
    _uint8x16_t r = _uint8x16_t::reverse(a);
    CHECK("_uint8x16_t::reverse([0..15])[0]", r.d[0], 15);
    CHECK("_uint8x16_t::reverse([0..15])[1]", r.d[1], 14); }
  // _uint8x16_t::cmp_ always zero
  { _uint8x16_t a = _uint8x16_t::set1(1), b = _uint8x16_t::set1(0);
    CHECK("_uint8x16_t::cmp(1,0) (nonzero expected)", _uint8x16_t::cmp(a,b), 0xffff); }
  // _uint8x16_t::scatter only writes 8 of 16 lanes
  { uint8_t buf[16] = {0}; _uint8x16_t off, val;
    for (int i=0;i<16;i++){off.d[i]=i; val.d[i]=0xAA;}
    _uint8x16_t::scatter(buf, off, val);
    CHECK("_uint8x16_t::scatter buf[15]", buf[15], 0xAA); }
  // _uint16x8_t operator= from _uint8x16_t does not assign
  { _uint8x16_t a = _uint8x16_t::set1(0x5a); _uint16x8_t b = _uint16x8_t::set1(0);
    b = a;
    CHECK("_uint16x8_t = _uint8x16_t (v64[0])", b.v64[0], 0x5a5a5a5a5a5a5a5aull); }
  return 0;
}
