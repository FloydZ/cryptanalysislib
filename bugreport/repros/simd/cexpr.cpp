#include "simd/simd.h"
#include <cstdio>
constexpr uint8x32_t cA = uint8x32_t::set1(0xFF);
constexpr uint8x32_t cR = uint8x32_t::srli(cA, 2);      // constexpr path
constexpr uint32x8_t cB = uint32x8_t::set1(0xFFFFFFFFu);
constexpr uint32x8_t cS = uint32x8_t::srli(cB, 8);
int main() {
  const uint8x32_t rA = uint8x32_t::set1(0xFF);
  const uint8x32_t rR = uint8x32_t::srli(rA, 2);          // runtime path
  printf("uint8x32_t::srli(0xFF,2): constexpr=0x%x runtime=0x%x exp=0x3f %s\n", cR.v128[0][0], rR.d[0], cR.v128[0][0]==0x3f?"OK":"BUG");
  printf("uint32x8_t::srli(0xFFFFFFFF,8): constexpr=0x%x exp=0xffffff %s\n", cS.v128[0][0], cS.v128[0][0]==0xffffff?"OK":"BUG");
}
