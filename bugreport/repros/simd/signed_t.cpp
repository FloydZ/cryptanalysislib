#include "simd/simd.h"
int main(){ auto a = int8x32_t::set1(1); auto b = int16x16_t::set1(1); auto c = int32x8_t::set1(1); return a.d[0]+b.d[0]+c.d[0]-3; }
