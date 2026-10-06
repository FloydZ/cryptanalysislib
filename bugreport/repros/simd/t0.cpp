#include "simd/simd.h"
int main(){ uint32x8_t a = uint32x8_t::set1(1); return a.d[0]-1; }
