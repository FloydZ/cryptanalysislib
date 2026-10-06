#include "simd/simd.h"
#include "simd/swar.h"
int main(){ cryptanalysislib::swar::swar<uint8_t,4> s{1}; return s[0]-1; }
