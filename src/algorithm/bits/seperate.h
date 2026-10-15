#pragma once 

#include <cstdint>


// Return  word with bits of w separated as indicated by m.
// Bits at positions where m is 0/1 are moved to the low/high end.
// Examples:
//  w = ABCDEFGH
//  m = 00111100
//  ==> CDEFABGH
//
//  w = ABCDEFGH
//  m = 01010101
//  ==> BDFHACEG
//
// s must contain the number of ones set in m.
// For the default s exactly half of the bits must be set in m
static inline uint64_t bit_separate(uint64_t w, uint64_t m, uint64_t s=BITS_PER_LONG/2)
{
    uint64_t a0 = bit_gather(w, ~m);
    uint64_t a1 = bit_gather(w,  m);
    return  (a0 ^ (a1<<s));
}
// -------------------------


// Bit-separate all length-s subwords of w.
// s>=2 must divide BITS_PER_LONG.
// The mask must be such that exactly half of the bits
//   in each subword are set.
// Examples:
//  s = 4
//  w = ABCD EFGH
//  m = 0011 1010
//  ==> CDAB EGFH
//
//  s = 2
//  w = AB CD EF GH
//  m = 01 10 01 10
//  ==> BA CD FE GH
//
// For s==BITS_PER_LONG the result is as with bit_separate(w, m)
//  if half of the bits of m are set.
static inline uint64_t bit_separate_subwords(uint64_t w, uint64_t m, uint64_t s)
{
    uint64_t swm = ~0UL >> ( BITS_PER_LONG - s );  // sub word mask
    swm <<= ( BITS_PER_LONG - s );  // at high end
    uint64_t h = s/2;
    uint64_t a = 0;  // return
    do
    {
        a <<= s;
        uint64_t m1 = m  & swm;
        uint64_t m0 = m1 ^ swm;
        uint64_t a0 = bit_gather(w, m0);
        uint64_t a1 = bit_gather(w, m1);
        a |= (a0 ^ (a1<<h));
    }
    while ( (swm>>=s) );
    return  a;
}
