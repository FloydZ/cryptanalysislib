#pragma once 

#include <cstdint>

/// Return word that consists of the lowest p bits of a repeated.
/// E.g.: if p==3 and a=*****xyz (8-bit), the return yzxyzxyz.
/// Must have p>0.
template<typename T>
static inline T bit_copy_periodic(T a, T p) {
    constexpr static uint32_t BITS = sizeof(T)*8;
    a &= ( ~0UL >> (BITS-p) );
    for (T s=p; s<BITS; s<<=1)  { a |= (a<<s); }
    return a;
}

/// Return word that consists of the lowest p bits of a repeated
/// in the lowest ldn bits (upper bits are zero).
/// E.g.: if p==3, ldn=7 and a=*****xyz (8-bit), the return 0zxyzxyz.
/// Must have p>0 and ldn>0.
template<typename T>
static inline T bit_copy_periodic(T a, T p, T ldn) {
    constexpr static uint32_t BITS = sizeof(T)*8;
    a &= ( ~0UL >> (BITS-p) );
    for (T s=p; s<ldn; s<<=1)  { a |= (a<<s); }
    a &= ( ~0UL >> (BITS-ldn) );
    return a;
}
