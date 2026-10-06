#include <cstdio>
#include <cstdint>
#include <vector>
#include "compression/compression.h"
template<class T> void rt(T v){ uint8_t buf[16]={0}; leb128_encode<T>(buf, v); uint8_t* p=buf; T d=leb128_decode<T>(&p); printf("%zu-byte %llu -> %llu %s\n", sizeof(T), (unsigned long long)v,(unsigned long long)d, d==v?"ok":"MISMATCH"); }
int main(){ rt<uint16_t>(300); rt<uint32_t>(1u<<29); rt<uint32_t>(0xFFFFFFFFu); rt<uint64_t>(1ull<<40); rt<uint64_t>(~0ull); rt<uint8_t>(200); }
