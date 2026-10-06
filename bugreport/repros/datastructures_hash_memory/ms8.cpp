#include <cstdio>
#include <cstring>
#include <cstdint>
#include "memory/memory.h"
int main(){ int f=0; alignas(64) uint8_t a[64], r[64];
  for (size_t n=0;n<=16;n++){ memset(a,0x11,64); memset(r,0x11,64); cryptanalysislib::memset(a,(uint8_t)0xA7,n); memset(r,0xA7,n);
    if (memcmp(a,r,64)) { printf("memset u8 n=%zu mismatch:", n); for(size_t i=0;i<n+1;i++) printf(" %02x",a[i]); printf("\n"); f++; } }
  printf("fails=%d\n",f); }
