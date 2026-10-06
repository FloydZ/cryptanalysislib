#include <cstdio>
#include <cstring>
#include <cstdint>
#include "memory/memory.h"
int main(){
  int fails=0;
  // memcpy
  alignas(64) static uint8_t src[512], dst[512], ref[512];
  for (int i=0;i<512;i++) src[i]=(uint8_t)(i*7+1);
  for (size_t off=0; off<32; off++) for (size_t n=0;n<200;n++){
    memset(dst,0,512); memset(ref,0,512);
    cryptanalysislib::memcpy(dst+off, src+3, n);
    std::memcpy(ref+off, src+3, n);
    if (std::memcmp(dst,ref,512)) { if (fails<5) printf("memcpy mismatch off=%zu n=%zu\n",off,n); fails++; }
  }
  printf("memcpy fails=%d\n",fails);
  // memset
  int f2=0;
  { alignas(64) uint16_t a[64], r[64];
    for (size_t n=0;n<40;n++) for(size_t off=0;off<4;off++){ for(int i=0;i<64;i++){a[i]=0;r[i]=0;}
      cryptanalysislib::memset(a+off,(uint16_t)0xABCD,n); for(size_t i=0;i<n;i++) r[off+i]=0xABCD;
      if(std::memcmp(a,r,sizeof a)){ if(f2<5) printf("memset u16 mismatch n=%zu off=%zu\n",n,off); f2++;} } }
  { alignas(64) uint32_t a[64], r[64];
    for (size_t n=0;n<40;n++) for(size_t off=0;off<4;off++){ for(int i=0;i<64;i++){a[i]=0;r[i]=0;}
      cryptanalysislib::memset(a+off,(uint32_t)0xABCDEF12,n); for(size_t i=0;i<n;i++) r[off+i]=0xABCDEF12;
      if(std::memcmp(a,r,sizeof a)){ if(f2<10) printf("memset u32 mismatch n=%zu off=%zu\n",n,off); f2++;} } }
  { alignas(64) uint64_t a[64], r[64];
    for (size_t n=0;n<40;n++) for(size_t off=0;off<4;off++){ for(int i=0;i<64;i++){a[i]=0;r[i]=0;}
      cryptanalysislib::memset(a+off,(uint64_t)0xABCDEF1234567ull,n); for(size_t i=0;i<n;i++) r[off+i]=0xABCDEF1234567ull;
      if(std::memcmp(a,r,sizeof a)){ if(f2<15) printf("memset u64 mismatch n=%zu off=%zu\n",n,off); f2++;} } }
  { alignas(64) uint8_t a[256], r[256];
    for (size_t n=0;n<150;n++) for(size_t off=0;off<33;off++){ memset(a,0,256);memset(r,0,256);
      cryptanalysislib::memset(a+off,(uint8_t)0xA7,n); memset(r+off,0xA7,n);
      if(std::memcmp(a,r,sizeof a)){ if(f2<20) printf("memset u8 mismatch n=%zu off=%zu\n",n,off); f2++;} } }
  printf("memset fails=%d\n",f2);
}
