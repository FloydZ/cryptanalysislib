#include <cstdint>
#include <cstdio>
#include <cstring>
#include <span>
#include "algorithm/copy.h"
int main(){
  alignas(64) static uint8_t src[512], dst[512];
  int bad=0;
  for (int mis=0; mis<32; mis++) for (int n=0; n<=300; n++) {
    for (int i=0;i<512;i++){ src[i]=(uint8_t)(i*13+1); dst[i]=0; }
    std::span<uint8_t> s(src+5, n), d(dst+mis, n);
    cryptanalysislib::copy(s.begin(), s.end(), d.begin());
    int first=-1; for(int i=0;i<n;i++) if(dst[mis+i]!=src[5+i]){first=i;break;}
    bool over=false; for(int i=0;i<512;i++) if((i<mis||i>=mis+n) && dst[i]) over=true;
    if(first>=0||over){ if(bad<6) printf("copy<u8> n=%d dst%%32=%d: first wrong idx=%d overwrite_outside=%d\n",n,mis,first,over); bad++; }
  }
  printf("total bad=%d\n",bad);
  // minimal: 8 x uint64, destination misaligned by 24 bytes from a 32B boundary
  alignas(32) static uint64_t s64[8], d64[16];
  for(int i=0;i<8;i++) s64[i]=i+1;
  std::span<uint64_t> a(s64,8), b(d64+3,8);
  cryptanalysislib::copy(a.begin(), a.end(), b.begin());
  printf("u64 copy n=8 dst+24B: got"); for(int i=0;i<8;i++) printf(" %llu",(unsigned long long)d64[3+i]); printf("  (expected 1..8)\n");
}
