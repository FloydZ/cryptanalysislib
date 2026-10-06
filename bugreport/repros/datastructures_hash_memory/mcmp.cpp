#include <cstdio>
#include <cstring>
#include <cstdint>
#include "memory/memcmp.h"
int main(){ uint8_t a[100], b[100]; for(int i=0;i<100;i++) a[i]=b[i]=i;
  int f=0; for (size_t n=0;n<=70;n++){ bool lib=cryptanalysislib::memcmp(a,b,n); bool ref = std::memcmp(a,b,n)!=0; if(lib!=ref){ if(f<6) printf("equal buffers n=%zu: lib=%d std!=0:%d\n",n,lib,ref); f++;} }
  b[40]^=1; for (size_t n=41;n<=70;n++){ bool lib=cryptanalysislib::memcmp(a,b,n); if(!lib){ if(f<12) printf("differing buffers n=%zu: lib says equal\n",n); f++;} }
  printf("fails=%d\n",f);}
