#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include ADLER
static uint8_t data[20000];
int main(){ for (int i=0;i<20000;i++) data[i]=(uint8_t)((i*131+7)&0xff);
 size_t ns[]={37,64,100,300,1000,5552,5553,20000}; for (size_t n: ns) { printf("%zu 0x%x", n, adler32(1,data,n));
#ifdef USE_AVX2
 printf(" avx2=0x%x", avx2_adler32(1,data,n));
#endif
 printf("\n"); } }
