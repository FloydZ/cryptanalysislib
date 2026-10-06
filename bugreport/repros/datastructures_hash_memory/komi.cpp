#include <initializer_list>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include "hash/komihash.h"
#include "ref/kref.h"
int main(){
  const char* ks[]={"This is a 32-byte testing string","The cat is out of the bag","A 16-byte string","The new string","7 chars"};
  for (auto k: ks) printf("%016llx %016llx\n",(unsigned long long)komihash(k,strlen(k),0),(unsigned long long)komihash_ref(k,strlen(k),0));
  static uint8_t b[4096]; for(int i=0;i<4096;i++) b[i]=i*13+5;
  int f=0; for (size_t n=0;n<1500;n++) for (uint64_t s: {0ull,1ull,0x0123456789abcdefull}) if (komihash(b,n,s)!=komihash_ref(b,n,s)) {if(f<5)printf("mismatch n=%zu\n",n); f++;}
  printf("fails %d\n",f);
}
